import argparse
import math
import os
import time
from typing import Callable

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agent.encoder import FeaturesExtractor
from src.agent.flowLevelEncoder import FlowLevelFeaturesExtractor
from src.app.flowLevelDrlScheduler import clone_network
from src.env.env import NetEnv
from src.env.flowLevelEnv import FlowLevelNetEnv
from src.network.net import FlowGenerator, Network, generate_graph


def summarize_eval_results(results: list[dict]) -> dict:
    if not results:
        return {
            "num_episodes": 0,
            "success_rate": 0.0,
            "gcl_avg": 0.0,
            "gcl_max": 0,
            "episode_reward_avg": 0.0,
        }

    successes = [result for result in results if result["success"]]
    rewards = [result["episode_reward"] for result in results]
    gcl_avgs = [result["gcl_avg"] for result in successes]
    gcl_maxes = [result["gcl_max"] for result in successes]

    return {
        "num_episodes": len(results),
        "success_rate": len(successes) / len(results),
        "gcl_avg": float(np.mean(gcl_avgs)) if gcl_avgs else 0.0,
        "gcl_max": int(max(gcl_maxes)) if gcl_maxes else 0,
        "episode_reward_avg": float(np.mean(rewards)),
    }


def aggregate_experiment_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    metric_columns = [
        column for column in [
            "success_rate",
            "gcl_avg",
            "gcl_max",
            "episode_reward_avg",
            "train_time_s",
        ]
        if column in df.columns
    ]
    grouped = df.groupby(["env_kind", "train_steps"], as_index=False)
    aggregated = grouped[metric_columns].agg(["mean", "std"]).reset_index()
    aggregated.columns = [
        "_".join(filter(None, column)).rstrip("_")
        if isinstance(column, tuple) else column
        for column in aggregated.columns
    ]
    aggregated = aggregated.loc[:, aggregated.columns != "index"]
    num_seeds = df.groupby(["env_kind", "train_steps"])["seed"].nunique().reset_index()
    num_seeds = num_seeds.rename(columns={"seed": "num_seeds"})
    return aggregated.merge(num_seeds, on=["env_kind", "train_steps"])


def _gcl_stats(network: Network, schedule_res: dict) -> dict:
    gcl_lengths = []
    for link in network.links_dict.values():
        operations = schedule_res.get(link, [])
        gated_operations = [(flow, operation) for flow, operation in operations
                            if operation.gating_time is not None]
        if not gated_operations:
            gcl_lengths.append(0)
            continue

        gcl_cycle = math.lcm(*[flow.period for flow, _ in gated_operations])
        gcl_length = sum(gcl_cycle // flow.period for flow, _ in gated_operations) * 2
        gcl_lengths.append(gcl_length)

    return {
        "gcl_avg": float(np.mean(gcl_lengths)),
        "gcl_max": int(max(gcl_lengths)),
    }


def make_network(topo: str, num_flows: int, link_rate: int,
                 jitter: float, seed: int) -> Network:
    graph = generate_graph(topo, link_rate)
    flow_generator = FlowGenerator(graph, seed=seed, jitters=jitter)
    return Network(graph, flow_generator(num_flows))


def evaluate_model(model: MaskablePPO,
                   env_factory: Callable[[], NetEnv],
                   num_episodes: int) -> dict:
    results = []
    for _ in range(num_episodes):
        env = env_factory()
        obs, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        info = {"success": False}

        while not (terminated or truncated):
            action_masks = env.action_masks()
            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=action_masks,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        result = {
            "success": bool(info.get("success", False)),
            "episode_reward": episode_reward,
        }
        if result["success"]:
            result |= _gcl_stats(env.network if hasattr(env, "network") else Network(env.graph, env.flows),
                                 info["ScheduleRes"])
        results.append(result)

    return summarize_eval_results(results)


def _make_env_factory(env_kind: str, network: Network, max_hops: int = None) -> Callable[[], NetEnv]:
    def _factory():
        fresh_network = clone_network(network)
        if env_kind == "hop":
            return NetEnv(fresh_network)
        if env_kind == "flow":
            return FlowLevelNetEnv(fresh_network, max_hops=max_hops)
        raise ValueError(f"Unknown env kind: {env_kind}")

    return _factory


def policy_kwargs_for_env(env_kind: str) -> dict:
    if env_kind == "hop":
        return {"features_extractor_class": FeaturesExtractor}
    if env_kind == "flow":
        return {"features_extractor_class": FlowLevelFeaturesExtractor}
    raise ValueError(f"Unknown env kind: {env_kind}")


def train_and_evaluate(env_kind: str,
                       network: Network,
                       seed: int,
                       total_timesteps: int,
                       eval_every: int,
                       eval_episodes: int,
                       max_hops: int = None) -> list[dict]:
    env_factory = _make_env_factory(env_kind, network, max_hops=max_hops)
    vec_env = DummyVecEnv([env_factory])
    policy_kwargs = policy_kwargs_for_env(env_kind)

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    rows = []
    trained_steps = 0
    while trained_steps < total_timesteps:
        chunk = min(eval_every, total_timesteps - trained_steps)
        start = time.time()
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        train_time = time.time() - start
        trained_steps += chunk

        summary = evaluate_model(model, env_factory, eval_episodes)
        rows.append({
            "env_kind": env_kind,
            "seed": seed,
            "train_steps": trained_steps,
            "train_time_s": train_time,
            **summary,
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topo", default="L5")
    parser.add_argument("--num_flows", type=int, default=10)
    parser.add_argument("--link_rate", type=int, default=100)
    parser.add_argument("--jitter", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    rows = []
    for seed in range(args.seed, args.seed + args.num_seeds):
        network = make_network(
            args.topo,
            args.num_flows,
            args.link_rate,
            args.jitter,
            seed,
        )
        max_hops = max(len(flow.path) for flow in network.flows)
        rows.extend(train_and_evaluate(
            "hop",
            network,
            seed,
            args.total_timesteps,
            args.eval_every,
            args.eval_episodes,
        ))
        rows.extend(train_and_evaluate(
            "flow",
            network,
            seed,
            args.total_timesteps,
            args.eval_every,
            args.eval_episodes,
            max_hops=max_hops,
        ))

    df = pd.DataFrame(rows)
    summary = aggregate_experiment_rows(rows)
    if args.out_csv is not None:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        root, ext = os.path.splitext(args.out_csv)
        summary.to_csv(f"{root}_summary{ext or '.csv'}", index=False)
    print(df.to_string(index=False))
    if not summary.empty:
        print("\nSummary:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
