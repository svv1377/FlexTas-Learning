import argparse
import logging
import multiprocessing
import os
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import SubprocVecEnv

from definitions import OUT_DIR
from src.agent.flow_level_encoder import FlowLevelFeaturesExtractor
from src.app.flow_level_drl_scheduler import clone_network
from src.env.flow_level_env import FlowLevelNetEnv
from src.lib.log_config import log_config
from src.lib.timing_decorator import timing_decorator
from src.network.net import FlowGenerator, Network, generate_graph

TOPO = "CEV"
NUM_ENVS = multiprocessing.cpu_count()
NUM_FLOWS = 50
DRL_ALG = "MaskablePPO"
MONITOR_ROOT_DIR = os.path.join(OUT_DIR, "flow_level_monitor")
MONITOR_DIR = None


def get_best_model_path():
    return os.path.join(OUT_DIR, f"best_flow_level_model_{TOPO}_{DRL_ALG}")


def parse_jitters(jitters: list[str]) -> np.ndarray:
    parsed = []
    for jitter_group in jitters:
        values = [float(jitter) for jitter in jitter_group.split(",")]
        assert all(0 <= jitter <= 1 for jitter in values), \
            ValueError("Jitters should be in range [0, 1].")
        parsed.append(values)
    return np.array(parsed, dtype=object)


def build_policy_kwargs() -> dict:
    return dict(
        features_extractor_class=FlowLevelFeaturesExtractor,
    )


def infer_max_hops(graph) -> int:
    es_nodes = [node for node, data in graph.nodes(data=True) if data["node_type"] == "ES"]
    max_hops = 0
    for src in es_nodes:
        for dst in es_nodes:
            if src == dst or not nx.has_path(graph, src, dst):
                continue
            max_hops = max(max_hops, nx.shortest_path_length(graph, src, dst))
    if max_hops == 0:
        raise ValueError("Cannot infer max_hops from graph; no ES-to-ES path found.")
    return max_hops


def make_env(num_flows, rank: int, graph, list_jitters, max_hops: int,
             training: bool = True, monitor_dir: str = None):
    def _init():
        list_flow_generators = []
        for jitters in list_jitters:
            flow_generator = FlowGenerator(graph, jitters=list(jitters))
            list_flow_generators.append(flow_generator)

        flows = random.choice(list_flow_generators)(num_flows)
        network = Network(graph, flows)
        env = FlowLevelNetEnv(clone_network(network), max_hops=max_hops)

        active_monitor_dir = MONITOR_DIR if monitor_dir is None else monitor_dir
        if active_monitor_dir is not None:
            env = Monitor(env, os.path.join(active_monitor_dir, f'{"train" if training else "eval"}_{rank}'))
        return env

    return _init


@timing_decorator(logging.info)
def train(topo: str, num_time_steps, jitters, num_flows=NUM_FLOWS,
          pre_trained_model=None, link_rate=100):
    os.makedirs(OUT_DIR, exist_ok=True)

    n_envs = NUM_ENVS
    graph = generate_graph(topo, link_rate)
    max_hops = infer_max_hops(graph)
    env = SubprocVecEnv([
        make_env(num_flows, i, graph, jitters, max_hops)
        for i in range(n_envs)
    ])

    if pre_trained_model is not None:
        model = MaskablePPO.load(pre_trained_model, env)
    else:
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=build_policy_kwargs(),
            verbose=1,
        )

    eval_env = SubprocVecEnv([
        make_env(num_flows, i, graph, jitters, max_hops, training=False)
        for i in range(n_envs)
    ])
    callback = EvalCallback(
        eval_env,
        best_model_save_path=get_best_model_path(),
        log_path=OUT_DIR,
        eval_freq=max(10000 // n_envs, 1),
    )

    model.learn(total_timesteps=num_time_steps, callback=callback)
    logging.info("------Finish flow-level learning------")


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Flow-Level Learning Curve"):
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.savefig(os.path.join(log_folder, "reward.png"))
    plt.show()


def _prepare_monitor_dir() -> str:
    i = 0
    while True:
        monitor_dir = os.path.join(MONITOR_ROOT_DIR, str(i))
        try:
            os.makedirs(monitor_dir, exist_ok=False)
            return monitor_dir
        except OSError:
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_steps", type=int, required=True)
    parser.add_argument("--num_flows", type=int, nargs="?", default=NUM_FLOWS)
    parser.add_argument("--num_envs", type=int, default=NUM_ENVS)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--topo", type=str, default="CEV")
    parser.add_argument("--link_rate", type=int, default=100)
    parser.add_argument("--jitters", type=str, nargs="+", required=True)
    args = parser.parse_args()

    support_link_rates = [100, 1000]
    assert args.link_rate in support_link_rates, \
        f"Unknown link rate {args.link_rate}, which is not in supported link rates {support_link_rates}"

    TOPO = args.topo
    NUM_ENVS = args.num_envs
    list_jitters = parse_jitters(args.jitters)

    log_config(os.path.join(OUT_DIR, "flow_level_train.log"), logging.DEBUG)
    logging.info(args)

    MONITOR_DIR = _prepare_monitor_dir()

    logging.info("start flow-level training...")
    train(args.topo, args.time_steps,
          list_jitters,
          num_flows=args.num_flows,
          pre_trained_model=args.model,
          link_rate=args.link_rate)

    plot_results(MONITOR_DIR)
