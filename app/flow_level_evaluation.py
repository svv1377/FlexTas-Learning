import argparse
import os
import time
from dataclasses import asdict, dataclass

import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

from src.app.flow_level_drl_scheduler import FlowLevelDrlScheduler
from src.app.evaluation import SchedulerTester
from src.network.net import FlowGenerator, Network, generate_graph


@dataclass
class FlowLevelEvalSettings:
    topo: str
    num_flows: int
    link_rate: int
    jitter: float
    seed: int
    timeout: int
    model_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in parse_str_list(value)]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in parse_str_list(value)]


def make_network(settings: FlowLevelEvalSettings) -> Network:
    graph = generate_graph(settings.topo, settings.link_rate)
    flow_generator = FlowGenerator(graph, seed=settings.seed, jitters=settings.jitter)
    return Network(graph, flow_generator(settings.num_flows))


def evaluate_single(settings: FlowLevelEvalSettings) -> dict:
    network = make_network(settings)
    scheduler = FlowLevelDrlScheduler(
        network,
        num_envs=1,
        timeout_s=settings.timeout,
        vec_env_cls=DummyVecEnv,
    )
    scheduler.load_model(settings.model_path, "MaskablePPO")

    start = time.time()
    tester = SchedulerTester(network, scheduler)
    try:
        result = tester.stress_test()
    finally:
        scheduler.env.close()

    result = settings.to_dict() | result
    result["wall_time"] = time.time() - start
    result["success"] = scheduler.get_res() is not None
    return result


def evaluate(settings_list: list[FlowLevelEvalSettings]) -> pd.DataFrame:
    rows = [evaluate_single(settings) for settings in settings_list]
    return pd.DataFrame(rows)


def build_settings(args) -> list[FlowLevelEvalSettings]:
    settings_list = []
    for topo in parse_str_list(args.topos):
        for num_flows in parse_int_list(args.list_num_flows):
            for jitter in parse_float_list(args.jitters):
                for seed in range(args.seed, args.seed + args.num_tests):
                    settings_list.append(FlowLevelEvalSettings(
                        topo=topo,
                        num_flows=num_flows,
                        link_rate=args.link_rate,
                        jitter=jitter,
                        seed=seed,
                        timeout=args.timeout,
                        model_path=args.model,
                    ))
    return settings_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topos", type=str, required=True)
    parser.add_argument("--list_num_flows", type=str, required=True)
    parser.add_argument("--link_rate", type=int, default=100)
    parser.add_argument("--jitters", type=str, default="0.1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_tests", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--to_csv", type=str, default=None)
    args = parser.parse_args()

    assert os.path.isfile(args.model), f"Cannot find flow-level model {args.model}"

    df = evaluate(build_settings(args))
    if args.to_csv is not None:
        os.makedirs(os.path.dirname(args.to_csv) or ".", exist_ok=True)
        df.to_csv(args.to_csv, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
