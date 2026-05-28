import unittest

from app.compareFlowLevel import (
    aggregate_experiment_rows,
    policy_kwargs_for_env,
    summarize_eval_results,
)
from src.agent.encoder import FeaturesExtractor
from src.agent.flowLevelEncoder import FlowLevelFeaturesExtractor


class TestFlowLevelCompare(unittest.TestCase):
    def test_aggregate_experiment_rows(self):
        rows = [
            {"env_kind": "hop", "train_steps": 10, "seed": 1, "success_rate": 1.0, "gcl_avg": 2.0},
            {"env_kind": "hop", "train_steps": 10, "seed": 2, "success_rate": 0.0, "gcl_avg": 4.0},
            {"env_kind": "flow", "train_steps": 10, "seed": 1, "success_rate": 1.0, "gcl_avg": 6.0},
        ]

        aggregated = aggregate_experiment_rows(rows)

        self.assertNotIn("index", aggregated.columns)
        hop_row = aggregated[
            (aggregated["env_kind"] == "hop") &
            (aggregated["train_steps"] == 10)
        ].iloc[0]
        self.assertEqual(hop_row["num_seeds"], 2)
        self.assertEqual(hop_row["success_rate_mean"], 0.5)
        self.assertEqual(hop_row["gcl_avg_mean"], 3.0)

    def test_policy_kwargs_for_env(self):
        self.assertIs(
            policy_kwargs_for_env("hop")["features_extractor_class"],
            FeaturesExtractor,
        )
        self.assertIs(
            policy_kwargs_for_env("flow")["features_extractor_class"],
            FlowLevelFeaturesExtractor,
        )

    def test_summarize_eval_results(self):
        summary = summarize_eval_results([
            {"success": True, "gcl_avg": 1.0, "gcl_max": 2, "episode_reward": 3.0},
            {"success": False, "episode_reward": -1.0},
            {"success": True, "gcl_avg": 3.0, "gcl_max": 4, "episode_reward": 5.0},
        ])

        self.assertEqual(summary["num_episodes"], 3)
        self.assertEqual(summary["success_rate"], 2 / 3)
        self.assertEqual(summary["gcl_avg"], 2.0)
        self.assertEqual(summary["gcl_max"], 4)
        self.assertEqual(summary["episode_reward_avg"], 7 / 3)


if __name__ == "__main__":
    unittest.main()
