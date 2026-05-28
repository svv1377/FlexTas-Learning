import unittest

import numpy as np
from stable_baselines3.common.env_checker import check_env

from src.env.flow_level_env import (
    FlowLevelNetEnv,
    decode_gating_action,
    encode_gating_pattern,
)
from src.network.net import Flow, Network, generate_linear_5


class TestFlowLevelEncoding(unittest.TestCase):
    def test_encode_decode_gating_pattern(self):
        action = encode_gating_pattern([1, 0, 1], max_hops=3)

        self.assertEqual(action, 5)
        self.assertEqual(decode_gating_action(action, max_hops=3), [1, 0, 1])

    def test_rejects_pattern_longer_than_max_hops(self):
        with self.assertRaises(ValueError):
            encode_gating_pattern([1, 0, 1, 0], max_hops=3)


class TestFlowLevelNetEnv(unittest.TestCase):
    def setUp(self):
        graph = generate_linear_5()
        path = [("E1", "S1"), ("S1", "S2"), ("S2", "E2")]
        flows = [
            Flow("F0", "E1", "E2", path, payload=64, period=2000, jitter=2000),
            Flow("F1", "E1", "E2", path, payload=64, period=4000, jitter=2000),
        ]
        self.network = Network(graph, flows)

    def test_check_env(self):
        env = FlowLevelNetEnv(self.network)

        check_env(env)

    def test_observation_contains_flow_path_and_mask(self):
        env = FlowLevelNetEnv(self.network)

        obs, _ = env.reset()

        self.assertIn("flow_feature", obs)
        self.assertIn("path_features", obs)
        self.assertIn("path_mask", obs)
        self.assertEqual(obs["path_features"].shape[0], env.max_hops)
        self.assertEqual(obs["path_mask"].shape, (env.max_hops,))
        self.assertTrue(np.array_equal(obs["path_mask"], np.ones(env.max_hops, dtype=np.float32)))
        self.assertTrue(env.observation_space.contains(obs))

    def test_one_step_schedules_one_full_flow(self):
        env = FlowLevelNetEnv(self.network)
        env.reset()

        action = encode_gating_pattern([1, 1, 1], env.max_hops)
        obs, reward, terminated, truncated, info = env.step(action)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["success"])
        self.assertEqual(env.flow_index, 1)
        self.assertEqual(env.temp_operations, [])
        self.assertGreater(reward, 0)
        self.assertTrue(env.observation_space.contains(obs))

    def test_two_steps_can_schedule_two_flows(self):
        env = FlowLevelNetEnv(self.network)
        env.reset()

        action = encode_gating_pattern([0, 0, 0], env.max_hops)
        env.step(action)
        obs, reward, terminated, truncated, info = env.step(action)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])
        self.assertIn("ScheduleRes", info)
        self.assertTrue(env.observation_space.contains(obs))

    def test_action_mask_disallows_padding_bits(self):
        env = FlowLevelNetEnv(self.network, max_hops=5)
        env.reset()

        valid_action = encode_gating_pattern([0, 0, 0], env.max_hops)
        invalid_action = encode_gating_pattern([0, 0, 0, 1, 0], env.max_hops)
        mask = env.action_masks()

        self.assertTrue(mask[valid_action])
        self.assertFalse(mask[invalid_action])


if __name__ == "__main__":
    unittest.main()
