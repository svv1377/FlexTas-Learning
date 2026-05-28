import unittest

import numpy as np

from app.flow_level_train import build_policy_kwargs, infer_max_hops, make_env, parse_jitters
from src.agent.flow_level_encoder import FlowLevelFeaturesExtractor
from src.network.net import generate_linear_5


class TestFlowLevelTrain(unittest.TestCase):
    def test_parse_jitters(self):
        jitters = parse_jitters(["0.1", "0.2,0.5"])

        self.assertTrue(np.array_equal(jitters, np.array([[0.1], [0.2, 0.5]], dtype=object)))

    def test_build_policy_kwargs(self):
        self.assertIs(
            build_policy_kwargs()["features_extractor_class"],
            FlowLevelFeaturesExtractor,
        )

    def test_infer_max_hops_from_topology(self):
        graph = generate_linear_5()

        self.assertEqual(infer_max_hops(graph), 6)

    def test_make_env_uses_fixed_max_hops(self):
        graph = generate_linear_5()
        max_hops = infer_max_hops(graph)
        list_jitters = parse_jitters(["0.1"])

        env_fn = make_env(
            num_flows=4,
            rank=0,
            graph=graph,
            list_jitters=list_jitters,
            max_hops=max_hops,
        )
        env = env_fn()
        obs, _ = env.reset()

        self.assertEqual(obs["path_features"].shape[0], max_hops)
        self.assertEqual(env.observation_space["path_features"].shape[0], max_hops)


if __name__ == "__main__":
    unittest.main()
