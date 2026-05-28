import unittest

import torch

from src.agent.flow_level_encoder import FlowLevelFeaturesExtractor
from src.env.flow_level_env import FlowLevelNetEnv
from src.network.net import Flow, Network, generate_linear_5


class TestFlowLevelFeaturesExtractor(unittest.TestCase):
    def setUp(self):
        graph = generate_linear_5()
        path = [("E1", "S1"), ("S1", "S2"), ("S2", "E2")]
        flows = [
            Flow("F0", "E1", "E2", path, payload=64, period=2000, jitter=2000),
        ]
        self.env = FlowLevelNetEnv(Network(graph, flows), max_hops=5)

    def test_embedding_shape(self):
        obs, _ = self.env.reset()
        tensor_obs = {
            key: torch.from_numpy(value).unsqueeze(0)
            for key, value in obs.items()
        }
        extractor = FlowLevelFeaturesExtractor(self.env.observation_space)

        out = extractor(tensor_obs)

        self.assertEqual(out.shape, (1, 128))

    def test_path_mask_changes_embedding(self):
        obs, _ = self.env.reset()
        full_obs = {
            key: torch.from_numpy(value).unsqueeze(0)
            for key, value in obs.items()
        }
        masked_obs = {
            key: value.clone()
            for key, value in full_obs.items()
        }
        masked_obs["path_mask"][:, -1] = 1
        masked_obs["path_features"][:, -1, :] = 10
        extractor = FlowLevelFeaturesExtractor(self.env.observation_space)

        full_out = extractor(full_obs)
        masked_out = extractor(masked_obs)

        self.assertFalse(torch.allclose(full_out, masked_out))


if __name__ == "__main__":
    unittest.main()
