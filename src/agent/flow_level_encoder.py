import gymnasium as gym
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FlowLevelFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict,
                 flow_embedding: int = 64, path_embedding: int = 64):
        super().__init__(observation_space, features_dim=flow_embedding + path_embedding)

        flow_dim = observation_space["flow_feature"].shape[0]
        link_feature_dim = observation_space["path_features"].shape[-1]

        self.flow_encoder = nn.Sequential(
            nn.Linear(flow_dim, flow_embedding),
            nn.ReLU(),
        )
        self.path_link_encoder = nn.Sequential(
            nn.Linear(link_feature_dim, path_embedding),
            nn.ReLU(),
        )

    def forward(self, observations) -> torch.Tensor:
        flow_encoded = self.flow_encoder(observations["flow_feature"])

        path_features = observations["path_features"]
        path_mask = observations["path_mask"].unsqueeze(-1)

        path_encoded = self.path_link_encoder(path_features)
        path_encoded = path_encoded * path_mask
        path_lengths = path_mask.sum(dim=1).clamp_min(1.0)
        path_encoded = path_encoded.sum(dim=1) / path_lengths

        return torch.cat([flow_encoded, path_encoded], dim=1)
