import math
from typing import Iterable

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType

from src.env.env import NetEnv
from src.network.net import Network


def encode_gating_pattern(pattern: Iterable[int], max_hops: int) -> int:
    bits = [int(bit) for bit in pattern]
    if len(bits) > max_hops:
        raise ValueError(f"Pattern length {len(bits)} exceeds max_hops {max_hops}.")
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("Gating pattern must contain only 0 or 1.")

    action = 0
    for hop_index, bit in enumerate(bits):
        action |= bit << hop_index
    return action


def decode_gating_action(action: int, max_hops: int) -> list[int]:
    if action < 0:
        raise ValueError("Action must be non-negative.")
    if action >= (1 << max_hops):
        raise ValueError(f"Action {action} exceeds Discrete(2^{max_hops}).")
    return [(action >> hop_index) & 1 for hop_index in range(max_hops)]


class FlowLevelNetEnv(NetEnv):
    """
    Flow-level variant of NetEnv.

    One environment step schedules the current flow end-to-end. The action is a
    bitmask over hops: bit i decides whether hop i enables gating. Per-hop timing
    scheduling is delegated to NetEnv.step(), so timing rules stay aligned with
    the existing project environment.
    """

    def __init__(self, network: Network = None, max_hops: int = None):
        super().__init__(network)

        inferred_max_hops = max(len(flow.path) for flow in self.flows)
        self.max_hops = inferred_max_hops if max_hops is None else max_hops
        if self.max_hops < inferred_max_hops:
            raise ValueError(
                f"max_hops {self.max_hops} is smaller than the longest flow path "
                f"{inferred_max_hops}."
            )

        flow_feature = self.state_encoder._flow_feature()
        link_feature = self.state_encoder._link_feature(self.current_link().link_id)

        self.observation_space = gym.spaces.Dict({
            "flow_feature": gym.spaces.Box(
                low=0,
                high=np.inf,
                shape=flow_feature.shape,
                dtype=np.float32,
            ),
            "path_features": gym.spaces.Box(
                low=0,
                high=np.inf,
                shape=(self.max_hops, link_feature.shape[0]),
                dtype=np.float32,
            ),
            "path_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.max_hops,),
                dtype=np.float32,
            ),
        })
        self.action_space = gym.spaces.Discrete(1 << self.max_hops)

    def _generate_state(self) -> ObsType:
        flow = self.current_flow()
        if len(flow.path) > self.max_hops:
            raise ValueError(
                f"Current flow path length {len(flow.path)} exceeds max_hops {self.max_hops}."
            )

        link_feature_template = self.state_encoder._link_feature(flow.path[0])
        path_features = np.zeros((self.max_hops, link_feature_template.shape[0]), dtype=np.float32)
        path_mask = np.zeros((self.max_hops,), dtype=np.float32)

        for hop_index, link_id in enumerate(flow.path):
            path_features[hop_index] = self.state_encoder._link_feature(link_id)
            path_mask[hop_index] = 1.0

        return {
            "flow_feature": self.state_encoder._flow_feature(),
            "path_features": path_features,
            "path_mask": path_mask,
        }

    def action_masks(self) -> np.ndarray:
        flow = self.current_flow()
        path_len = len(flow.path)
        masks = np.zeros(self.action_space.n, dtype=bool)

        for action in range(self.action_space.n):
            pattern = decode_gating_action(action, self.max_hops)
            if any(pattern[hop_index] for hop_index in range(path_len, self.max_hops)):
                continue
            if self._pattern_has_impossible_gating(flow, pattern[:path_len]):
                continue
            masks[action] = True

        return masks

    def _pattern_has_impossible_gating(self, flow, pattern: list[int]) -> bool:
        for hop_index, gating in enumerate(pattern):
            if not gating:
                continue
            link = self.link_dict[flow.path[hop_index]]
            if not self.add_gating(link, flow.period, attempt=True):
                return True
        return False

    def step(
            self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict]:
        action = int(action)
        pattern = decode_gating_action(action, self.max_hops)
        flow = self.current_flow()
        path_len = len(flow.path)
        if any(pattern[hop_index] for hop_index in range(path_len, self.max_hops)):
            obs = self.observation_space.sample()
            return obs, -1.0, True, False, {
                "success": False,
                "msg": "Padding hops must not enable gating.",
                "flow_level_action": action,
                "gating_pattern": pattern,
            }

        total_reward = 0.0
        last_obs = None
        last_info = {}
        terminated = False
        truncated = False

        for hop_index in range(path_len):
            last_obs, reward, terminated, truncated, last_info = super().step(pattern[hop_index])
            total_reward += reward
            if terminated or truncated:
                break

        last_info = dict(last_info)
        last_info["flow_level_action"] = action
        last_info["gating_pattern"] = pattern[:path_len]
        last_info["flow_level_reward"] = total_reward

        if last_obs is None:
            last_obs = self._generate_state()

        return last_obs, total_reward, terminated, truncated, last_info

    @staticmethod
    def action_space_size(max_hops: int) -> int:
        return int(math.pow(2, max_hops))
