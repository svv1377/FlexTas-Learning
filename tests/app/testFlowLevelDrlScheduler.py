import unittest

from stable_baselines3.common.vec_env import DummyVecEnv

from src.app.flowLevelDrlScheduler import FlowLevelDrlScheduler
from src.env.flowLevelEnv import FlowLevelNetEnv
from src.network.net import Flow, Network, generate_linear_5


class TestFlowLevelDrlScheduler(unittest.TestCase):
    def setUp(self):
        graph = generate_linear_5()
        path = [("E1", "S1"), ("S1", "S2"), ("S2", "E2")]
        flows = [
            Flow("F0", "E1", "E2", path, payload=64, period=2000, jitter=2000),
        ]
        self.network = Network(graph, flows)

    def test_initializes_flowLevelEnv(self):
        scheduler = FlowLevelDrlScheduler(
            self.network,
            num_envs=1,
            time_steps=1,
            vec_env_cls=DummyVecEnv,
        )

        self.assertEqual(scheduler.max_hops, 3)
        self.assertEqual(scheduler.env.get_attr("max_hops")[0], 3)
        self.assertEqual(scheduler.env.get_attr("action_space")[0].n, 8)
        self.assertIsInstance(scheduler.env.envs[0], FlowLevelNetEnv)
        self.assertIsNone(scheduler.get_res())
        scheduler.env.close()


if __name__ == "__main__":
    unittest.main()
