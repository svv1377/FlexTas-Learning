import unittest

import numpy as np

from app.flow_level_train import build_policy_kwargs, parse_jitters
from src.agent.flow_level_encoder import FlowLevelFeaturesExtractor


class TestFlowLevelTrain(unittest.TestCase):
    def test_parse_jitters(self):
        jitters = parse_jitters(["0.1", "0.2,0.5"])

        self.assertTrue(np.array_equal(jitters, np.array([[0.1], [0.2, 0.5]], dtype=object)))

    def test_build_policy_kwargs(self):
        self.assertIs(
            build_policy_kwargs()["features_extractor_class"],
            FlowLevelFeaturesExtractor,
        )


if __name__ == "__main__":
    unittest.main()
