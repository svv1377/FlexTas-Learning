import unittest

from app.flow_level_evaluation import (
    FlowLevelEvalSettings,
    parse_float_list,
    parse_int_list,
    parse_str_list,
)


class TestFlowLevelEvaluation(unittest.TestCase):
    def test_parse_str_list(self):
        self.assertEqual(parse_str_list("RRG,ERG"), ["RRG", "ERG"])

    def test_parse_int_list(self):
        self.assertEqual(parse_int_list("10,50"), [10, 50])

    def test_parse_float_list(self):
        self.assertEqual(parse_float_list("0.1,0.5"), [0.1, 0.5])

    def test_settings_to_dict(self):
        settings = FlowLevelEvalSettings(
            topo="RRG",
            num_flows=50,
            link_rate=100,
            jitter=0.1,
            seed=3,
            timeout=5,
            model_path="model.zip",
        )

        self.assertEqual(settings.to_dict()["topo"], "RRG")
        self.assertEqual(settings.to_dict()["num_flows"], 50)
        self.assertEqual(settings.to_dict()["model_path"], "model.zip")


if __name__ == "__main__":
    unittest.main()
