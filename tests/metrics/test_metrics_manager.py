import unittest
from contextlib import ExitStack
from unittest.mock import patch, MagicMock, Mock

from src.metrics import metrics_manager
from src.metrics.metrics_manager import EnergyCapture, PerplexityCapture


class MetricsManagerTest(unittest.TestCase):

    def _patch(self, attr_path):
        return self._patch_stack.enter_context(patch(attr_path))

    def setUp(self):
        self._patch_stack = ExitStack()
        self.addCleanup(self._patch_stack.close)
        self.makedirs_patch = self._patch('src.metrics.metrics_manager.os.makedirs')
        self.open_patch = self._patch('src.metrics.metrics_manager.open')
        self.writer_patch = self._patch('src.metrics.metrics_manager.csv.writer')
        self.writer_mock_scenario_1 = Mock()
        self.writer_mock_scenario_2 = Mock()
        self.writer_patch.side_effect = [self.writer_mock_scenario_1, self.writer_mock_scenario_2]

    def test_accept_energy(self):
        energy_capture_1 = EnergyCapture(
            label="my_label_1",
            layer_idx=5,
            head_idxs=[0, 1],
            average_energy_per_token_mj=234,
            average_time_per_token_ms=12
        )
        energy_capture_2 = EnergyCapture(
            label="my_label_2",
            layer_idx=3,
            head_idxs=[2, 3],
            average_energy_per_token_mj=150,
            average_time_per_token_ms=8
        )
        energy_capture_3 = EnergyCapture(
            label="my_label_1",
            layer_idx=7,
            head_idxs=[4, 5],
            average_energy_per_token_mj=300,
            average_time_per_token_ms=15
        )
        energy_captures = [energy_capture_1, energy_capture_2, energy_capture_3]

        scenario_1 = "scenario_1"
        scenario_2 = "scenario_2"
        scenarios = [scenario_1, scenario_2, scenario_1]

        for scenario, energy_capture in zip(scenarios, energy_captures):
            metrics_manager.accept_energy(energy_capture, scenario=scenario)

        expected_scenarios = [scenario_1, scenario_2]

        self.assertEqual(2, self.open_patch.call_count)
        for i, scenario in enumerate(expected_scenarios):
            expected_dir = f'{metrics_manager.metrics_directory}/{scenario}'
            expected_path = f'{expected_dir}/energy-metrics.csv'
            self.assertEqual(expected_dir, self.makedirs_patch.call_args_list[i][0][0])
            self.assertEqual((expected_path, 'w'), self.open_patch.call_args_list[i][0])
        for args in self.writer_patch.call_args_list:
            self.assertEqual(self.open_patch.return_value, args[0][0])

        expected_header = (
            'label',
            'layer_idx',
            'head_idxs',
            'average_energy_per_token_mj',
            'average_time_per_token_ms',
        )

        self.assertEqual(3, self.writer_mock_scenario_1.writerow.call_count)
        self.assertEqual(expected_header, self.writer_mock_scenario_1.writerow.call_args_list[0][0][0])
        for i, energy_capture in enumerate([energy_capture_1, energy_capture_3]):
            self.assertEqual(energy_capture, self.writer_mock_scenario_1.writerow.call_args_list[1 + i][0][0])

        self.assertEqual(2, self.writer_mock_scenario_2.writerow.call_count)
        self.assertEqual(expected_header, self.writer_mock_scenario_2.writerow.call_args_list[0][0][0])
        self.assertEqual(energy_capture_2, self.writer_mock_scenario_2.writerow.call_args_list[1][0][0])


    def test_accept_perplexity(self):
        perplexity_capture_1 = PerplexityCapture(
            label="my_label_1",
            layer_idx=5,
            head_idxs=[0, 1],
            perplexity= 0.9
        )
        perplexity_capture_2 = PerplexityCapture(
            label="my_label_2",
            layer_idx=3,
            head_idxs=[2, 3],
            perplexity= 12.3
        )
        perplexity_capture_3 = PerplexityCapture(
            label="my_label_1",
            layer_idx=7,
            head_idxs=[4, 5],
            perplexity= 4.3
        )
        perplexity_captures = [perplexity_capture_1, perplexity_capture_2, perplexity_capture_3]

        scenario_1 = "scenario_1"
        scenario_2 = "scenario_2"
        scenarios = [scenario_1, scenario_2, scenario_1]

        for scenario, perplexity_capture in zip(scenarios, perplexity_captures):
            metrics_manager.accept_perplexity(perplexity_capture, scenario=scenario)

        expected_scenarios = [scenario_1, scenario_2]

        self.assertEqual(2, self.open_patch.call_count)
        for i, scenario in enumerate(expected_scenarios):
            expected_dir = f'{metrics_manager.metrics_directory}/{scenario}'
            expected_path = f'{expected_dir}/perplexity-metrics.csv'
            self.assertEqual(expected_dir, self.makedirs_patch.call_args_list[i][0][0])
            self.assertEqual((expected_path, 'w'), self.open_patch.call_args_list[i][0])
        for args in self.writer_patch.call_args_list:
            self.assertEqual(self.open_patch.return_value, args[0][0])

        expected_header = (
            'label',
            'layer_idx',
            'head_idxs',
            'perplexity'
        )

        self.assertEqual(3, self.writer_mock_scenario_1.writerow.call_count)
        self.assertEqual(expected_header, self.writer_mock_scenario_1.writerow.call_args_list[0][0][0])
        for i, energy_capture in enumerate([perplexity_capture_1, perplexity_capture_3]):
            self.assertEqual(energy_capture, self.writer_mock_scenario_1.writerow.call_args_list[1 + i][0][0])

        self.assertEqual(2, self.writer_mock_scenario_2.writerow.call_count)
        self.assertEqual(expected_header, self.writer_mock_scenario_2.writerow.call_args_list[0][0][0])
        self.assertEqual(perplexity_capture_2, self.writer_mock_scenario_2.writerow.call_args_list[1][0][0])


if __name__ == '__main__':
    unittest.main()
