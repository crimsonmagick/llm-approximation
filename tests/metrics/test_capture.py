import unittest
from unittest.mock import patch, Mock

import torch
from torch import nn
from types import SimpleNamespace

from src.metrics.capture import instrument
from src.metrics.metrics_manager import EnergyCapture
from tests.util.test_util_mixin import TestUtilMixin

SEQUENCE_COUNT = 6
SEQUENCE_LENGTH = 11
VOCABULARY_SIZE = 23


class CaptureTests(TestUtilMixin, unittest.TestCase):
    
    @patch("src.metrics.capture.EnergyRecorder")
    @patch("src.metrics.capture.get_allocated_memory")
    @patch("src.metrics.function.objective.aggregate_perplexity")
    @patch("src.metrics.function.objective.cross_entropy")
    @patch("src.metrics.metrics_manager.accept")
    def test_instrument(self, mock_accept, mock_cross_entropy,
                        mock_aggregate_perplexity, mock_get_allocated_memory, mock_energy_recorder):
        expected_label = 'my_expected_label'
        expected_suite = 'my_expected_suite'
        expected_layer_idx = 3
        expected_head_idxs = [0, 1, 4, 9]
        expected_logits = self.rand_logits(SEQUENCE_COUNT, SEQUENCE_LENGTH, VOCABULARY_SIZE)
        
        # actual losses arbitrary here, but the tensor length should look like this
        expected_losses = self.rand_losses((SEQUENCE_LENGTH - 1 * SEQUENCE_COUNT) * SEQUENCE_COUNT * SEQUENCE_LENGTH)
        mock_cross_entropy.return_value = expected_losses
        
        expected_aggregate_perplexity = 1.25
        mock_aggregate_perplexity.return_value = expected_aggregate_perplexity
        
        expected_memory = 1923.12
        mock_get_allocated_memory.return_value = expected_memory
        
        energy_recorder_instance = mock_energy_recorder.return_value
        energy_recorder_instance.start.return_value = None
        expected_time_ms = 240
        expected_energy_mj = 300
        expected_temperature_c = 52
        energy_recorder_instance.end.return_value.get_energy_metrics.return_value \
            = (expected_energy_mj, expected_time_ms, expected_temperature_c)
        
        stubbed_prediction = SimpleNamespace(logits=expected_logits)
        model = Mock(spec=nn.Module)
        model.forward.return_value = stubbed_prediction
        
        input_ids = self.rand_labels(SEQUENCE_COUNT, SEQUENCE_LENGTH, VOCABULARY_SIZE)
        attention_mask = torch.ones(SEQUENCE_COUNT, SEQUENCE_LENGTH)
        
        # instrument is under test
        instrumented_model = instrument(model, expected_label, expected_layer_idx, expected_head_idxs, expected_suite)
        instrumented_model.forward(input_ids=input_ids, attention_mask=attention_mask)
        
        self.assertEqual(model, instrumented_model)
        
        args, kwargs = mock_accept.call_args
        suite: str = kwargs['suite']
        metrics: EnergyCapture = args[0]
        
        token_count = SEQUENCE_COUNT * SEQUENCE_LENGTH
        
        self.assertEqual(expected_suite, suite)
        self.assertEqual(f'{expected_label}-0', metrics.label)
        self.assertEqual(expected_layer_idx, metrics.layer_idx)
        self.assertEqual(expected_aggregate_perplexity, metrics.perplexity)
        self.assertEqual(expected_energy_mj / token_count, metrics.average_energy_per_token_mj)
        self.assertEqual(expected_time_ms / token_count, metrics.average_time_per_token_ms)
        self.assertEqual(expected_memory, metrics.allocated_memory)
        self.assertEqual(expected_temperature_c, metrics.temperature)

