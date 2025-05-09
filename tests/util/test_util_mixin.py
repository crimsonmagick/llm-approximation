import torch
from torch import Tensor

from tests.util.torch_utils.device import get_device


class TestUtilMixin:
    
    def setUp(self) -> None:
        self.device = get_device()
        generator_seed = 12  # arbitrary
        # TODO support generators for different devices - need to save multiple versions of expected tensors
        self.generator_device = torch.device("cpu")
        self.generator = torch.Generator(device=self.generator_device).manual_seed(generator_seed)
    
    def rand_logits(self, sequence_count, sequence_length, vocabulary_size) -> Tensor:
        logits = (torch.rand(sequence_count, sequence_length, vocabulary_size,
                             generator=self.generator, device=self.generator_device) - 0.5) * 10
        return logits.to(device=self.device)
    
    def rand_losses(self, sequence_count) -> Tensor:
        losses = (torch.rand(sequence_count, generator=self.generator, device=self.generator_device) - 0.5) * 10
        return losses.to(device=self.device)
    
    def rand_labels(self, sequence_count, sequence_length, vocabulary_size) -> Tensor:
        labels = torch.randint(0, vocabulary_size, size=(sequence_count, sequence_length), generator=self.generator,
                               device=self.generator_device)
        return labels.to(device=self.device)
