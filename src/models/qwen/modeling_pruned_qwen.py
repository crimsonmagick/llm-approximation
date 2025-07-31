from typing import Dict, List

from transformers import Qwen2ForCausalLM, Qwen2PreTrainedModel, GenerationMixin


class PrunedQwen2ForCausalLM(Qwen2ForCausalLM):

    # def __init__(self, config):
    #     Qwen2PreTrainedModel.__init__(self, config)
    #     GenerationMixin.__init__(self)
    #     self.model = PrunedLlamaModel(config)
    #     self.vocab_size = config.vocab_size
    #     self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    #
    #     # Initialize weights and apply final processing
    #     self.post_init()

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]):
        pass
        # self.model._prune_heads(heads_to_prune)
