import gc
import logging

import torch

from transformers import LlamaForCausalLM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LlamaModelPruner:
    
    def __init__(self, model: LlamaForCausalLM):
        self.model = model
    
    @torch.no_grad()
    def prune_heads(self, head_dictionary):
        for layer_idx, head_idxs in head_dictionary.items():
            transformer_block = self.model.model.layers[layer_idx]
            llama_attention = transformer_block.self_attn
            llama_attention.prune_heads(head_idxs)

            referrers = gc.get_referrers(llama_attention)
            print('any params in grad graph?')
            for param in llama_attention.parameters():
                print(param.grad_fn)
            print(f"References to llama_attention: {len(referrers)}")
            for ref in referrers:
                print(type(ref), ref)
            
    @torch.no_grad()
    def prune_layers(self, layers):
        for layer_idx in layers:
            self.model.model.layers.pop(layer_idx)