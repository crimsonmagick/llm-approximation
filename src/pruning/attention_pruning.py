import gc
import logging

import torch

from src.llama.models.modeling_pruned_llama import PrunedLlamaSdpaAttention
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaSdpaAttention

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
            if not isinstance(llama_attention, LlamaSdpaAttention):
                error_message = f'Only LlamaSdpaAttention is currently supported, type={type(llama_attention)}'
                logger.error(error_message)
                raise TypeError(error_message)
        
            config = llama_attention.config
            pruned_llama_attention = PrunedLlamaSdpaAttention(config, layer_idx, head_idxs)
            pruned_llama_attention.eval()
            pruned_llama_attention.to(llama_attention.q_proj.weight.device, dtype=llama_attention.q_proj.weight.dtype)

            pruned_llama_attention.load_state_dict(llama_attention.state_dict())
            pruned_llama_attention.prune()
            transformer_block.self_attn = pruned_llama_attention
            referrers = gc.get_referrers(llama_attention)
            print('any params in grad graph?')
            for param in llama_attention.parameters():
                print(param.grad_fn)
            print(f"References to llama_attention: {len(referrers)}")
            for ref in referrers:
                print(type(ref), ref)
            for param in llama_attention.parameters():
                param.data = param.data.detach().cpu()
            del llama_attention
            
    @torch.no_grad()
    def prune_layers(self, layers):
        for layer_idx in layers:
            self.model.model.layers.pop(layer_idx)