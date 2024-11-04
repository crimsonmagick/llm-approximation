import logging
from itertools import chain

import torch
from torch import no_grad

from llama.models.modeling_pruned_llama import PrunedLlamaSdpaAttention
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaSdpaAttention

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LlamaAttentionPruner:
    
    def __init__(self, model: LlamaForCausalLM):
        self.model = model
    
    @torch.no_grad()
    def prune(self, layer_idx):
        llama_attention = self.model.model.layers[layer_idx].self_attn
        if not isinstance(llama_attention, LlamaSdpaAttention):
            error_message = f'Only LlamaSdpaAttention is currently supported, type={type(llama_attention)}'
            logger.error(error_message)
            raise TypeError(error_message)
    
        config, layer_idx = llama_attention.config, llama_attention.layer_idx
        pruned_llama_attention = PrunedLlamaSdpaAttention(config, layer_idx)
        pruned_llama_attention.eval()
        pruned_llama_attention.to(llama_attention.q_proj.weight.device, dtype=llama_attention.q_proj.weight.dtype)

        pruned_llama_attention.load_state_dict(llama_attention.state_dict())
        self.model.model.layers[layer_idx].self_attn = pruned_llama_attention
        
        print(self.model)