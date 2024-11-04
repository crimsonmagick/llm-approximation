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
    
    def prune(self, layer_idx):
        llama_attention = self.model.model.layers[layer_idx].self_attn
        if not isinstance(llama_attention, LlamaSdpaAttention):
            error_message = f'Only LlamaSdpaAttention is currently supported, type={type(llama_attention)}'
            logger.error(error_message)
            raise TypeError(error_message)
        
        config, layer_idx = llama_attention.config, llama_attention.layer_idx
        pruned_llama_attention = PrunedLlamaSdpaAttention(config, layer_idx)
        
        params = chain(zip(llama_attention.q_proj.parameters(), pruned_llama_attention.q_proj.parameters()),
                       zip(llama_attention.k_proj.parameters(), pruned_llama_attention.k_proj.parameters()),
                       zip(llama_attention.v_proj.parameters(), pruned_llama_attention.v_proj.parameters()))
        pruned_llama_attention.to(llama_attention.q_proj.weight.device, dtype=llama_attention.q_proj.weight.dtype)
        
        with torch.no_grad():
            # pruned_llama_attention.q_proj.weight.copy_(llama_attention.q_proj.weight)
            # pruned_llama_attention.k_proj.weight.copy_(llama_attention.k_proj.weight)
            # pruned_llama_attention.v_proj.weight.copy_(llama_attention.v_proj.weight)
            for param, copied_param in params:
                copied_param.copy_(param)
            self.model.model.layers[layer_idx].self_attn = pruned_llama_attention
            
        print(self.model)
        # logger.info(f"hi, layer_idx={layer_idx}")