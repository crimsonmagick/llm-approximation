from transformers.utils import logging

from transformers import Cache
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, apply_rotary_pos_emb
from transformers.models.llama.configuration_llama import LlamaConfig

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import math
from typing import List, Optional, Tuple, Union


logger = logging.get_logger(__name__)


class PrunedLlamaSdpaAttention(LlamaSdpaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        
        bsz, q_len, _ = hidden_states.size()
        
        query_states_1 = self.q_proj(hidden_states)
        key_states_1 = self.k_proj(hidden_states)
        value_states_1 = self.v_proj(hidden_states)
        
        query_states_2 = query_states_1.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states_2 = key_states_1.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states_2 = value_states_1.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states_2, position_ids)
        else:
            cos, sin = position_embeddings
        query_states_3, key_states_3 = apply_rotary_pos_emb(query_states_2, key_states_2, cos, sin)
        
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states_3, value_states_2 = past_key_value.update(key_states_3, value_states_2, self.layer_idx, cache_kwargs)
        
        key_states_4 = self.repeat_kv(key_states_3, self.num_key_value_groups)
        value_states_3 = self.repeat_kv(value_states_2, self.num_key_value_groups)
        
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states_4.shape[-2]]
        
        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states_3.device.type == "cuda" and causal_mask is not None:
            query_states_3 = query_states_3.contiguous()
            key_states_4 = key_states_4.contiguous()
            value_states_3 = value_states_3.contiguous()
        
        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False
        
        attn_output_1 = torch.nn.functional.scaled_dot_product_attention(
            query_states_3,
            key_states_4,
            value_states_3,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        
        attn_output_2 = attn_output_1.transpose(1, 2).contiguous()
        attn_output_3 = attn_output_2.view(bsz, q_len, -1)
        
        attn_output_4 = self.o_proj(attn_output_3)
        
        return attn_output_4, None, past_key_value
    
    @staticmethod
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)