from itertools import zip_longest, chain

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
    def __init__(self, config: LlamaConfig, layer_idx: int, prune_heads: Optional[List[int]] = None):
        super().__init__(config, layer_idx)
        self.prune_heads = sorted(prune_heads) if prune_heads is not None else prune_heads
        self.keep_idxs = self.get_keep_indices(self.get_heads(self.num_heads, self.prune_heads), self.head_dim)
        self.keep_kv_idxs = self.get_keep_indices(                      )
    
    def prune(self):
        if self.prune_heads is not None:
            # TODO validate prune_heads
            self.keep_hds = torch.tensor(self.get_heads(self.num_heads, self.prune_heads), dtype=torch.long,
                                         device=self.q_proj.weight.device)
            self.q_proj = self.prune_linear(self.q_proj, self.keep_idxs, 0)
            self.o_proj = self.prune_linear(self.o_proj, self.keep_idxs, 1)
            self.num_heads = self.num_heads - len(self.prune_heads)
    
    @staticmethod
    @torch.no_grad()
    def prune_linear(to_prune: nn.Linear, keep_idxs, dim) -> nn.Linear:
        pruned_weights = torch.index_select(
            to_prune.weight, dim, torch.tensor(keep_idxs, dtype=torch.long, device=to_prune.weight.device))
        pruned_linear = nn.Linear(in_features=pruned_weights.size(dim=1),
                                  out_features=pruned_weights.size(dim=0),
                                  bias=False, device=to_prune.weight.device, dtype=to_prune.weight.dtype)
        pruned_linear.train = to_prune.train
        pruned_linear.weight.copy_(pruned_weights)
        return pruned_linear
    
    @staticmethod
    def get_keep_indices(keep_hds, head_dim):
        return list(chain.from_iterable(map(lambda i: range(head_dim * i, head_dim * (i + 1)), range(len(keep_hds)))))
    
    @staticmethod
    def get_keep_kv_heads(keep_hds, num_groups):
        kv_heads = {hd // num_groups for hd in keep_hds}
        return list(kv_heads)
    
    @staticmethod
    def get_heads(num_hds, hds_to_prune):
        return list(set(range(num_hds)) - set(hds_to_prune))
    
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
        
        query_states_projected = self.q_proj(hidden_states)
        key_states_projected = self.k_proj(hidden_states)
        value_states_projected = self.v_proj(hidden_states)
        
        qs_per_attnhead = query_states_projected.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        ks_per_kvhead = key_states_projected.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        vs_per_kvhead = value_states_projected.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(vs_per_kvhead, position_ids)
        else:
            cos, sin = position_embeddings
        qs_pos_per_attnhead, ks_pos_per_kvhead = apply_rotary_pos_emb(qs_per_attnhead, ks_per_kvhead, cos, sin)
        
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            ks_pos_per_kvhead, vs_per_kvhead = past_key_value.update(ks_pos_per_kvhead, vs_per_kvhead, self.layer_idx,
                                                                     cache_kwargs)
        
        if self.prune_heads is None:
            ks_pos_grouped_per_head = self.repeat_kv(ks_pos_per_kvhead, self.num_key_value_groups)
            vs_pos_grouped_per_head = self.repeat_kv(vs_per_kvhead, self.num_key_value_groups)
        else:
            ks_pos_grouped_per_head = self.repeat_kv_pruned(ks_pos_per_kvhead, self.num_key_value_groups)
            vs_pos_grouped_per_head = self.repeat_kv_pruned(vs_per_kvhead, self.num_key_value_groups)
        
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : ks_pos_grouped_per_head.shape[-2]]
        
        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if qs_pos_per_attnhead.device.type == "cuda" and causal_mask is not None:
            qs_pos_per_attnhead = qs_pos_per_attnhead.contiguous()
            ks_pos_grouped_per_head = ks_pos_grouped_per_head.contiguous()
            vs_pos_grouped_per_head = vs_pos_grouped_per_head.contiguous()
        
        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False
        
        if torch.numel(qs_pos_per_attnhead) != 0:
        
            attn_output_per_attnhead = torch.nn.functional.scaled_dot_product_attention(
                qs_pos_per_attnhead,
                ks_pos_grouped_per_head,
                vs_pos_grouped_per_head,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            attn_output_per_attnhead = qs_per_attnhead
        
        attn_output_per_sequence_per_attnhead = attn_output_per_attnhead.transpose(1, 2).contiguous()
        attn_output_per_sequence = attn_output_per_sequence_per_attnhead.view(bsz, q_len, -1)
        
        attn_projected = self.o_proj(attn_output_per_sequence)
        
        return attn_projected, None, past_key_value
    
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
    
    def repeat_kv_pruned(self, states_per_kvhead: torch.Tensor, num_groups: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_kv_heads, seq_len, head_dim = states_per_kvhead.shape
        if num_groups == 1:
            return states_per_kvhead
        states_per_kvhead = states_per_kvhead[:, :, None, :, :].expand(batch, num_kv_heads, num_groups, seq_len,
                                                                       head_dim)
        unpruned = states_per_kvhead.reshape(batch, num_kv_heads * num_groups, seq_len, head_dim)
        return torch.index_select(unpruned, 1, self.keep_hds)
