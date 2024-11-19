from itertools import zip_longest, chain

from transformers.utils import logging

from transformers import Cache
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, apply_rotary_pos_emb
from transformers.models.llama.configuration_llama import LlamaConfig

import torch
import torch.utils.checkpoint
from torch import nn
from typing import List, Optional, Tuple, Union

logger = logging.get_logger(__name__)


class PrunedLlamaSdpaAttention(LlamaSdpaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int, prune_heads: Optional[List[int]] = None):
        super().__init__(config, layer_idx)
        self.pruned_heads = sorted(prune_heads) if prune_heads is not None else prune_heads
        self.keep_heads = self.get_heads(self.num_heads, self.pruned_heads)
        self.register_buffer('keep_idxs',
                             torch.tensor(self.get_keep_indices(self.keep_heads, self.head_dim), dtype=torch.long,
                                          device=self.q_proj.weight.device), False)
        self.register_buffer('keep_hds',
                             torch.tensor(self.get_heads(self.num_heads, self.pruned_heads), dtype=torch.long,
                                          device=self.q_proj.weight.device), False)
        self.prune_kv_heads = True
        if self.prune_kv_heads:
            self.keep_kv_heads = self.get_keep_kv_heads(self.keep_heads, self.num_key_value_groups)
            self.register_buffer('keep_kv_idxs',
                                 torch.tensor(self.get_keep_indices(self.keep_kv_heads, self.head_dim), dtype=torch.long,
                                              device=self.q_proj.weight.device), False)
            self.pruned_kv_counts = self.build_pruned_kv_counts(self.keep_heads, self.num_key_value_groups)
            self.num_key_value_groups = len(self.pruned_kv_counts)
        print("__init__ finished")
    
    def prune(self):
        if self.pruned_heads is not None:
            # TODO validate prune_heads, integrate with PretrainedModel#prune_heads
            self.q_proj = self.prune_linear(self.q_proj, self.keep_idxs, 0)
            self.o_proj = self.prune_linear(self.o_proj, self.keep_idxs, 1)
            self.num_heads = self.num_heads - len(self.pruned_heads)
            
            if self.prune_kv_heads:
                self.k_proj = self.prune_linear(self.k_proj, self.keep_kv_idxs, 0)
                self.v_proj = self.prune_linear(self.v_proj, self.keep_kv_idxs, 0)
                self.num_key_value_heads = len(self.keep_kv_heads)
    
    @staticmethod
    def build_pruned_kv_counts(keep_heads, num_key_value_groups) -> list[int]:
        kv_counts = dict()
        for i in keep_heads:
            group_idx = i // num_key_value_groups
            if group_idx not in kv_counts:
                kv_counts[group_idx] = 0
            kv_counts[group_idx] += 1
        return list(kv_counts.values())

        # return len({i // group_size for i in keep_heads})
    
    @staticmethod
    @torch.no_grad()
    def prune_linear(to_prune: nn.Linear, keep_idxs, dim) -> nn.Linear:
        pruned_weights = torch.index_select(to_prune.weight, dim, keep_idxs)
        pruned_linear = nn.Linear(in_features=pruned_weights.size(dim=1),
                                  out_features=pruned_weights.size(dim=0),
                                  bias=False, device=to_prune.weight.device, dtype=to_prune.weight.dtype)
        pruned_linear.train = to_prune.train
        pruned_linear.weight.copy_(pruned_weights)
        return pruned_linear
    
    @staticmethod
    def get_keep_indices(keep_hds, head_dim):
        return list(chain.from_iterable(map(lambda i: range(head_dim * i, head_dim * (i + 1)), keep_hds)))
    
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
            # TODO need to add support for pruning when not using memory-efficient SDPA attention
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
        
        if self.pruned_heads is None:
            ks_pos_grouped_per_head = self.repeat_kv(ks_pos_per_kvhead, self.num_key_value_groups)
            vs_pos_grouped_per_head = self.repeat_kv(vs_per_kvhead, self.num_key_value_groups)
        else:
            ks_pos_grouped_per_head = self.repeat_kv_pruned(ks_pos_per_kvhead, self.pruned_kv_counts)
            vs_pos_grouped_per_head = self.repeat_kv_pruned(vs_per_kvhead, self.pruned_kv_counts)
        
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
    
    @staticmethod
    def repeat_kv_pruned(states_per_kvhead: torch.Tensor, pruned_kv_counts: list[int]) -> torch.Tensor:
        """
        The hidden states go from (batch, num_key_value_heads (ignored), seqlen, head_dim) to
        (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, _, seq_len, head_dim = states_per_kvhead.shape
        splits = states_per_kvhead.split(1, dim=1)
        zipped = zip(splits, pruned_kv_counts)
        repeated = tuple(map(lambda t: t[0].expand(batch, t[1], seq_len, head_dim), zipped))
        return torch.cat(repeated, dim=1)

