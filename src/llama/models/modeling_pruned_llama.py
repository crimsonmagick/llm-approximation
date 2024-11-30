from itertools import chain
from typing import List, Optional, Tuple, Dict

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import Cache, LlamaForCausalLM, GenerationMixin, LlamaModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, apply_rotary_pos_emb, LLAMA_ATTENTION_CLASSES, \
    LlamaPreTrainedModel, LlamaRMSNorm, LlamaRotaryEmbedding, LlamaDecoderLayer, LlamaMLP
from transformers.utils import logging

logger = logging.get_logger(__name__)


class PrunedLlamaForCausalLM(LlamaForCausalLM):
    
    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        GenerationMixin.__init__(self)
        self.model = PrunedLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]):
        self.model._prune_heads(heads_to_prune)
    
    def prune_layers(self, layers):
        self.model.prune_layers(layers)


class PrunedLlamaModel(LlamaModel):
    
    def __init__(self, config: LlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [PrunedLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]):
        for layer_idx, head_idxs in heads_to_prune.items():
            transformer_block = self.layers[layer_idx]
            llama_attention = transformer_block.self_attn
            llama_attention.prune_heads(head_idxs)
    
    def prune_layers(self, layers):
        for layer_idx in layers:
            self.layers.pop(layer_idx)


class PrunedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        
        self.self_attn = PrunedLlamaSdpaAttention(config=config, layer_idx=layer_idx)
        
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class PrunedLlamaSdpaAttention(LlamaSdpaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.pruned_heads = list()
        self.register_buffer('pruned_kv_counts',
                             torch.full((self.num_key_value_heads,),
                                        self.num_key_value_groups, dtype=torch.long), False)
    
    def prune_heads(self, prune_heads: List[int]):
        self.pruned_heads = sorted(prune_heads) if prune_heads is not None else prune_heads
        keep_heads = self.get_heads(self.num_heads, self.pruned_heads)
        keep_idxs = torch.tensor(self.get_keep_indices(keep_heads, self.head_dim), dtype=torch.long,
                                 device=self.q_proj.weight.device)
        keep_kv_heads = self.get_keep_kv_heads(keep_heads, self.num_key_value_groups)
        keep_kv_idxs = torch.tensor(self.get_keep_indices(keep_kv_heads, self.head_dim), dtype=torch.long,
                                    device=self.q_proj.weight.device)
        self.pruned_kv_counts.data = torch.tensor(
            self.build_pruned_kv_counts(keep_heads, self.num_key_value_groups), dtype=torch.long,
            device=self.pruned_kv_counts.device)
        self.num_key_value_groups = max(self.pruned_kv_counts).item()
        self.prune_linear(self.q_proj, keep_idxs, 0)
        self.prune_linear(self.o_proj, keep_idxs, 1)
        self.num_heads = self.num_heads - len(self.pruned_heads)
        
        self.prune_linear(self.k_proj, keep_kv_idxs, 0)
        self.prune_linear(self.v_proj, keep_kv_idxs, 0)
        self.num_key_value_heads = len(keep_kv_heads)
    
    @staticmethod
    def build_pruned_kv_counts(keep_heads, num_key_value_groups) -> list[int]:
        kv_counts = dict()
        for i in keep_heads:
            group_idx = i // num_key_value_groups
            if group_idx not in kv_counts:
                kv_counts[group_idx] = 0
            kv_counts[group_idx] += 1
        return list(kv_counts.values())
    
    @staticmethod
    def prune_linear(to_prune: nn.Linear, keep_idxs, dim):
        # TODO add support for pruning biases (not needed with Llama 3 OOTB)
        pruned_weights = torch.index_select(to_prune.weight, dim, keep_idxs)
        to_prune.weight.data = pruned_weights
    
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

        ks_pos_grouped_per_head = self.repeat_kv(ks_pos_per_kvhead, self.pruned_kv_counts)
        vs_pos_grouped_per_head = self.repeat_kv(vs_per_kvhead, self.pruned_kv_counts)
        
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
    def repeat_kv(states_per_kvhead: torch.Tensor, pruned_kv_counts) -> torch.Tensor:
        """
        The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
        (batch, num_attention_heads, seqlen, head_dim)
        """
        # Repeat states along the 1st dimension (specific key_value_heads) according to `pruned_kv_counts`
        return states_per_kvhead.repeat_interleave(pruned_kv_counts, dim=1)


LLAMA_ATTENTION_CLASSES['sdpa_pruned'] = PrunedLlamaSdpaAttention
