import string
from enum import Enum

from torch import bfloat16
from transformers import PreTrainedModel

from src.models.llama.modeling_pruned_llama import PrunedLlamaForCausalLM


class LLMType(Enum):
    LLAMA_2 = 'llama2'
    LLAMA_3 = 'llama3'
    QWEN_2 = 'qwen2'
    BERT = 'BERT'


def resolve_model(llm_type: LLMType, path: string) -> PreTrainedModel:
    if llm_type == LLMType.LLAMA_3:
        return PrunedLlamaForCausalLM.from_pretrained(path, torch_dtype=bfloat16, device_map='cuda')
    else:
        raise ValueError(f'Model type not supported, llm_type: {llm_type}')
