import string
from enum import Enum

from torch import bfloat16
from transformers import PreTrainedModel

from src.models.llama.models.modeling_pruned_llama import PrunedLlamaForCausalLM


class LLMType(Enum):
    LLAMA_2 = 'llama2'
    LLAMA_3 = 'llama3'
    BERT = 'BERT'


def get_model(llm_type: LLMType, path: string, pruned=False) -> PreTrainedModel:
    if llm_type == LLMType.LLAMA_3:
        return PrunedLlamaForCausalLM.from_pretrained(path, torch_dtype=bfloat16, device_map='cuda')
    else:
        raise ValueError(f'Model type not supported, llm_type: {llm_type}')
