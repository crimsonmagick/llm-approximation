import string

from large_language_model import LargeLanguageModel, LlamaLargeLanguageModel
from llm_type import LLMType


def get_model(llm_type: LLMType, path: string) -> LargeLanguageModel:
    if llm_type == LLMType.LLAMA_2 or llm_type == LLMType.LLAMA_3:
        return LlamaLargeLanguageModel(llm_type, path, device='cuda')
    else:
        raise ValueError(f'Model type not supported, llm_type: {llm_type}')
