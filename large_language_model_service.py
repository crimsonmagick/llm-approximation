import string

from large_language_model import LargeLanguageModel, LlamaLargeLanguageModel, PrunedLargeLanguageModel
from llm_type import LLMType


class LlmMetricsFacade:
    def __init__(self, source_llm):
        self.source_llm = source_llm
        


def get_model(llm_type: LLMType, path: string) -> LargeLanguageModel:
    if llm_type == LLMType.LLAMA_2:
        return LlamaLargeLanguageModel(llm_type, path, False, device='cuda')
    elif llm_type == LLMType.LLAMA_3:
        return LlamaLargeLanguageModel(llm_type, path, True, device='cuda')
    elif llm_type == LLMType.PRUNED:
        return PrunedLargeLanguageModel(llm_type, path, device='cuda')
    else:
        raise ValueError(f'Model type not supported, llm_type: {llm_type}')
