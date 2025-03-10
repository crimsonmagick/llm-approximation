import string

from large_language_model import LargeLanguageModelFacade, LlamaFacade, TorchLlamaLoaderFacade
from llm_type import LLMType

def get_model(llm_type: LLMType, path: string, pruned=False) -> LargeLanguageModelFacade:
    if llm_type == LLMType.LLAMA_2:
        return LlamaFacade(llm_type, path, False, device='cuda')
    elif llm_type == LLMType.LLAMA_3:
        return LlamaFacade(llm_type, path, True, device='cuda', pruned=pruned)
    else:
        raise ValueError(f'Model type not supported, llm_type: {llm_type}')
