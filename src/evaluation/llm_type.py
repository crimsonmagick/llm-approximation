from enum import Enum


class LLMType(Enum):
    LLAMA_2 = 'llama2'
    LLAMA_3 = 'llama3'
    PRUNED = 'pruned'
    BERT = 'BERT'
