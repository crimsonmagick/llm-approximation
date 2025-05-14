from abc import ABC, abstractmethod


class PruningStrategy(ABC):
    
    @abstractmethod
    def __call__(self, model, layer_idx):
        pass
    
    
class EveryOtherHead(PruningStrategy):
    
    def __call__(self,  model, layer_idx):
        num_heads = model.config.num_attention_heads
        head_idxs = list(range(0, num_heads, 2))
        model.prune_heads({layer_idx: head_idxs})
        return model
