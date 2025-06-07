from abc import ABC, abstractmethod


class PruningException(Exception):
    pass


class PruningStrategy(ABC):

    @abstractmethod
    def __call__(self, model):
        pass


class EveryOtherHead(PruningStrategy):

    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

    def __call__(self, model):
        num_heads = model.config.num_attention_heads
        head_idxs = list(range(0, num_heads, 2))
        prune_params = {self.layer_idx: head_idxs}
        model.prune_heads(prune_params)
        return prune_params


    def metadata(self):
        return
