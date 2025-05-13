from abc import ABC, abstractmethod


class PruningStrategy(ABC):
    
    @abstractmethod
    def __call__(self, model):
        pass
    
    
class EveryOtherHeads(PruningStrategy):
    
    def __call__(self, model):
    