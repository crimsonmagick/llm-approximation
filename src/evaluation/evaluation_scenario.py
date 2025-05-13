import logging
import torch

from transformers import AutoConfig

from src.models.model_resolution import LLMType

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


class EvaluationScenario:
    def __init__(self, model_path: str = 'meta-llama/Meta-Llama-3-8B', *,
                 llm_type: LLMType = LLMType.LLAMA_3, evaluation_row_count: int = 20,
                 rng_seed: int = 633, scenario_name: str, dataset: tuple = ("Salesforce/wikitext", 'wikitext-2-v1'),
                 device='cuda'):
        self.layer_range = None
        self.model_path = model_path
        self.llm_type = llm_type
        self.evaluation_row_count = evaluation_row_count
        self.scenario_name = scenario_name
        self.device = device
        torch.manual_seed(rng_seed)

        self.config = AutoConfig.from_pretrained(model_path)

        self.num_attention_heads = self.config.num_attention_heads

        self.dataset = dataset
        self.warmup_evaluations = []
        self.baseline_evaluations = []
        self.pruned_evaluations = []

    def baseline(self):
        return self

    def pruned(self, *, pruning_strategy, layer_range=None):

        if layer_range is not None:
            first_layer, final_layer = layer_range
        else:
            first_layer = 0
            final_layer = self.config.num_hidden_layers - 1  # FIXME this is a Llama3 specific attr

        if (final_layer <= first_layer
                or final_layer - first_layer > config.num_hidden_layers
                or final_layer < 0 or first_layer < 0):
            error_message = f"Invalid layer range specified: {layer_range}, model layer_range={self.config.num_layers()}"
            raise Exception(error_message)
        self.layer_range = (first_layer, final_layer)
        return self

    def warmup(self):
        return self

    def execute(self, layers=None):
        return self
