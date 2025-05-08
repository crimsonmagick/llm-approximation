import logging
import torch

from transformers import AutoConfig

from src.evaluation.evaluation_runner import EvaluationRunner
from src.models.model_resolution import LLMType

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


class EvaluationScenario:
    def __init__(self, model_path: str = 'meta-llama/Meta-Llama-3-8B', *,
                 llm_type: LLMType = LLMType.LLAMA_3, evaluation_row_count: int = 20,
                 rng_seed: int = 633, layer_range=None, scenario_name: str, supports_attn_pruning: bool,
                 dataset: tuple = ("Salesforce/wikitext", 'wikitext-2-v1'), device='cuda'):
        self.model_path = model_path
        self.llm_type = llm_type
        self.evaluation_row_count = evaluation_row_count
        self.scenario_name = scenario_name
        self.supports_attn_pruning = supports_attn_pruning
        self.device = device
        torch.manual_seed(rng_seed)
        
        config = AutoConfig.from_pretrained(model_path)
        
        if layer_range is not None:
            first_layer, final_layer = layer_range
        else:
            first_layer = 0
            final_layer = config.num_hidden_layers - 1 # FIXME this is a Llama3 specific attr
        
        if (final_layer <= first_layer
            or final_layer - first_layer > config.num_hidden_layers
            or final_layer < 0 or first_layer < 0):
            error_message = f"Invalid layer range specified: {layer_range}, model layer_range={config.num_layers()}"
            raise Exception(error_message)
        
        self.num_attention_heads = config.num_attention_heads
        
        self.layer_range = (first_layer, final_layer)
        self.dataset = dataset
    
    def runner(self, *, results_path) -> EvaluationRunner:
        return EvaluationRunner(self.model_path, self.dataset, evaluation_row_count=self.evaluation_row_count,
                                scenario_name=self.scenario_name, layer_range=self.layer_range,
                                supports_attn_pruning=self.supports_attn_pruning, device=self.device,
                                num_heads=self.num_attention_heads, llm_type=self.llm_type,
                                results_path=results_path)

