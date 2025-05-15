import logging
from typing import List, Type

import torch
from datasets import load_dataset

from transformers import AutoConfig, AutoTokenizer
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import PaddingStrategy

from src.evaluation.evaluation import EnergyEvaluation, PerplexityEvaluation, PrunedEvaluation, Evaluation
from src.models.model_resolution import LLMType

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


class EvaluationScenario:
    def __init__(self, model_path: str = 'meta-llama/Meta-Llama-3-8B', *,
                 supports_attn_pruning: bool = True, batch_size: 5,
                 llm_type: LLMType = LLMType.LLAMA_3, evaluation_row_count: int = 20,
                 rng_seed: int = 633, scenario_name: str, dataset: tuple = ("Salesforce/wikitext", 'wikitext-2-v1'),
                 device='cuda'):
        self.layer_range = None
        self.model_path = model_path
        self.llm_type = llm_type
        self.evaluation_row_count = evaluation_row_count
        self.scenario_name = scenario_name
        self.device = device
        self.supports_attn_pruning = supports_attn_pruning
        self.batch_size = batch_size
        torch.manual_seed(rng_seed)

        self.config = AutoConfig.from_pretrained(model_path)

        self.num_attention_heads = self.config.num_attention_heads

        self.dataset = dataset

        self.deferred_warmup = []
        self.deferred_baseline = []
        self.deferred_pruned = []

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, device=self.device)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def __get_type_name(types: List[Type[Evaluation]]) -> str:
        return "_".join(t.__name__ for t in types)

    def baseline_evaluations(self, capture_energy=False, capture_perplexity=False, repetitions=1):
        return self.__non_pruned(capture_energy, capture_perplexity, repetitions, False)

    def warmup_evaluations(self, capture_energy=False, capture_perplexity=False, repetitions=1):
        return self.__non_pruned(capture_energy, capture_perplexity, repetitions, True)

    def __non_pruned(self, capture_energy: bool, capture_perplexity: bool, repetitions: int, warmup: bool):
        chain_of_command: List[Type[Evaluation]] = []
        if capture_perplexity:
            chain_of_command.append(PerplexityEvaluation)
        if capture_energy:
            chain_of_command.append(EnergyEvaluation)
        if len(chain_of_command) == 0:
            chain_of_command.append(Evaluation)
        type_name: str = "_".join(t.__name__ for t in chain_of_command)
        evaluation_type = type(type_name, tuple(chain_of_command), dict())

        kwargs = self.__get_default_kwargs()
        kwargs['repetitions'] = repetitions
        label_suffix = 'warmup' if warmup else 'baseline'
        label = f'scenario-{self.scenario_name}-{label_suffix}'
        kwargs['label'] = label
        deferred = lambda: evaluation_type(**kwargs)
        if warmup:
            self.deferred_warmup.append(deferred)
        else:
            self.deferred_baseline.append(deferred)
        return self


    def add_pruned(self, *, pruning_strategy=None, capture_energy=False, capture_perplexity=False, layer_range=None, evaluation_name, repetitions=1):

        if layer_range is not None:
            first_layer, final_layer = layer_range
        else:
            first_layer = 0
            final_layer = self.config.num_hidden_layers - 1  # FIXME this is a Llama3 specific attr

        if (final_layer <= first_layer
                or final_layer - first_layer > self.config.num_hidden_layers
                or final_layer < 0 or first_layer < 0):
            error_message = f"Invalid layer range specified: {layer_range}, model layer_range={self.config.num_layers()}"
            raise Exception(error_message)
        chain_of_command: List[Type[Evaluation]] = []
        if pruning_strategy is not None:
            chain_of_command.append(PrunedEvaluation)
        if capture_perplexity:
            chain_of_command.append(PerplexityEvaluation)
        if capture_energy:
            chain_of_command.append(EnergyEvaluation)
        if len(chain_of_command) == 0:
            chain_of_command.append(Evaluation)
        type_name: str = "_".join(t.__name__ for t in chain_of_command)
        evaluation_type = type(type_name, tuple(chain_of_command), dict())

        default_kwargs = self.__get_default_kwargs()
        default_kwargs['repetitions'] = repetitions

        for layer_idx in range(first_layer, final_layer + 1):
            label = f'scenario-{self.scenario_name}-evaluation-{evaluation_name}-layer{layer_idx}'
            kwargs = default_kwargs.copy()
            kwargs['label'] = label
            self.deferred_pruned.append(lambda: evaluation_type(**kwargs))
        return self

    def __get_default_kwargs(self):
        default_kwargs = dict()
        default_kwargs['model_path'] = self.model_path
        default_kwargs['dataset'] = self.dataset
        default_kwargs['evaluation_row_count'] = self.evaluation_row_count
        default_kwargs['scenario_name'] = self.scenario_name
        default_kwargs['supports_attn_pruning'] = self.supports_attn_pruning
        default_kwargs['device'] = self.device
        default_kwargs['batch_size'] = self.batch_size
        default_kwargs['tokenizer'] = self.tokenizer
        default_kwargs['llm_type'] = self.llm_type
        return default_kwargs

    def add_warmup(self):
        return self

    def execute(self):
        tokens_by_batch = []
        evaluation_data = load_dataset(*self.dataset)["test"].filter(
            lambda ex: ex["text"] and len(ex["text"].strip()) > 500
        )
        num_batches = (self.evaluation_row_count + self.batch_size - 1) // self.batch_size

        for batch_index in range(num_batches):
            start_idx = batch_index * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.evaluation_row_count)
            batch = evaluation_data.select(range(start_idx, end_idx))
            prompts_batch = [example["text"] for example in batch]
            tokens_batch = self._tokenize(prompts_batch)
            tokens_by_batch.append(tokens_batch)

        for deferred_evaluation in self.deferred_warmup:
            evaluation = deferred_evaluation()
            evaluation.evaluate(tokens_by_batch)
            del evaluation
        for deferred_evaluation in self.deferred_baseline:
            evaluation = deferred_evaluation()
            evaluation.evaluate(tokens_by_batch)
            del evaluation
        for deferred_evaluation in self.deferred_pruned:
            evaluation = deferred_evaluation()
            evaluation.evaluate(tokens_by_batch)
            del evaluation
        del tokens_by_batch
        return self

    def _tokenize(self, prompt):
        return self.tokenizer(prompt, return_tensors='pt', padding=PaddingStrategy.LONGEST, padding_side='left',
                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=512).to(self.device)
