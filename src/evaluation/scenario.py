import logging
from functools import partial
from typing import List, Type, Set

import torch
from datasets import load_dataset

from transformers import AutoConfig, AutoTokenizer
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import PaddingStrategy

from src.evaluation.evaluation import EnergyEvaluation, PerplexityEvaluation, Evaluation
from src.evaluation.pruning import PerLayerStrategy, PruningStrategy
from src.metrics.metrics_manager import EnergyLogger, PerplexityLogger
from src.models.model_resolution import LLMType

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


class DuplicateEvaluationError(Exception):
    pass


class EvaluationScenario:
    def __init__(self, model_path: str = 'meta-llama/Meta-Llama-3-8B', *,
                 batch_size, llm_type: LLMType = LLMType.LLAMA_3,
                 rng_seed: int = 633, # TODO should we get rid of rng_seed? we're not doing any sampling
                 scenario_name: str,
                 dataset: tuple = ("Salesforce/wikitext", 'wikitext-2-v1'),
                 device='cuda'):
        self.model_path = model_path
        self.llm_type = llm_type
        self.scenario_name = scenario_name
        self.device = device
        self.batch_size = batch_size
        torch.manual_seed(rng_seed)

        self.config = AutoConfig.from_pretrained(model_path)

        self.num_attention_heads = self.config.num_attention_heads

        self.dataset = dataset

        self.deferred_warmup = []
        self.deferred_evaluations = []

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, device=self.device)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.energy_logger = EnergyLogger(scenario_name)
        self.perplexity_logger = PerplexityLogger(scenario_name)

    @staticmethod
    def __get_type_name(types: List[Type[Evaluation]]) -> str:
        return "_".join(t.__name__ for t in types)

    def add_baseline_evaluation(self, capture_energy=False, capture_perplexity=False, repetitions=1,
                                warmup_repetitions=0):
        return self._add_evaluation(capture_energy, capture_perplexity, repetitions, False,
                                    warmup_repetitions, None, "baseline")

    def add_warmup_evaluation(self, capture_energy=False, capture_perplexity=False, repetitions=1):
        return self._add_evaluation(capture_energy, capture_perplexity, repetitions, False,
                                    0, None, "warmup")

    def _add_evaluation(self, capture_energy: bool, capture_perplexity: bool, repetitions: int,
                        is_warmup_evaluation: bool,
                        warmup_repetitions, pruning_strategy: PruningStrategy|None, evaluation_name):

        kwargs = self.__get_default_kwargs()
        kwargs['repetitions'] = repetitions
        kwargs['warmup_repetitions'] = warmup_repetitions
        kwargs['label'] = f'{self.scenario_name}-{evaluation_name}'
        kwargs['pruning_strategy'] = pruning_strategy

        chain_of_command: List[Type[Evaluation]] = []
        if capture_perplexity:
            chain_of_command.append(PerplexityEvaluation)
            kwargs['perplexity_logger'] = self.perplexity_logger
        if capture_energy:
            chain_of_command.append(EnergyEvaluation)
            kwargs['energy_logger'] = self.energy_logger
        if len(chain_of_command) == 0:
            chain_of_command.append(Evaluation)
        type_name: str = "_".join(t.__name__ for t in chain_of_command)
        evaluation_type = type(type_name, tuple(chain_of_command), dict())

        deferred = partial(evaluation_type, **kwargs)
        if is_warmup_evaluation:
            self.deferred_warmup.append(deferred)
        else:
            self.deferred_evaluations.append(deferred)
        return self

    def add_pruned_evaluation(self, *, pruning_strategy, capture_energy=False, capture_perplexity=False,
                              evaluation_name, repetitions: int = 1, warmup_repetitions=0):
        return self._add_evaluation(capture_energy, capture_perplexity, repetitions, False,
                                    warmup_repetitions, pruning_strategy, evaluation_name)

    def __get_default_kwargs(self):
        default_kwargs = dict()
        default_kwargs['model_path'] = self.model_path
        default_kwargs['scenario_name'] = self.scenario_name
        default_kwargs['device'] = self.device
        default_kwargs['llm_type'] = self.llm_type
        return default_kwargs

    def execute(self):
        # FIXME this is janky AF, replace with a more general purpose test data provider
        # This assumes that the dataset, after being filtered, will have more test rows available than specified batch size
        evaluation_data = load_dataset(*self.dataset)["test"].filter(
            lambda ex: ex["text"] and len(ex["text"].strip()) > 500
        )

        batch = evaluation_data.select(range(self.batch_size))
        prompts_batch = [example["text"] for example in batch]
        tokens_batch = self._tokenize(prompts_batch)

        for deferred_evaluation in self.deferred_warmup:
            evaluation = deferred_evaluation()
            evaluation.evaluate(tokens_batch)
            del evaluation
        for deferred_evaluation in self.deferred_evaluations:
            evaluation = deferred_evaluation()
            evaluation.evaluate(tokens_batch)
            del evaluation
        del tokens_batch
        return self

    def _tokenize(self, prompt):
        # Warning - hard dependency on pytorch tensors here
        return self.tokenizer(prompt, return_tensors='pt', padding=PaddingStrategy.LONGEST, padding_side='left',
                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=512).to(self.device)


class PerLayerEvaluationScenario(EvaluationScenario):

    def add_per_layer_evaluations(self, *, pruning_strategy_type: Type[PerLayerStrategy], capture_energy=False,
                                  capture_perplexity=False, layer_range=None, repetitions: int = 1,
                                  warmup_repetitions=0, evaluation_name_suffix=''):
        if layer_range is not None:
            first_layer, final_layer = layer_range
        else:
            first_layer = 0
            final_layer = self.config.num_hidden_layers  # FIXME this is a Llama3 specific attr

        if (final_layer <= first_layer
                or final_layer - first_layer > self.config.num_hidden_layers
                or final_layer < 0 or first_layer < 0):
            error_message = f"Invalid layer range specified: {layer_range}, model layer_range=(0, {self.config.num_hidden_layers})"
            raise Exception(error_message)

        for layer_idx in range(first_layer, final_layer):
            evaluation_name = f'pruned-layer-{layer_idx}{evaluation_name_suffix}'
            pruning_strategy_instance = pruning_strategy_type(layer_idx)
            self.add_pruned_evaluation(pruning_strategy=pruning_strategy_instance, capture_energy=capture_energy,
                                       capture_perplexity=capture_perplexity, repetitions=repetitions,
                                       warmup_repetitions=warmup_repetitions, evaluation_name=evaluation_name)
        return self
