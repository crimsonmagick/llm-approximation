import logging
import gc

import torch

from large_language_model_service import get_model
from src.pruning.attention_pruning import LlamaModelPruner
from llm_type import LLMType
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


class TestRunner:
    
    def __init__(self, model, dataset_path: tuple):
        self.llm = model
        self.test_data = load_dataset(*dataset_path)["test"].filter(
            lambda ex: ex["text"] and ex["text"].strip() != ""
        )
    
    def batch_evaluate(self, rows_to_evaluate, batch_size=5):
        num_batches = (rows_to_evaluate + batch_size - 1) // batch_size
        for batch_index in range(num_batches):
            start_idx = batch_index * batch_size
            end_idx = min(start_idx + batch_size, rows_to_evaluate)
            batch = self.test_data.select(range(start_idx, end_idx))
            prompts = [example["text"] for example in batch]
            tokens = self.llm.tokenize(prompts)
            self.llm.evaluate(tokens)
            # self.llm.per_token_losses(tokens)
        logger.info(f'Vocab Size: {self.llm.vocab_size()}')

def main():
    model = get_model(LLMType.LLAMA_3, 'meta-llama/Meta-Llama-3-8B')
    # print('--------------------------------')
    # print('ORIGINAL')
    # print('--------------------------------')
    #
    # runner = TestRunner(model, ("Salesforce/wikitext", 'wikitext-2-v1'))
    # runner.batch_evaluate(100)
    #
    # print('--------------------------------')
    # print('PRUNED')
    # print('--------------------------------')

    print("Before pruning:")
    print(torch.cuda.memory_summary())
    pruner = LlamaModelPruner(model.model)
    heads = dict()
    for i in range(0, 32):
        heads[i] = [0, 1, 2, 3]
    pruner.prune_heads(heads)
    gc.collect()
    torch.cuda.empty_cache()
    print("After pruning:")
    print(torch.cuda.memory_summary())
    
    runner = TestRunner(model, ("Salesforce/wikitext", 'wikitext-2-v1'))
    runner.batch_evaluate(100)


if __name__ == '__main__':
    main()
