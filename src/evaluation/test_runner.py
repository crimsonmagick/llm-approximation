import logging

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
    
    def __log_memory(self, prexfix):
        logger.info(f"{prexfix} Allocated Memory: {self.llm.get_allocated_memory() / 1024 ** 2:.2f} MB")
        logger.info(f"{prexfix} Reserved Memory: {self.llm.get_reserved_memory() / 1024 ** 2:.2f} MB")
    
    def batch_evaluate(self, rows_to_evaluate):
        for example in self.test_data.select(range(rows_to_evaluate)):
            text = example["text"]
            tokens = self.llm.tokenize(text)
            evaluation = self.llm.evaluate(tokens)
            self.llm.per_token_losses(tokens)
            # generated_tokens = evaluation[0]
            # detokenized = self.llm.detokenize(generated_tokens)
            # logger.info(f"Prompt: {text}, generated: {detokenized}")
        logger.info(f'Vocab Size: {self.llm.vocab_size()}')
    
    # runner = TestRunner(LLMType.LLAMA_3, "nvidia/Llama-3.1-Minitron-4B-Width-Base", ("Salesforce/wikitext", 'wikitext-2-v1'))
    # runner = TestRunner(LLMType.LLAMA_2, '/home/welb/ai/models/decapoda-research-llama-7B-hf',
    #                     ("Salesforce/wikitext", 'wikitext-2-v1'))
    # runner = TestRunner(LLMType.PRUNED, '/home/welb/workspace/LLM-Pruner/prune_log/llama_prune/pytorch_model.bin',
    #                     ("Salesforce/wikitext", 'wikitext-2-v1'))
    # runner =  TestRunner(LLMType.PRUNED, 'google-bert/bert-base-uncased',
    #                                          ("Salesforce/wikitext", 'wikitext-2-v1'))


def main():
    model = get_model(LLMType.LLAMA_3, 'meta-llama/Meta-Llama-3-8B')
    pruner = LlamaModelPruner(model.model)
    heads = dict()
    for i in range(0, 32):
        heads[i] = list(range(0, 31))
    pruner.prune_heads(heads)
    
    runner = TestRunner(model, ("Salesforce/wikitext", 'wikitext-2-v1'))
    runner.batch_evaluate(20)


if __name__ == '__main__':
    main()
