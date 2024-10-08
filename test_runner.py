import logging

from large_language_model_service import get_model
from llm_type import LLMType
from datasets import load_dataset
import sys

sys.path.append('/home/welb/workspace/LLM-Pruner')

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


class TestRunner:
    
    def __init__(self, llm_type, model_path, dataset_path: tuple):
        self.llm = get_model(llm_type, model_path)
        self.dataset = load_dataset(*dataset_path)
    
    def __log_memory(self, prexfix):
        logger.info(f"{prexfix} Allocated Memory: {self.llm.get_allocated_memory() / 1024 ** 2:.2f} MB")
        logger.info(f"{prexfix} Reserved Memory: {self.llm.get_reserved_memory() / 1024 ** 2:.2f} MB")
    
    def batch_evaluate(self, rows_to_evaluate):
        for example in self.dataset["test"].select(range(rows_to_evaluate)):
            text = example["text"]
            tokens = self.llm.tokenize(text)
            evaluation = self.llm.evaluate(tokens)
            generated_tokens = evaluation[0]
            detokenized = self.llm.detokenize(generated_tokens)
            logger.info(f"Prompt: {text}, generated: {detokenized}")


def main():
    # model_path = "nvidia/Llama-3.1-Minitron-4B-Width-Base"
    # model_path = "/home/welb/ai/models/decapoda-research-llama-7B-hf"
    # runner = TestRunner(LLMType.LLAMA_3, "meta-llama/Meta-Llama-3-8B", ("Salesforce/wikitext", 'wikitext-2-v1'))
    # runner = TestRunner(LLMType.LLAMA_2, '/home/welb/ai/models/decapoda-research-llama-7B-hf',
    #                     ("Salesforce/wikitext", 'wikitext-2-v1'))
    runner = TestRunner(LLMType.PRUNED, '/home/welb/workspace/LLM-Pruner/prune_log/llama_prune/pytorch_model.bin',
                        ("Salesforce/wikitext", 'wikitext-2-v1'))
    runner.batch_evaluate(20)


if __name__ == '__main__':
    main()
