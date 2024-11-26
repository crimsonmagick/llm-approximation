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

def main():
    model = get_model(LLMType.LLAMA_3, 'meta-llama/Meta-Llama-3-8B')
    pruner = LlamaModelPruner(model.model)
    heads = dict()
    for i in range(0, 32):
        heads[i] = list(range(0,16))
    pruner.prune_heads(heads)
    
    runner = TestRunner(model, ("Salesforce/wikitext", 'wikitext-2-v1'))
    runner.batch_evaluate(20)


if __name__ == '__main__':
    main()
