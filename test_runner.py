import logging

from large_language_model import LargeLanguageModel
from metrics_recorder import MetricsRecorder
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


class TestRunner:
    
    def __init__(self, model_path, dataset_path: tuple):
        self.llm = LargeLanguageModel(model_path)
        
        self.dataset = load_dataset(*dataset_path)
    
    def __log_memory(self, prexfix):
        logger.info(f"{prexfix} Allocated Memory: {self.llm.get_allocated_memory() / 1024 ** 2:.2f} MB")
        logger.info(f"{prexfix} Reserved Memory: {self.llm.get_reserved_memory() / 1024 ** 2:.2f} MB")
    
    def batch_evaluate(self, rows_to_evaluate):
        metrics_recorder = MetricsRecorder().start()
        generated_token_count = 0
        
        for example in self.dataset["test"].select(range(rows_to_evaluate)):
            text = example["text"]
            tokens = self.llm.tokenize(text)
            evaluation = self.llm.evaluate(tokens)
            generated_tokens = evaluation[0]
            generated_token_count += evaluation.shape[1]
            detokenized = self.llm.detokenize(generated_tokens)
            logger.info(f": {text} || {detokenized}")
        
        energy_usage, execution_time_ms = metrics_recorder.end().get_metrics()
        average_time_per_token_ms = execution_time_ms / generated_token_count
        average_energy_per_token_mj = energy_usage / generated_token_count
        logger.info(f"execution_time={execution_time_ms} ms, energy_usage={energy_usage} mj")
        logger.info(
            f"average_time_per_token={average_time_per_token_ms:.2f} ms, "
            f"average_energy_per_token_mj={average_energy_per_token_mj:.2f} mj")


if __name__ == '__main__':
    # model_path = "nvidia/Llama-3.1-Minitron-4B-Width-Base"
    model_path = "meta-llama/Meta-Llama-3-8B"
    runner = TestRunner(model_path, ("Salesforce/wikitext", 'wikitext-2-v1'))
    runner.batch_evaluate(20)
