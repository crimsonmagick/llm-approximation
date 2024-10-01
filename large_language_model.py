import logging
import torch
from transformers import AutoTokenizer, LlamaForCausalLM


class LargeLanguageModel:
    def __init__(self, model_path, device="cuda"):
        logging.basicConfig(level=logging.INFO, force=True)
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        self.dtype = torch.bfloat16
        self.model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=self.dtype, device_map=self.device)
    
    def tokenize(self, prompt):
        return self.tokenizer(prompt, return_tensors='pt',
                              max_length=512).to(self.model.device)
    
    def tokenizer(self):
        return self.tokenizer
    
    def evaluate(self, tokens, max_length=500):
        attention_mask = tokens["attention_mask"]
        input_ids = tokens["input_ids"]
        evaluation = self.model.generate(input_ids, attention_mask=attention_mask, max_length=max_length,
                                         pad_token_id=self.tokenizer.eos_token_id)
        return evaluation
    
    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens)
    
    def get_allocated_memory(self):
        if self.device == "cuda":
            return torch.cuda.memory_allocated(device=self.device)
        self.logger.warning(f"Unknown device={self.device}, return 0 MB as allocated memory")
        return 0
    
    def get_reserved_memory(self):
        if self.device == "cuda":
            return torch.cuda.memory_reserved(device=self.device)
        self.logger.warning(f"Unknown device={self.device}, return 0 MB as reserved memory")
        return 0
