import logging
import string
from abc import ABC, abstractmethod

import torch

from llm_type import LLMType
from metrics import capture_evaluation
from transformers import AutoTokenizer, LlamaForCausalLM


class LargeLanguageModel(ABC):
    
    def __init__(self, llm_type: LLMType, model_path: string, device: string):
        logging.basicConfig(level=logging.INFO, force=True)
        self.logger = logging.getLogger(__name__)
        self.llm_type = llm_type
        self.model_path = model_path
        self.device = device
    
    @abstractmethod
    def detokenize(self, tokens):
        pass
    
    @abstractmethod
    def tokenize(self, prompt):
        pass
    
    @abstractmethod
    def evaluate(self, tokens, max_length=500):
        pass
    
    @abstractmethod
    def loss(self, tokens):
        pass
    
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


class MissingParameterException(Exception):
    def __init__(self, param_name):
        super().__init__(f'Missing parameter "{param_name}"')


class LlamaLargeLanguageModel(LargeLanguageModel):
    def __init__(self, llm_type: LLMType, model_path: string, use_fast=True, device: string = "cuda"):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        self.dtype = torch.bfloat16
        self.model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=self.dtype, device_map=self.device)
        self.model.eval()
        super(LlamaLargeLanguageModel, self).__init__(llm_type, model_path, device)
    
    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens)
    
    def tokenize(self, prompt):
        return self.tokenizer(prompt, return_tensors='pt',
                              max_length=512).to(self.model.device)
    
    @capture_evaluation
    def evaluate(self, tokens, max_length=500):
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        with torch.no_grad():
            evaluation = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                top_k=50,
                max_length=max_length,
                top_p=0.95,
                temperature=1.0,
            )
        return evaluation[0], evaluation.shape[1]
    
    def loss(self, tokens):
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        labels = input_ids.clone()
        labels = labels[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        attention_mask=attention_mask[:, :-1].contiguous()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss


class PrunedLargeLanguageModel(LargeLanguageModel):
    def __init__(self, llm_type: LLMType, model_path: string, device: string = "cuda"):
        self.model_path = model_path
        self.device = device
        checkpoint = torch.load(model_path)
        self.model = checkpoint["model"].to(self.device)
        self.model.eval()
        self.tokenizer = checkpoint["tokenizer"]
        super().__init__(llm_type, model_path, device)
    
    def tokenize(self, prompt):
        tokens = self.tokenizer(prompt, return_tensors='pt', )
        tokens["input_ids"] = tokens["input_ids"].to(self.model.device)
        tokens["attention_mask"] = tokens["attention_mask"].to(self.model.device)
        return tokens
    
    @capture_evaluation
    def evaluate(self, tokens, max_length=500):
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        labels = input_ids.clone()
        
        with torch.no_grad():
            evaluation = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                top_k=50,
                max_length=max_length,
                top_p=0.95,
                temperature=1.0,
            )
        return evaluation[0], evaluation.shape[1]
    
    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens.data, skip_special_tokens=True)
