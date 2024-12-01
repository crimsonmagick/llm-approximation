import logging
import string
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as functional

from llm_type import LLMType
from src.llama.models.modeling_pruned_llama import PrunedLlamaForCausalLM
from src.metrics.metrics import capture_evaluation, capture_loss
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import PaddingStrategy


class LargeLanguageModelFacade(ABC):
    
    def __init__(self, llm_type: LLMType, model, model_path: string, device: string):
        logging.basicConfig(level=logging.INFO, force=True)
        self.logger = logging.getLogger(__name__)
        self.llm_type = llm_type
        self.model = model
        self.model_path = model_path
        self.device = device
        model.eval()
    
    @abstractmethod
    def detokenize(self, tokens):
        pass
    
    @abstractmethod
    def tokenize(self, prompt):
        pass
        
    @abstractmethod
    def vocab_size(self):
        pass
    
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
        return evaluation[0], evaluation.shape[0] * evaluation.shape[1]
    
    @capture_loss
    def per_token_losses(self, tokens):
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        labels = input_ids.clone() # labels are derived from input
        labels = labels[:, 1:].contiguous() # Drop the first label - it isn't useful as no logits are generated for it
        labels = labels.view(-1)  # Flatten the labels to a vector, [batch_size * labels_sequence_length], in preperation for cross_entroy calculation
        logits = logits[:, :-1].contiguous()  # Last logit has no label to compare with
        logits = logits.view(-1, logits.size(-1))  # Flatten the logits to a matrix [batch_size * sequence_length, vocab_size]
        per_token_loss = functional.cross_entropy(logits, labels, reduction='none') # vector of per token losses
        attention_mask_vector = attention_mask[:, :-1].reshape(-1).contiguous()
        return per_token_loss * attention_mask_vector # apply the attention mask to remove padding, which can skew perplexity measurements


class MissingParameterException(Exception):
    def __init__(self, param_name):
        super().__init__(f'Missing parameter "{param_name}"')


class LlamaFacade(LargeLanguageModelFacade):
    
    def __init__(self, llm_type: LLMType, model_path: string, use_fast=True, device: string = "cuda", pruned=False):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        self.dtype = torch.bfloat16
        if pruned:
            model = PrunedLlamaForCausalLM.from_pretrained(model_path, torch_dtype=self.dtype, device_map=self.device)
        else:
            model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=self.dtype, device_map=self.device)
        super(LlamaFacade, self).__init__(llm_type, model, model_path, device)
    
    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens)
    
    def tokenize(self, prompt):
        return self.tokenizer(prompt, return_tensors='pt', padding=PaddingStrategy.LONGEST, padding_side='left',
                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=512).to(self.model.device)
    
    def vocab_size(self):
        return self.tokenizer.vocab_size

class TorchLlamaLoaderFacade(LargeLanguageModelFacade):
    def __init__(self, llm_type: LLMType, model_path: string, device: string = "cuda"):
        self.model_path = model_path
        self.device = device
        checkpoint = torch.load(model_path)
        model = checkpoint["model"].to(self.device)
        self.tokenizer = checkpoint["tokenizer"]
        super().__init__(llm_type, model, model_path, device)
    
    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens.data, skip_special_tokens=True)
    
    def tokenize(self, prompt):
        tokens = self.tokenizer(prompt, return_tensors='pt', )
        tokens["input_ids"] = tokens["input_ids"].to(self.model.device)
        tokens["attention_mask"] = tokens["attention_mask"].to(self.model.device)
        return tokens
    
    def vocab_size(self):
        return self.tokenizer.vocab_size()
    
