from llama.pruning.attention_pruning import LlamaModelPruner
from transformers import AutoTokenizer, LlamaForCausalLM

import torch

model_name = 'meta-llama/Meta-Llama-3-8B'
torch.manual_seed(633)
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='cuda')
print(model)
pruner = LlamaModelPruner(model)

# pruner.prune_heads( {0: [0, 1]})
# pruner.prune_layers([16, 17])
pruner.prune_heads( {0: list(range(0, 32))})
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')
print(model)

tokenize = AutoTokenizer.from_pretrained(model_name, use_fast=True)
prompt = "The Mark of Zorro is a film about "
tokens = tokenize(prompt, return_tensors='pt')
input_ids = tokens["input_ids"].to(model.device)
attention_mask = tokens["attention_mask"].to(model.device)
evaluation = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    do_sample=True,
    top_k=0,
    max_length=500,
    top_p=0.9,
    temperature=0.85,
    repetition_penalty=1.1
)
generated = tokenize.decode(evaluation[0])
print(generated)
