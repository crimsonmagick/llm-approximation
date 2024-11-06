from llama.pruning.attention_pruning import LlamaAttentionPruner
from transformers import AutoTokenizer, LlamaForCausalLM

import torch

model_name = 'meta-llama/Meta-Llama-3-8B'
torch.manual_seed(632)
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='cuda')
print(model)
pruner = LlamaAttentionPruner(model)

# pruner.prune_heads( {0: [0, 1]})
# pruner.prune_layers([7, 8])
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')
print(model)

tokenize = AutoTokenizer.from_pretrained(model_name, use_fast=True)
prompt = "The Mark of Zorro is a film about a masked avenger "
tokens = tokenize(prompt, return_tensors='pt')
input_ids = tokens["input_ids"].to(model.device)
attention_mask = tokens["attention_mask"].to(model.device)
evaluation = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    do_sample=True,
    top_k=50,
    max_length=500,
    top_p=0.95,
    temperature=1.0,
)
generated = tokenize.decode(evaluation[0])
print(generated)
