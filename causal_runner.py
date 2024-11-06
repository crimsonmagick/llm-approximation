from llama.pruning.attention_pruning import LlamaAttentionPruner
from transformers import AutoTokenizer, LlamaForCausalLM

import torch

model_name = 'meta-llama/Meta-Llama-3-8B'
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='cuda')
print(model)
pruner = LlamaAttentionPruner(model)

# heads_to_prune = {
#     0: [0]
# }
pruner.prune(1)
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')
print(model)

tokenize = AutoTokenizer.from_pretrained(model_name, use_fast=True)
prompt = "The Mark of Zorro is"
tokens = tokenize(prompt, return_tensors='pt')
input_ids = tokens["input_ids"].to(model.device)
attention_mask = tokens["attention_mask"].to(model.device)
with torch.no_grad():
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

