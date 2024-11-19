from llama.pruning.attention_pruning import LlamaModelPruner
from transformers import AutoTokenizer, LlamaForCausalLM

import torch

model_name = 'meta-llama/Meta-Llama-3-8B'
# torch.manual_seed(633)
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='cuda')
print(model)
pruner = LlamaModelPruner(model)

pruner.prune_heads({0: [1, 6, 12, 16, 31], 3: [2, 6], 4: [5, 9], 25: [19]})
# pruner.prune_heads({0: [0]})
# pruner.prune_heads( {1: list(range(0, 26)), 3: [2, 6], 4: [5, 9], 25: [19]})
# pruner.prune_layers([16, 17])
# pruner.prune_heads( {0: list(range(0, 32))})
# pruner.prune_heads({0: [0,1,2,3,4,5,6,7]})
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')
print(model)

tokenize = AutoTokenizer.from_pretrained(model_name, use_fast=True)
prompt = ("[SYSTEM] You are a film summary assistant. Briefly Provide the names of the main characters and the director of the moving. Summarize the plot of the movie in great detail.[/SYSTEM]"
          "[USER] Provide a summary to The Mask of Zorro (1998 film)[/USER]"
          "[ASSISTANT]")
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
