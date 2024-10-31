from triton.language import bfloat16

from transformers import AutoTokenizer, LlamaForCausalLM

import torch


def prune_attention_heads(model, heads_to_prune):
    # Calculate new embedding dimension based on pruned heads
    original_embedding_dim = model.model.embed_tokens.embedding_dim
    num_heads = model.model.layers[0].self_attn.num_heads
    head_size = original_embedding_dim // num_heads
    
    # Determine which heads to keep
    all_heads = set(range(num_heads))
    keep_heads = sorted(all_heads - set([head for layer_heads in heads_to_prune.values() for head in layer_heads]))
    
    # Calculate new embedding dimension
    new_embedding_dim = len(keep_heads) * head_size
    
    # Adjust the embedding layer
    embed_weights = model.model.embed_tokens.weight[:, :new_embedding_dim]
    model.model.embed_tokens = torch.nn.Embedding(model.model.embed_tokens.num_embeddings, new_embedding_dim)
    model.model.embed_tokens.weight.data = embed_weights
    
    for layer_idx, heads in heads_to_prune.items():
        attention_layer = model.model.layers[layer_idx].self_attn
        
        # Recalculate keep_indices to reflect the reduced dimension per head
        keep_indices = torch.cat([torch.arange(h * head_size, (h + 1) * head_size) for h in keep_heads])
        
        # Prune q_proj
        new_q_proj_weight = attention_layer.q_proj.weight.data[:, keep_indices]
        attention_layer.q_proj = torch.nn.Linear(new_q_proj_weight.shape[1], new_q_proj_weight.shape[0], bias=False)
        attention_layer.q_proj.weight = torch.nn.Parameter(new_q_proj_weight)
        
        # Prune k_proj
        new_k_proj_weight = attention_layer.k_proj.weight.data[:, keep_indices]
        attention_layer.k_proj = torch.nn.Linear(new_k_proj_weight.shape[1], new_k_proj_weight.shape[0], bias=False)
        attention_layer.k_proj.weight = torch.nn.Parameter(new_k_proj_weight)
        
        # Prune v_proj
        new_v_proj_weight = attention_layer.v_proj.weight.data[:, keep_indices]
        attention_layer.v_proj = torch.nn.Linear(new_v_proj_weight.shape[1], new_v_proj_weight.shape[0], bias=False)
        attention_layer.v_proj.weight = torch.nn.Parameter(new_v_proj_weight)
        
        # Adjust o_proj dimensions accordingly (since output will be smaller)
        out_keep_indices = torch.cat([torch.arange(h * head_size, (h + 1) * head_size) for h in keep_heads])
        new_o_proj_weight = attention_layer.o_proj.weight.data[out_keep_indices, :]
        attention_layer.o_proj = torch.nn.Linear(new_o_proj_weight.shape[1], new_o_proj_weight.shape[0], bias=False)
        attention_layer.o_proj.weight = torch.nn.Parameter(new_o_proj_weight)


model_name = 'meta-llama/Meta-Llama-3-8B'
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='cuda')
print(model)

heads_to_prune = {
    0: [0],  # Prune head 0 in the first layer
}
prune_attention_heads(model, heads_to_prune)
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')
print(model)

tokenize = AutoTokenizer.from_pretrained(model_name, use_fast=True)
prompt = "hello guys, what's up I'm"
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

