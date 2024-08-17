import torch
from transformers import AutoTokenizer, LlamaForCausalLM

# Load the tokenizer and model
model_path = "nvidia/Llama-3.1-Minitron-4B-Width-Base"
# model_path = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = 'cuda'
dtype = torch.bfloat16
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)

# Prepare the input text
prompt = "The Open-Closed Principle is"
inputs = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

# Generate the output
outputs = model.generate(inputs, max_length=500)

# Decode and print the output
output_text = tokenizer.decode(outputs[0])
print(output_text)
