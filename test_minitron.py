import logging
import time
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# before
allocated_memory = torch.cuda.memory_allocated(device='cuda')
print(f"BEFORE Allocated Memory: {allocated_memory / 1024**2:.2f} MB")

# Check reserved memory
reserved_memory = torch.cuda.memory_reserved(device='cuda')
print(f"BEFORE Reserved Memory: {reserved_memory / 1024**2:.2f} MB")

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

# loaded
allocated_memory = torch.cuda.memory_allocated(device='cuda')
print(f"LOADED Allocated Memory: {allocated_memory / 1024**2:.2f} MB")

# Check reserved memory
reserved_memory = torch.cuda.memory_reserved(device='cuda')
print(f"LOADED Reserved Memory: {reserved_memory / 1024**2:.2f} MB")

# Generate the output
before_ts_ms = int(time.time() * 1000)
outputs = model.generate(inputs, max_length=500)
after_ts_ms = int(time.time() * 1000)

# Decode and print the output
output_text = tokenizer.decode(outputs[0])
print(output_text)

execution_time_ms = after_ts_ms - before_ts_ms
num_generated_tokens = outputs.shape[1]
average_time_per_token_ms = execution_time_ms / num_generated_tokens

logger.info(f"execution_time={execution_time_ms} ms")
logger.info(f"average_time_per_token={average_time_per_token_ms:.2f} ms")
