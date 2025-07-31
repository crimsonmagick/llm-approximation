from torch import bfloat16
from transformers import AutoTokenizer, Qwen2ForCausalLM
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import PaddingStrategy

from src.models.llama import PrunedLlamaForCausalLM
from src.models.qwen.modeling_pruned_qwen import PrunedQwen2ForCausalLM

LLAMA_LLM_PATH = '/home/welby/models/Meta-Llama-3-8B/'
QWEN_LLM_PATH = '/home/welby/models/Qwen2-7B'
DEVICE = 'cuda'

if __name__ == '__main__':
    # model =  PrunedLlamaForCausalLM.from_pretrained(LLAMA_LLM_PATH, torch_dtype=bfloat16, device_map='cuda')
    model = PrunedQwen2ForCausalLM.from_pretrained(QWEN_LLM_PATH, torch_dtype=bfloat16, device_map='cuda')
    # for i in range(0, 2):
    #     model.prune_heads({i: list(range(0, 32, 2))})
    prompt = 'Salutations, I'
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_LLM_PATH, use_fast=True, device=DEVICE)
    tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(prompt, return_tensors='pt', padding=PaddingStrategy.LONGEST, padding_side='left',
                       truncation=TruncationStrategy.LONGEST_FIRST, max_length=512).to(DEVICE)
    outputs = model.generate(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        max_new_tokens=50,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.1
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded)
