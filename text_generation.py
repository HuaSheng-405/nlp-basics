import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

#1.加载GPT-2中文模型
model_path = "D:/learn/llm/nlp-basics/model/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

#GPT-2的tokenizer没有pad_token 补一个
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#2.生成函数
def generate(prompt,strategy = "greedy",max_length = 50,temperature = 1.0,top_k = 50,top_p = 0.9):
    inputs = tokenizer(prompt,return_tensors = "pt").to(device)

    if strategy == "greedy":
        output = model.generate(
            **inputs,max_length = max_length,do_sample = False,
            pad_token_id = tokenizer.eos_token_id
        )
    elif strategy == "beam":
        output = model.generate(
            **inputs,max_length = max_length,num_beams = 4,early_stopping = True,
            pad_token_id = tokenizer.eos_token_id
        )
    elif strategy == "top_k":
        output = model.generate(
            **inputs,max_length = max_length,do_sample = True,top_k = top_k,
            temperature = temperature,pad_token_id = tokenizer.eos_token_id
        )
    elif strategy =="top_p":
        output = model.generate(
            **inputs,max_length = max_length,do_sample = True,top_p = top_p,
            temperature = temperature,pad_token_id = tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0],skip_special_tokens = True).replace(" ","")


#3.对比四种策略
prompts = [
    "今天天气很好，",
    "人工智能的未来是",
    "小明走进教室，发现",
]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"输入: {prompt}")
    print(f"{'='*60}")
    print(f"贪心解码 (greedy):  {generate(prompt, 'greedy')}")
    print(f"束搜索   (beam=4):  {generate(prompt, 'beam')}")
    print(f"Top-K    (k=50):    {generate(prompt, 'top_k')}")
    print(f"Top-P    (p=0.9):   {generate(prompt, 'top_p')}")