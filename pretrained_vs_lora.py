"""对比微调前后效果"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "model/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
prompt = "写一段美食探店视频的开场白："

# ========== 微调前（原始模型） ==========
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu",dtype = torch.float16)
def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=True, top_p=0.9, temperature=0.7)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("【微调前】")
print(generate(prompt))

# ========== 微调后（加载 LoRA） ==========
del model
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",
    dtype = torch.float16,
)
model = PeftModel.from_pretrained(model, "lora_output/final",dtype = torch.float16)
print("\n【微调后】")
print(generate(prompt))