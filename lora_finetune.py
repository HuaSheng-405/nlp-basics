"""LoRA 微调 Qwen2.5-7B（视频脚本领域）"""
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,TrainingArguments,Trainer,DataCollatorForSeq2Seq,BitsAndBytesConfig
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
from datasets import Dataset
import bitsandbytes,accelerate
import json

# ========== 1. 加载模型（4bit 量化省显存） ==========
model_name = "model/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = bnb_config,
    device_map = "auto",
)

# ========== 2. LoRA 配置 ==========
lora_config = LoraConfig(
    r = 8,
    lora_alpha = 8,
    target_modules = ['q_proj','k_proj','v_proj','o_proj'],
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM",
)

model = get_peft_model(model,lora_config)
model.print_trainable_parameters()

# ========== 3. 数据准备 ==========
def format_example(example):
    #prompt = f"### 指令: {example['instruction']}\n### 输出:"
    full = example['text'] + tokenizer.eos_token
    tokenized = tokenizer(full,truncation = True,max_length = 512)
    return {
        "input_ids":tokenized['input_ids'],
        "label":tokenized['input_ids'].copy(),
    }

with open("data/lora_train.json",encoding ="utf-8") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list(raw_data).map(format_example)

# ========== 4. 训练 ==========
training_args = TrainingArguments(
    output_dir = "lora_output",
    num_train_epochs = 3,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 16,
    learning_rate = 2e-4,
    logging_steps = 10,
    save_steps = 100,
    fp16 = True,
    optim = "paged_adamw_8bit",
    report_to = "none",
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = dataset,
    data_collator = DataCollatorForSeq2Seq(tokenizer),
)

trainer.train()

# ========== 5. 保存 ==========
model.save_pretrained("lora_output/final")
tokenizer.save_pretrained("lora_output/final")
print("LoRA权重已保存")