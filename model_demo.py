from transformers import AutoModel,AutoModelForSequenceClassification,AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
texts = ["我爱你", "Transformer很难但是很有用"]
encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 基础模型：输出 hidden states
model = AutoModel.from_pretrained("bert-base-chinese")
with torch.no_grad():
    outputs = model(**encoded)
print(f"last_hidden_state:{outputs.last_hidden_state.shape}")   #torch.Size([2, 10, 768])
print(f"pooler_output:{outputs.pooler_output.shape}")           #torch.Size([2, 768])

# 分类模型：直接输出 logits
clf_model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese",num_labels = 2)
with torch.no_grad():
    outputs = clf_model(**encoded)
print(f"logits:{outputs.logits.shape}")
print(f"logits:{outputs.logits}")