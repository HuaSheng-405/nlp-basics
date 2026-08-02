from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

#方法1:tokenize - 看分词结果
text = "我爱自然语言处理"
tokens = tokenizer.tokenize(text)
print(f"分词结果:{tokens}")

# 方法2: encode — 转成数字
ids = tokenizer.encode(text)
print(f"编码结果:{ids}")
print(f"解码结果:{tokenizer.decode(ids)}")

# 方法3: __call__ — 批量处理
texts = ["我爱你","Transformer很难但是很有用"]
encoded = tokenizer(texts,padding = True,truncation = True,return_tensors = "pt")
print(f"input_ids shape: {encoded['input_ids'].shape}")
print(f"input_ids:\n{encoded['input_ids']}")
print(f"attention_mask:\n{encoded['attention_mask']}")