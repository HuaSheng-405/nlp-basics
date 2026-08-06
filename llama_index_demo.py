from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import os

# ========== 1. 准备本地 Qwen 作为生成模型 ==========
model_path = "D:/learn/llm/nlp-basics/model/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ========== 2. LlamaIndex 核心流程：加载 → 切分 → 向量化 → 索引 → 检索 ==========
# 加载文档
documents = SimpleDirectoryReader('D:/learn/llm/nlp-basics/data/llama_test').load_data()
print(f"加载文档数:{len(documents)}")

# 设置embedding模型(用本地Hugging Face 不用OpenAI)
Settings.embed_model = HuggingFaceEmbedding(
    model_name ="D:/learn/llm/nlp-basics/model/multilingual-MiniLM"
)

Settings.llm = None     # 不用 LlamaIndex 内置 LLM，调本地 Qwen

# 切分节点
Settings.text_splitter = SentenceSplitter(chunk_size = 100,chunk_overlap = 20)

# 建索引
index = VectorStoreIndex.from_documents(documents)
print(f"索引构建完成")

# 检索
query_engine = index.as_query_engine(similarity_top_k = 2)
retrieved = query_engine.retrieve("什么是Transformer?")
print(f"\n检索到 {len(retrieved)} 条相关片段:")
for i,node in enumerate(retrieved):
    print(f"  [{i+1}] score={node.score:.4f}: {node.text[:80]}...")

# ========== 3. 手动拼一个 RAG prompt 验证检索效果 ==========
print("=" * 60)
print("手动 RAG 测试")
print("=" * 60)

query = "解释一下Transformer和BERT的关系"
context = "\n".join([n.text for n in retrieved])

prompt = f"""根据以下参考资料回答问题。如果资料中没有足够信息，请说明。

参考资料：
{context}

问题：{query}
回答："""

inputs = tokenizer(prompt,return_tensors = "pt").to(device)
with torch.no_grad():
    output = model.generate(
        **inputs,max_length = 200,do_sample = False,
        pad_token_id = tokenizer.eos_token_id
    )
answer = tokenizer.decode(output[0],skip_special_tokens = True)
print(f"问题: {query}")
print(f"检索到的上下文:\n{context}")
print(f"\n模型回答: {answer[-200:]}")