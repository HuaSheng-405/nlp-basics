from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from pydantic import BaseModel,Field
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch

# 加载本地模型
model_path = 'D:/learn/llm/nlp-basics/model/Qwen2.5-0.5B'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def generate(prompt_text,max_length = 200):
    inputs = tokenizer(prompt_text,return_tensors = "pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,max_length = max_length,do_sample = False,
            pad_token_id = tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0],skip_special_tokens = True)

# ========== 一、PromptTemplate：变量复用 ==========
print("=" * 60)
print("一、PromptTemplate")
print("=" * 60)

# 不用 LangChain：字符串拼接，模板散落在代码里
topic = "大语言模型"
old_way = f"请用一句话解释什么是{topic}"
print(f"老方法:{old_way}")

# 用 LangChain：模板和逻辑分离
template = PromptTemplate.from_template("请用一句话解释什么是{topic}")
prompt = template.invoke({"topic":"大语言模型"})
print(f"新方法:{prompt.to_string()}")

# 同一个模板可以更换变量 多次复用
for t in ["强化学习", "注意力机制", "向量数据库"]:
    result = generate(template.invoke({"topic":t}).to_string(),max_length = 80)
    print(f"\n{t}: {result.split(chr(10))[-1][:60]}...")    #char(10)代表换行符\n

# ========== 二、ChatPromptTemplate：系统 + 用户多角色 ==========
print("\n" + "=" * 60)
print("二、ChatPromptTemplate（多角色 prompt）")
print("=" * 60)

chat_template = ChatPromptTemplate.from_messages([
    ("system","你是一个专业的科技记者,擅长用生动的比喻解释技术概念."),
    ("user","请用一个小白能听懂的比喻解释:{concept},不超过100字."),
])

prompt = chat_template.format_messages(concept = "Transformer")
# LangChain 生成的是消息列表，转成纯文本给 Qwen
prompt_text = "\n".join([m.content for m in prompt])
print(prompt_text)
result = generate(prompt_text)
print(f"\n生成结果: {result.split(chr(10))[-1][:100]}...")

# ========== 三、OutputParser：自动解析结构化输出 ==========
print("\n" + "=" * 60)
print("三、PydanticOutputParser")
print("=" * 60)

class MovieReview(BaseModel):
    title:str = Field(description = "电影名称")
    rating:int = Field(description = "1-5星评分")
    summary:str = Field(description = "一句话评价,不超过20个字")

parser = PydanticOutputParser(pydantic_object = MovieReview)

# LangChain 自动生成 format_instructions
template = PromptTemplate(
    template = "根据用户输入提取电影评价信息。\n{format_instructions}\n用户输入：{input}\n输出：",
    input_variables=["input"],
    partial_variables={"format_instructions":parser.get_format_instructions()},
)

print("LangChain 自动生成的格式说明:")
print(parser.get_format_instructions())

review_prompt = template.invoke({"input":"昨天看了《流浪地球3》，打5星，画面震撼剧情紧凑"})
result = generate(review_prompt.to_string(),max_length = 300)
print(f"\n原始输出: {result.split(chr(10))[-1][:150]}...")

# ========== 四、Chain：用 | 串联多个步骤 ==========
print("\n" + "=" * 60)
print("四、LCEL Chain（管道串联）")
print("=" * 60)

# LangChain 的核心设计：用 | 把步骤串起来
# prompt | model | output_parser
# 可惜本地 Qwen 不能直接用 | ，所以这里展示概念：
# 等价于你之前手写的：
# prompt = template.format(...)           ← Step 1: 填模板
# output = generate(prompt)               ← Step 2: 生成
# result = parser.parse(output)           ← Step 3: 解析
# 用 LangChain + OpenAI API 的话就是：
# chain = template | model | parser
# result = chain.invoke({"input": "..."})