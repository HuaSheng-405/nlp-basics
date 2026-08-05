from pydantic import BaseModel,Field
import json
from typing import Optional
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

# ========== 一、Entity Extraction 模板 ==========
EXTRACT_ENTITY_PROMPT = """从以下文本中提取信息。严格按照 JSON Schema 输出，不要输出任何其他内容。

  注意：
  - amount_billion 必须是纯数字（不要加"亿"或"元"等单位）。例如"200亿元人民币"应写成 200.0
  - date 格式必须为 YYYY-MM-DD

  {{
    "company": "公司名称",
    "location": "城市",
    "amount_billion": 0.0,
    "date": "YYYY-MM-DD"
  }}

  文本：{text}
  输出："""

class EntityExtraction(BaseModel):
    company:str = ""
    location:str = ""
    amount_billion:float = 0.0
    date:str = ""

# ========== 二、情感分类模板 =======================
SENTIMENT_PROMPT = """只输出 JSON，不要解释。

  {{"sentiment": "positive", "confidence": 0.95, "reason": "文本内容较积极"}}

  文本：这个东西质量太差了，用了三次就坏了
  输出：{{"sentiment": "negative", "confidence": 0.98, "reason": "用户投诉产品质量"}}

  文本：{text}
  输出："""

class SentimentResult(BaseModel):
    sentiment:str
    confidence:float
    reason:str

# ========== 三、带重试的生成函数 ====================
def generate_validate(model,tokenizer,device,prompt_template,text,pydantic_class,max_retries = 2):
    """生成JSON并校验,失败自动重试"""
    prompt = prompt_template.format(text = text)
    for attempt in range(max_retries + 1):
        inputs = tokenizer(prompt,return_tensors = "pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,max_length = 200,do_sample = False,
                pad_token_id = tokenizer.eos_token_id
            )
        raw = tokenizer.decode(output[0],skip_special_tokens = True)
        print(f"[原始输出] {raw[-200:]}")
        #从输出中提取json
        try:
            parts = raw.split("输出：")
            raw_json = parts[1].strip() if len(parts) >= 2 else raw.strip()
            json_str = extract_first_json(raw_json)
            data = json.loads(json_str)
            result = pydantic_class(**data)
            return result
        except Exception as e:
            print(f"[尝试 {attempt+1}/{max_retries+1}] 校验失败: {e}")
            if attempt == max_retries:
                print("重试次数耗尽,返回None")
                return None

# ========== 四、修复:只取第一个JSON对象 ====================
def extract_first_json(text: str):
    """从文本中提取第一个合法的 JSON 对象"""
    # 找到第一个 { 和匹配的 }
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return text

# ========== 五、测试 ====================
if __name__ == "__main__":
    model_path = "D:/learn/llm/nlp-basics/model/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 测试 NLPEntity提取
    text = "2024年3月15日，阿里巴巴集团宣布在杭州投资200亿元建设云计算数据中心"
    result = generate_validate(
        model,tokenizer,device,EXTRACT_ENTITY_PROMPT,text,EntityExtraction
    )
    if result:
        print(f"提取成功: {result.model_dump()}")

    # 测试情感分析
    text2 = "这个东西质量太差了，用了三次就坏了"
    result2 = generate_validate(
        model,tokenizer,device,SENTIMENT_PROMPT,text2,SentimentResult
    )
    if result2:
        print(f"情感:{result2.model_dump()}")
