import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

model_path = 'D:/learn/llm/nlp-basics/model/Qwen2.5-0.5B'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def ask(prompt,max_length = 200):
    inputs = tokenizer(prompt,return_tensors = "pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,max_length = max_length,do_sample = True,
            top_p = 0.9,temperature = 0.7,pad_token_id = tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0],skip_special_tokens = True)


if __name__ == '__main__':
    # 同一个问题，三种 prompt
    question = "什么是强化学习"

    # 1.零样本(zero-shot):直接问
    prompt_zero = question
    print(f"【Zero-shot】\n{ask(prompt_zero)}\n")

    # 2.少样本(few-shot):给几个例子
    prompt_few = """以下是一些问答示例：

  Q: 什么是机器学习？
  A: 机器学习是人工智能的一个分支，让计算机从数据中学习规律。

  Q: 什么是深度学习？
  A: 深度学习是机器学习的一个子集，使用多层神经网络学习数据的层次化表示。

  现在请直接回答以下问题，不需要重复格式：
  Q: 什么是强化学习？
  A:"""
    print(f"【Few-shot】\n{ask(prompt_few)}\n")

    # 3.思维链(CoT):让模型一步步思考
    prompt_cot = """请按以下步骤回答问题，每步用一句话：

  步骤1: 强化学习是什么？
  步骤2: 它和监督学习有什么区别？
  步骤3: 举一个具体例子。

  问题：什么是强化学习？
  回答：
  步骤1:"""
    print(f"【CoT】\n{ask(prompt_cot)}\n")

