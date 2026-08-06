from openai import OpenAI

client = OpenAI(
    base_url = "http://localhost:11434/v1",
    api_key = "ollama",
)

response = client.chat.completions.create(
    model = "deepseek-r1:7b",
    messages = [
        {"role":"user","content":"用一句话解释什么是 Transformer"},
    ],
    temperature = 0.7,
)

print(response.choices[0].message.content)