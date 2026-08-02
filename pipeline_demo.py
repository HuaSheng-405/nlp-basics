import torch
from transformers import pipeline


classifier = pipeline("sentiment-analysis")
result = classifier("I love this movie!")
print(result)

classifier = pipeline("sentiment-analysis",model="uer/roberta-base-finetuned-jd-binary-chinese")
result = classifier("这个东西质量太差了")
print(result)