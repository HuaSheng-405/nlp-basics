# nlp-basics

Hugging Face 实战练习：文本分类、NER、文本生成 | BERT/GPT 探索

## 项目内容

| 任务 | 文件 | 模型 | 结果 |
|------|------|------|------|
| Pipeline/Tokenizer/Model | `pipeline_demo.py` `tokenizer_demo.py` `model_demo.py` | — | 三大核心 API 实战 |
| 文本分类 | `text_classification.py` | bert-base-chinese 全参数微调 | **准确率 98.36%** |
| 命名实体识别 | `ner.py` | bert-base-chinese 全参数微调 | **F1 94.65%** |
| 文本生成 | `text_generation.py` | GPT-2 / Qwen2.5-0.5B | 四种解码策略对比 |

## 文件结构

```
nlp-basics/
├── pipeline_demo.py          # HuggingFace Pipeline 一行调用
├── tokenizer_demo.py         # Tokenizer 分词/编码/批量处理
├── model_demo.py             # AutoModel vs AutoModelForSequenceClassification
├── text_classification.py    # THUCNews 新闻分类（10类）
├── ner.py                    # CLUENER2020 命名实体识别（10类实体）
├── text_generation.py        # GPT-2 + Qwen 文本生成对比
├── notes/
│   ├── week2-day5-bert.md    # BERT 论文笔记
│   └── BERTvsGPT.md          # 技术博客文章
└── data/                     # 本地数据集
```

## 技术博客

[BERT vs GPT：为什么一个能理解、一个能生成](https://zhuanlan.zhihu.com/p/2068217051169154411)

## 运行

```bash
pip install transformers datasets scikit-learn tqdm torch
python text_classification.py
python ner.py
python text_generation.py
```
