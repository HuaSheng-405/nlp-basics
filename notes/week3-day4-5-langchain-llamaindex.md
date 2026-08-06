# LangChain & LlamaIndex 实战笔记

  ## 一、LangChain Chain 概念

  Chain 用 `|` 把多个步骤串联成管道：

  prompt | model | output_parser
    ↓       ↓         ↓
  填模板   生成文本   解析成 Pydantic 对象

  ### 对比手写 vs Chain

  ```python
  # 手写三步（每天在写）
  prompt = template.format(topic="大语言模型")
  output = generate(prompt)
  result = parser.parse(output)

  # LangChain Chain（接 API 时一行替代）
  chain = template | model | parser
  result = chain.invoke({"topic": "大语言模型"})
  ```

### 本地模型的限制

  本地 Qwen 不能直接用 | 串进 LangChain Chain——| 需要 LangChain 封装的 Runnable 对象。
  当前实验中 Chain 概念用于理解框架思想，后续接 OpenAI API / vLLM 时可以直接用。

  ## 二、LangChain vs LlamaIndex

|           |             LangChain             |         LlamaIndex         |
| :-------: | :-------------------------------: | :------------------------: |
| 核心定位  |   通用 LLM 应用框架（瑞士军刀）   | 数据索引和检索（RAG 专家） |
| 文档加载  |            100+ Loader            |     同样丰富，偏结构化     |
| 索引/检索 |  手动拼 VectorStore + Retriever   |   一行 from_documents()    |
|  向量化   |        手动指定 Embeddings        |   全局 Settings 统一管理   |
|   Agent   |  核心能力，LangGraph 专为此设计   |  有但不如 LangChain 灵活   |
| 学习曲线  | 概念多（Chain/Agent/Tool/Memory） |     接口简洁，上手更快     |

  使用建议

  - 搭 RAG 知识库 → LlamaIndex 更快，开箱即用
  - 搭复杂 Agent 系统 → LangChain/LangGraph 更强，工作流编排灵活
  - 生产环境 → 两个一起用：LlamaIndex 做检索，LangChain 做编排

 ## 三、PydanticOutputParser

###   作用

  自动生成 JSON Schema 格式说明，塞进 prompt，指导模型输出结构化 JSON。

###   生产经验

| 模型大小 |                       策略                        |
| :------: | :-----------------------------------------------: |
|   > 7B   | parser.get_format_instructions() 英文指令正常理解 |
|   < 3B   |            手写中文 Schema 描述更可靠             |

  当前 Qwen 0.5B 测试：英文 Schema 指令意外生效，说明 JSON 格式本身对模型是强信号。

##   四、Embedding 模型 vs 生成模型

|      |    Embedding 模型     |   生成模型   |
| :--: | :-------------------: | :----------: |
| 输入 |         文本          |    prompt    |
| 输出 |    768 维定长向量     | 逐词生成文本 |
| 用途 | 语义相似度计算、检索  |  回答、续写  |
| 代表 | sentence-transformers |  Qwen、GPT   |
| 速度 |        毫秒级         |     秒级     |

  RAG 系统 = Embedding 模型做检索 + 生成模型做回答。