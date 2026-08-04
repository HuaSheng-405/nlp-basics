## BERT 为什么是双向的

答：bert采用的是Transformer的encoder架构，encoder层中没有causal mask，模型可以轻松的看到句子中的所有词。如，encoder_layer.py中的self.my_attention(x,x,x,mask),这里的mask只能是None或者src_mask.

## GPT 为什么是单向的

答：GPT采用的是Transformer的decoder-only架构，decoder层包含masked-attention和ffn，masked-attention中采用了causal mask，遮住了后面的词，模型只能看到当前词及之前的词。如,decoder_layer.py中的

```python
causal_mask = torch.triu(torch.ones(seq_length,seq_length),diagonal = 1).bool()
```

这行代码决定了模型不能看到后面的词.

## BERT 怎么用于下游任务

答：对于每个任务，采用特定的输入和输出，把它们传进bert并分别微调.在文本分类任务中,我将[CLS]的最终隐藏层输出映射为分类标签.在命名实体识别(NER)任务中,我将每个token的最终隐藏层输出映射为实体标签.在 THUCNews 新闻分类上达到 98.36% 准确率，在 CLUENER NER 上达到 94.65% F1。

## 自回归生成的重复问题

答：这是因为采用了确定性解码，模型固定输出特定的内容导致的.对于同一句话"今天天气很好,"采用贪心解码得到的结果可能是,"今天天气很好,,,,,,,,,,,",一旦陷入死胡同就出不来了.如果是top_k策略,"今天天气很好,风吹吹的吹的,蛮嗲的"虽然效果也不好,但是具有多样性,不会直接在一个地方卡死.