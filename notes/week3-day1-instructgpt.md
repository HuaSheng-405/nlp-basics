## GPT 进化链

  - GPT-1: 预训练 + 微调范式(1.17亿参数)
  - GPT-2: 规模放大10倍(15亿参数) → zero-shot 能力涌现
  - GPT-3: 再放大 100 倍 (1750亿参数)→ in-context learning，few-shot 不微调直接做任务
  - InstructGPT: RLHF 对齐人类意图,按照用户的意图执行任务

  ## InstructGPT 核心方法

  三个步骤：
  1. SFT — 人工写答案，有监督微调
  2. RM — 模型生成多个回答，人工排序，得到的数据集用来训练奖励模型
  3. PPO — 用 RM 打分，强化学习优化策略

  ## RLHF 损失函数

$$
objective(φ) =E(x,y)_{∼D_{π_{θ}^{RL}}} [r_{θ}(x,y)-βlog(π_{θ}^{RL}(y|x)/π^{SFT}(y|x))]+γE_{x∼D_{pretrain}} 
[log(π_{φ}^{RL}(x))]
$$

  三部分组成：
  1. 奖励 r(x,y) — RM 给的分数，越高越好
  2. KL 惩罚 — 别偏离 SFT 太远，防止模型刷分
  3. 预训练 γ — 保留原始语言能力，别忘本

  ## 关键发现

  - 1.3B 的 InstructGPT > 175B 的 GPT-3（人类评价），对齐比规模更重要
  - K²/2 对比复用：一次 forward 算完所有对比对