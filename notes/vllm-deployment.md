# vLLM 部署笔记：推理优化原理 + WSL2 踩坑实录

## 一、vLLM 为什么快（推理优化原理）

### 传统推理的瓶颈

1. **KV Cache 碎片化**：每个请求的 KV cache 是独立分配的连续显存块，请求长度不一 → 显存碎片 → 浪费 60-80%
2. **请求排队**：一个请求算完才处理下一个，GPU 有空闲时也在等

### vLLM 的两个核心优化

**PagedAttention（分页注意力）**

```
传统: [=========== 请求A的KV cache ===========]  ← 连续块，长度不齐就浪费
vLLM: 请求A: [块1][块4][块7]                      ← 按页分配，像操作系统的虚拟内存
      请求B: [块2][块5]
      请求C: [块3][块6][块8]

KV cache 按固定大小的页（block）管理，不连续也没关系
→ 显存利用率从 40% 提到 90%+，同样的显存能塞 2-3 倍请求
```

**Continuous Batching（连续批处理）**

```
传统: 请求1: [=======完成=======] → 空闲 → 请求2: [======完成======]
vLLM: 请求1: [====完成]               ← 请求1 生成了 10 个 token 就结束
      请求2:      [=====完成]
      请求3:           [========完成]
      → 每个 token 生成完立即腾出位置给新请求，GPU 永远在满负荷工作
```

### 实测结果（Qwen2.5-0.5B，100 并发请求）

```
并发请求: 100
总耗时:   5.7s
吞吐量:   2039 tokens/s
```

对照 HF 原生单请求串行 ~50-100 tokens/s → **vLLM 吞吐提升 20-40 倍**。
不是单请求更快，而是 100 个请求一起处理时总吞吐不塌。

## 二、部署环境：Windows/WSL2 踩坑实录

### 问题链

| 尝试 | 结果 |
|------|------|
| Windows 原生 pip install vllm | ❌ `vllm._C_stable_libtorch` 缺失——vLLM 官方不支持 Windows |
| WSL2 + 最新版 vLLM | ❌ UVA not available——WSL2 GPU 虚拟化不支持 |
| 降级 vLLM 0.8.5.post1 | ✅ UVA 问题解决 |
| 新版 transformers | ❌ `all_special_tokens_extended` 属性被移除 |
| 版本矩阵不匹配 | ❌ torch 扩展符号错误 |

### 最终可行的版本矩阵

```
环境: WSL2 + Ubuntu 22.04LTS + venv
vllm==0.8.5.post1
transformers==4.57.0（或更低，需保留 all_special_tokens_extended）
torch==2.6.0
```

### 关键教训

1. **vLLM 只支持 Linux**——Windows 用户必须走 WSL2 或 Docker
2. **WSL2 的 GPU 兼容性是渐进式的**——新版 vLLM 依赖 UVA，老版本反而兼容
3. **vLLM 依赖链极其严格**（vllm ↔ transformers ↔ torch ↔ CUDA 扩展）——版本必须按官方兼容矩阵配对，手动装环境时逐对降级直到能跑
4. **模型太大放不下显存时，用更小的模型验证流程**——最终 0.5B 跑通，原理验证不受模型大小影响

## 三、生产部署决策总结

| 引擎 | 适用场景 | 吞吐 |
|------|---------|------|
| HuggingFace 原生 | 开发调试、单请求推理 | 低（~100 tokens/s） |
| Ollama (llama.cpp) | 本地单机部署，简单 | 中 |
| vLLM | 生产服务、高并发 | 高（2000+ tokens/s） |

**生产环境为什么不用原生 transformers 推理：**
1. KV cache 碎片导致显存利用率低 → 单卡能服务的请求数少
2. 无批处理 → GPU 空闲浪费
3. vLLM/Ollama 等专用引擎解决这两点，吞吐提升一个数量级
