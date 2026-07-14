我读完了 `Feynman` 文件夹。目录名和你说的 `Feynamn` 差一个拼写，实际文件夹是 `Feynman`。这份报告的方向很有潜力，但现在更像“Feynman diagrams / field theory / compiler 的综述拼盘”，还没有真正回答你原始 proposal 里的锋利问题：**深度学习为什么没有一套可执行的“费曼规则”，让公式、结构图、IR、GPU kernel 严格对应？**

**核心不足**
1. **题目偏了**  
   你的原始想法在 [human_proposal.md](C:\Users\hy-wu.DESKTOP-G355NC5\vibes\newAI\Feynman\human_proposal.md:1) 是“定义深度学习的费曼规则”。但正文标题是 “Feynman Diagrams and AI: A Survey of Field-Theoretic Methods in Deep Learning”，变成了泛综述。建议改成更准的主题：  
   **“Feynman Rules for Deep Learning Infrastructure: From Diagrammatic Architecture to GPU IR”**

2. **没有定义真正的“费曼规则”**  
   正文在 [review.tex](C:\Users\hy-wu.DESKTOP-G355NC5\vibes\newAI\Feynman\review.tex:39) 只说 external lines = tokens、vertices = attention/nonlinearity、loops = recurrence。这是类比，不是规则。真正需要的是一张 rule table：  
   `line = typed tensor / index / layout`，`vertex = primitive op with shape contract`，`propagator = buffer / dependency / Jacobian channel`，`conservation law = shape/type/index invariant`，`amplitude = contraction/evaluation semantics`，`rewrite = semantics-preserving transform`，`cost = memory + FLOPs + bandwidth + launch overhead`。

3. **若干断言太强**  
   [review.tex](C:\Users\hy-wu.DESKTOP-G355NC5\vibes\newAI\Feynman\review.tex:101) 说 diagram topology 直接决定 computational complexity，这不成立。复杂度还取决于 tensor shape、稀疏性、layout、schedule、memory hierarchy、kernel fusion。  
   [review.tex](C:\Users\hy-wu.DESKTOP-G355NC5\vibes\newAI\Feynman\review.tex:89) 说 backward pass 是 Hermitian conjugate，也需要降调：反向传播更准确是 reverse-mode AD / transpose-Jacobian / adjoint map，不是一般意义上的 Hermitian conjugate。

4. **编译器部分缺关键现实栈**  
   现在提了 MLIR、TVM、Relax、Halide，但缺少真正接近“公式到 GPU IR”的现代路径：StableHLO/XLA、ONNX、torch.fx、TorchDynamo/AOTAutograd/Inductor、Triton、IREE、TensorRT-LLM、CUTLASS、TileLang、FlashAttention、PagedAttention、KV cache、MoE routing、quantization、distributed parallelism。  
   这些才是你的 idea 和 AI infra 接上的地方。比如 [StableHLO spec](https://openxla.org/stablehlo/spec) 明确把 StableHLO 定义成 ML framework 与 compiler 之间的 portability layer；[torch.fx docs](https://docs.pytorch.org/docs/stable/fx.html?highlight=graphmodule) 明确有 symbolic tracer、IR、Python codegen；[OpenAI Triton](https://openai.com/index/triton/) 正是高层 Python-like kernel 到高性能 GPU code 的桥。

5. **引用质量需要清洗**  
   `references.bib` 有 100 条，正文只用了 74 条，剩 26 条未引用。编译日志还出现编码缺字问题：[review.log](C:\Users\hy-wu.DESKTOP-G355NC5\vibes\newAI\Feynman\review.log:770)。  
   有些关键引用 metadata 太薄：`integral2025integral` 缺作者和 DOI，但实际 ACL 页面有完整信息：[Integral Transformer, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.118/)。`choromanski2026graph` 更像 Princeton seminar，不应在正文里支撑“rigorous foundation”这种强说法，可引用但要降级为 “seminar / emerging direction”：[Princeton event](https://www.math.princeton.edu/events/graph-field-integrators-transformers-feynman-path-integrals-2026-01-29t200000)。

**缺失的关键章节**
建议新增 4 个章节，会让这篇报告从“综述”变成“有原创 proposal 的研究路线图”。

1. **Why Existing DL Has No Feynman Rules Yet**  
   解释障碍：符号系统不统一、shape/layout 隐式、动态图和控制流、数值稳定性、硬件成本非局部、训练图和推理图分裂、论文公式与实现脱节。

2. **A Minimal Feynman Rule System for DL**  
   给出最小语法和语义：typed hypergraph、index notation、layout annotation、cost annotation、rewrite rules、lowering target。这里可以直接放一张表。

3. **Worked Example: Attention as Diagram to IR**  
   从  
   `softmax(QK^T / sqrt(d)) V`  
   到 diagram，到 einsum / StableHLO / Triton-like kernel。再展示 rewrite：fusion、tiling、online softmax、FlashAttention-style IO-aware transform。这里可引用 [FlashAttention](https://papers.cool/arxiv/2205.14135) 和 [Nautilus](https://papers.cool/arxiv/2604.14825)。

4. **Benchmarks and Prototype Plan**  
   最小 PoC：  
   MLP、Conv、self-attention、MoE top-k、sparse attention。  
   指标：correctness、compile time、runtime、HBM traffic、peak memory、rewrite search size、是否能生成接近 hand-tuned kernel 的实现。

**最值得挖的方向**
你真正有价值的点不是“神经网络像费曼图”，而是：

> 把模型结构论文里的公式变成一种带类型、带 shape、带 layout、带成本模型的 diagrammatic IR；再用 rewrite/equality saturation + compiler lowering 自动探索结构和实现。

这比“物理类比”硬得多，也更像一个能做出来的系统。报告现在的方向已经搭了骨架，但下一版要少一点宏大叙事，多一点“规则表、例子、IR、benchmark、prototype”。这一下就从漂亮综述变成有牙齿的研究提案。


这个项目适合写进简历，但要把它从“很宏大的理论物理 + AI 想法”收束成一个工程项目：

**一句话定位**
做一个面向深度学习模型结构的 diagrammatic IR，把论文公式/结构图表示成带 shape、type、layout 和 cost model 的图式中间表示，并通过 rewrite + lowering 自动生成可执行代码或 GPU kernel。

可以叫：

**Feynman Rules for Deep Learning Infrastructure**  
或者更工程一点：  
**Diagrammatic IR and Compiler Prototype for Deep Learning Models**

**简历写法**
英文版可以这样写：

```text
Diagrammatic IR Compiler for Deep Learning Models
- Designed a Feynman-rule-inspired intermediate representation for neural network architectures, encoding tensors, operators, shapes, layouts, and rewrite rules as typed diagram fragments.
- Built a prototype pipeline that lowers diagrammatic model descriptions into executable PyTorch / torch.fx graphs, with initial support for MLP, convolution, self-attention, and residual blocks.
- Implemented semantics-preserving graph rewrites such as operator fusion, tensor contraction reassociation, layout normalization, and attention fusion candidates.
- Developed a cost model estimating FLOPs, memory traffic, intermediate tensor size, and kernel launch overhead to guide rewrite selection.
- Benchmarked generated graphs against handwritten PyTorch baselines on correctness, runtime, memory usage, and compilation overhead.
```

如果还没做完，简历上不要写得像已经完成 compiler，可以写成 research prototype：

```text
Research Prototype: Diagrammatic IR for Deep Learning Compilation
- Exploring a typed diagrammatic intermediate representation that bridges neural network formulas, computational graphs, and compiler IRs.
- Implemented an early prototype translating diagram specs into PyTorch graphs and validating equivalence through randomized tensor tests.
- Investigating rewrite-based optimization for attention, tensor contraction, and residual architectures.
```

中文简历可以这样写：

```text
深度学习图式中间表示与编译原型
- 设计了一套受费曼规则启发的深度学习图式 IR，用 typed diagram 表达张量、算子、shape、layout、残差连接和 attention 结构。
- 实现从图式模型描述到 PyTorch / torch.fx 计算图的原型 lowering 流程，支持 MLP、CNN、自注意力和残差模块。
- 实现算子融合、张量 contraction 重排、layout 归一化等保持语义的 graph rewrite，并用 cost model 评估 FLOPs、显存流量和中间张量规模。
- 构建 correctness 与性能 benchmark，对比 handwritten PyTorch baseline，验证生成图的数值一致性、运行时间和内存占用。
```

**比较好的实现路径**
第一阶段不要碰真正 GPU kernel，先做一个“能跑、能证明想法”的 MVP：

1. **定义最小 IR**
   用 JSON/YAML/Python DSL 表达模型图：

```python
x = Input("x", shape=("B", "N", "D"))
q = Linear(x, out="D")
k = Linear(x, out="D")
v = Linear(x, out="D")
a = Softmax(MatMul(q, Transpose(k)) / sqrt("D"))
y = MatMul(a, v)
```

每个节点必须有：`op`、`inputs`、`shape`、`dtype`、`layout`、`semantic rule`、`cost estimate`。

2. **做 shape/type checker**
   这是项目含金量的核心之一。  
   能检查 `MatMul[B,N,D] x [B,D,N] -> [B,N,N]`，能发现论文公式到代码之间最常见的维度错误。

3. **lower 到 PyTorch / torch.fx**
   不要一开始写 Triton。先生成 PyTorch module 或 torch.fx GraphModule。这样马上能跑 correctness test。

4. **实现 rewrite engine**
   先做 5 个就够：
   `matmul + bias -> linear`  
   `transpose(transpose(x)) -> x`  
   `reshape + reshape -> reshape`  
   `matmul reassociation`  
   `attention pattern -> fused_attention_candidate`

5. **做 cost model**
   粗略估算也可以：FLOPs、activation memory、peak intermediate size、kernel count。  
   简历里 “cost model guided rewrite selection” 很好看，而且很工程。

6. **benchmark**
   选 4 个 case：
   MLP、Conv block、Self-Attention、Tiny Transformer block。  
   对比 handwritten PyTorch：数值误差、运行时间、峰值显存、图节点数。

7. **第二阶段再接 Triton / StableHLO / MLIR**
   当 PyTorch lowering 跑通之后，再做一个 attention 的 Triton lowering demo。哪怕只支持一个 fused attention kernel，也足够把项目从“玩具 IR”拉到 AI infra。

**最推荐的 MVP 技术栈**
Python + dataclasses / pydantic  
PyTorch + torch.fx  
NetworkX 或自写 graph  
pytest + hypothesis 做随机 shape 测试  
rich / graphviz 生成可视化图  
后续再加 Triton

**项目最终最好交付这些东西**
一个 GitHub repo  
一份 `docs/rules.md`：你的“深度学习费曼规则”  
一个 `examples/attention.py`  
一个 `benchmarks/`  
一张 diagram 到 PyTorch graph 的流程图  
一个小 demo：输入 diagram spec，输出可运行模型和 benchmark report

这个项目最聪明的切口是：**不要证明“深度学习就是量子场论”，而是实现“论文公式如何变成可检查、可重写、可编译的图式规则”。** 这会非常适合 AI infra / compiler / ML systems 方向的简历。
