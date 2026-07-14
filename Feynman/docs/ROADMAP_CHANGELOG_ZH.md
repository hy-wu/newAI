# FDIR 项目更新日志与演进待办 (Roadmap & Changelog)

> **Feynman Diagrammatic Intermediate Representation (FDIR)**  
> 集中归档项目历史更新日志与未来演进规划待办。

---

## 1. 演进待办 (Roadmap & TODOs)

### 1.1 EDA 启发式优化待办 (EDA-Inspired Infrastructure Optimization TODOs)

借鉴 EDA (Electronic Design Automation) 芯片编译与布局布线范式，后续拟推进以下优化方向：

1. **TODO-EDA-1: 形式语义验证与等价性校验 (Formal Equivalence Verification)**  
   引入 E-graph (Equality Saturation / e-matching) 或 SMT Solver，验证重写 Pass 前后数学语义的等价性，防止优化 Pass 引入数值漂移。
2. **TODO-EDA-2: 硬件设计规则检查器扩展 (Hardware DRC Extension)**  
   在 `checker.py` 中增加对 Shared Memory Bank Conflict 冲突、内存地址对齐 (Alignments) 和 Tensor Core 维度约束的静态校验。
3. **TODO-EDA-3: 层次化宏网表建模 (Hierarchical Netlist & Macro Submodules)**  
   增加对 Nested Macro Complex Vertices 的支持，将 Transformer Block 抽象为单一 Standard Cell 宏节点，支持局部子图下钻与宏拓扑搜索。

---

### 1.2 理论物理映射与对偶表征探讨待办 (Theoretical Physics Mapping & Representation Duality)

- **持久研究待办 (Theoretical Research Roadmap): 费曼图原生映射 (Primal) 与对偶映射 (Dual) 的范畴论对偶探讨**

#### 1. 原生映射 (Primal Representation) — 当前代码模式
- **顶点 ($V$, Vertex / Node)**：**计算过程 / 相互作用算子**（如 MatMul, Attention, Norm）；
- **传播子 ($E$, Line / Propagator)**：**张量数据流 / 状态场 ($|\psi\rangle$)**（如 $(B, S, D)$ 维度的自由传播）。
- **物理对应直觉**：在量子场论 (QFT) 中，传播子代表粒子在空间中的自由传播（不改变粒子数，仅改变相位/位置），顶点代表粒子碰撞与能量/动量的重新分配（相互作用）。
- **适用场景**：高层算法拓扑推导、算子融合（Operator Fusion）以及张量维度守恒律校验（Einsum 维度缩并即粒子动量守恒）。

#### 2. 对偶映射 (Dual Representation) — 内存主导模式
- **顶点 ($V'$, Vertex / Node)**：**内存 Buffer / 存储状态 / SRAM Tiling 块**（如 HBM 中的张量区、Shared Memory 中的 Tile 缓存）；
- **传播子 ($E'$, Line / Propagator)**：**计算过程 / 内核调度 / 搬运算子**（数据流的转换向量）。
- **物理与工程对应直觉**：
  1. **现代 GPU 的访存主导物理特征**：在 Hopper H100、Blackwell B200 等硬件上，绝大多数 DL 算子的瓶颈在显存带宽 (HBM) 或 SRAM 调度，而非 ALU 算力。在数据流图 (Dataflow Graph) 或 Petri 网中，将 Buffer 视为状态中心 (State Centers)，将计算看作连接不同 Buffer 之间的迁移向量，更能精准反映 SRAM Bank 冲突和 HBM 读写流向。
  2. **范畴论 (Category Theory) 与图对偶 (Graph Duality)**：在图论中，平面图与它的对偶图 (Dual Graph) 共享拓扑信息。将顶点与边互换，即形成了 Dataflow Model。
- **适用场景**：显存分配分析 (Memory Layout Optimization)、SRAM 块搬运调度、硬件布局布线 (Placement & Routing) 以及探究跨 Kernel 的 Data Reuse 开销。

#### 3. 研究目标与统一下发策略
探究在 FDIR 中引入 **Dual Graph Transposition (对偶图转置算子)** 的可行性。高层拓扑优化在 Primal 图上进行，物理 HBM/SRAM 内存调度降级在 Dual 图上进行，实现双图对偶协同。


---

## 2. 更新日志 (Changelog)

### 2026-07-14 多层流行 LLM 架构 (LLaMA-3) 验证支持
- **多层 Transformer 架构构建**：在 `FormulaMapper` 中实现了 `llama_architecture_to_diagram()` 方法，支持构建多层堆叠的 LLaMA-3 Decoder 架构（包含前置 RMSNorm、Multi-Head SDPA Attention、SwiGLU FFN、以及双重残差旁路）。
- **多层端到端验证例程**：新增 `llama3_multi_layer_demo.py` 验证程序，针对 2 层堆叠 LLaMA-3（45 个节点，54 组通道）进行了维度守恒检查、物理 GPU 硬件剖析、CUDA Tile IR / Triton Kernel 下发与 DeepSeek LLM 智能体优化验证。

### 2026-07-14 二维视觉排布与 Markdown 公式格式优化

- **二维分层与自适应标签排布**：扩展了 `visualizer.py` 中的横向与纵向间距（横向间距 4.5，纵向间距 2.8），并按 Y 轴正负位置自适应决定 TikZ 节点的 label 标注方位（`above` / `below`），消除节点标签文本重叠碰撞问题。
- ** Markdown 公式导出**：将公式与下标分析导出的文件修改为 Markdown 格式 (`formula_math.md`)。

### 2026-07-14 架构解耦与 GPU 硬件性能分析集成
- **物理硬件性能测量**：增加 `profiler_runner.py` 模块，利用 PyTorch CUDA Events 和 max_memory_allocated 测量物理显卡上的实际耗时和显存；
- **模块解耦分离**：将智能决策和突变逻辑移入 `Feynman.agent` 独立包，底层编译器部分移入 `Feynman.fdir` 包；
- **DeepSeek API 对接**：集成 `DeepSeekClient` 与 `LLMDesignAgent`，跑通了包含真实 GPU 执行数据反馈的闭环优化流程；
- **分析制品生成器**：新增 `generate_analysis_artifacts.py` 示例程序，一键编译导出 FDIR 分析制品（包括 LaTeX .tex / .pdf 费曼图、SVG 矢量图、HTML 网页、`formula_math.md` Markdown 数学公式描述、DSL 代码、CUDA Tile IR 和 Triton Kernel 汇编与性能指标文件）。
- **词汇优化**：全量清理说明文档中的夸大词汇（如“全量”、“严格”、“完美”、“极致”、“零损”），转为更客观的陈述。
