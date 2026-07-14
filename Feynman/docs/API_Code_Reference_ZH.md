# FDIR 项目全量模块 API 与代码设计中文指南

> **Feynman Diagrammatic Intermediate Representation (FDIR)**  
> **面向开发者与架构师的代码级工程手册**  
> **项目包结构**：包含底层编译器引擎包 `Feynman.fdir` 与上层智能 Agent 子系统包 `Feynman.agent`

---

## 目录

1. [项目架构设计与分层说明](#1-项目架构设计与分层说明)
2. [底层编译器引擎 `Feynman.fdir` 详细 API](#2-底层编译器引擎-feynmanfdir-详细-api)
   - 2.1 强类型描述符 (`types.py`)
   - 2.2 相互作用顶点与传播子 (`nodes.py`)
   - 2.3 图容器与拓扑操作 (`diagram.py`)
   - 2.4 维度守恒律检查器 (`checker.py`)
   - 2.5 活跃性显存与 Action 物理成本模型 (`cost.py`)
   - 2.6 拓扑突变重写引擎 (`rewriter.py`)
   - 2.7 数学公式双向转换器 (`formula.py`)
   - 2.8 物理费曼图渲染器 (`visualizer.py`)
   - 2.9 Python DSL 零损序列化器 (`codegen.py`)
   - 2.10 硬件下发编译器 (`lowering.py`, `lowering_tileir.py`, `lowering_triton.py`)
   - 2.11 双重性能评估引擎 (`evaluation.py`)
3. [上层智能 Agent 子系统 `Feynman.agent` 详细 API](#3-上层智能-agent-子系统-feynmanagent-详细-api)
   - 3.1 闭环设计环境接口 (`design_agent.py`)
   - 3.2 DeepSeek LLM API 客户端 (`llm_client.py`)
   - 3.3 DeepSeek 驱动的自主架构设计 Agent (`fdir_agent.py`)
4. [常见问题与抗鲁棒性设计](#4-常见问题与抗鲁棒性设计)

---

## 1. 项目架构设计与分层说明

为了保持系统的模块化与高内聚低耦合，项目被严格划分为**两层核心子系统**：

```text
Feynman/
├── fdir/                           # 核心编译器引擎层 (Pure Infrastructure Layer)
│   ├── types.py                    # 维度 Shape、DType、Layout 类型定义与守恒律验证
│   ├── nodes.py                    # Propagator 有向线与 8 种 Vertex 相互作用顶点定义
│   ├── diagram.py                  # Diagram 图容器，提供拓扑排序、节点删除与边重连
│   ├── checker.py                  # 静态类型检查器 ShapeTypeChecker
│   ├── cost.py                     # 基于活跃性分析 (Liveness Analysis) 的物理成本评估
│   ├── rewriter.py                 # 图重写引擎 (Attention 融合、双重转置消除)
│   ├── formula.py                  # Einstein 表达式 & LaTeX 公式与 AST 双向转换
│   ├── visualizer.py               # LaTeX tikz-feynman、SVG 与 HTML 矢量画图渲染器
│   ├── codegen.py                  # FDIR AST 与 Python DSL 源码双向转换器
│   ├── lowering.py                 # PyTorch nn.Module 编译下发器
│   ├── lowering_tileir.py          # NVIDIA CUDA Tile IR (cuda::tile) 编译下发器
│   ├── lowering_triton.py          # OpenAI Triton (@triton.jit) 编译下发器
│   └── evaluation.py               # Model Capacity + GPU Roofline 双重性能评估器
│
├── agent/                          # 上层自主 Agent 子系统 (Autonomous Intelligence Layer)
│   ├── design_agent.py             # DesignAgentInterface 闭环环境、观察状态与突变算子
│   ├── llm_client.py               # DeepSeek LLM API 客户端 (自动加载 .env 密钥)
│   └── fdir_agent.py               # LLMDesignAgent，接入 DeepSeek 形成自主架构进化闭环
│
├── examples/                       # 端到端例程
│   ├── transformer_block.py        # 基础 Transformer 模块建构与验证
│   ├── closed_loop_ecosystem_demo.py # 多模态闭环矩阵全链路 Demo
│   └── llm_agent_deepseek_demo.py  # 接入 DeepSeek API 的真实智能体架构优化 Demo
│
└── tests/                          # pytest 测试套件 (25/25 单元测试)
    ├── test_feynman_ir.py
    └── test_closed_loop_matrix.py
```

---

## 2. 底层编译器引擎 `Feynman.fdir` 详细 API

### 2.1 强类型描述符 (`types.py`)

#### `Shape`
- `dims: Tuple[Union[int, str], ...]`：维度元组，支持具体整数（如 `768`）与符号变量（如 `"B"`, `"S"`, `"D"`）。
- `resolve_dim(d, env)`：将符号维度结合环境变量映射表 `env` 解析为具体整数。
- `num_elements(env)`：计算张量元素总量。未绑定的符号变量保守按 `1` 计算。
- `is_compatible(other, env)`：静态维度守恒验证。匹配规则：
  1. 阶数 (Rank) 必须严格相等；
  2. 均解析为整数时，数值必须相等；
  3. 均未解析的符号变量，**名称必须完全相同**（如 `"D"` 与 `"D"` 兼容，与 `"H"` 不兼容）；
  4. 一个可解析另一个不可解析时，保守返回 `False`。

#### `TensorType`
- `shape: Shape`, `dtype: DType`, `layout: Layout`, `indices: Tuple[str, ...]`
- `size_in_bytes(env)`：计算张量在 GPU 显存中占用的字节数。

---

### 2.2 相互作用顶点与传播子 (`nodes.py`)

#### `Propagator`
有向线描述符，连接 `src_vertex_id` 与 `dst_vertex_id`，携带类型 `tensor_type` 及业务标签 `label`（如 `"bypass"` 表示残差旁路）。

#### `Vertex` 基类与派生子类
- `InputVertex(id, tensor_type)`：外部传入粒子状态。
- `OutputVertex(id)`：外部输出状态。
- `ContractionVertex(id, transpose_b=False)`：矩阵乘法/缩并。`get_output_type()` 严格校验缩并轴末尾维度 $K_1 = K_2$。
- `AttentionVertex(id, num_heads, head_dim)`：Rank-4 注意力算子 $\text{Softmax}(QK^T/\sqrt{d_k})V$。
- `PointwiseVertex(id, sub_op)`：逐元激活/二元运算，支持 `ReLU`, `GELU`, `Softmax`, `Add`, `Mul`, `Sub`, `Sigmoid`, `Tanh`。
- `ResidualVertex(id)`：旁路直连相加 $y = x + \mathcal{F}(x)$。
- `NormVertex(id, norm_type="LayerNorm"|"RMSNorm", eps=1e-5)`：归一化算子。
- `TransposeVertex(id, perm)`：维度重排算子。

---

### 2.3 图容器与拓扑操作 (`diagram.py`)

#### `Diagram`
- `add_vertex(vertex)` / `remove_vertex(vertex_id)`：添加或安全销毁顶点（自动解绑关联线）。
- `connect(src_id, dst_id, tensor_type, label)` / `remove_propagator(prop_id)`：挂载或擦除传播子。
- `topological_sort()`：采用 Kahn 算法结合 `collections.deque`（$O(1)$ popleft）计算 DAG 拓扑执行序，若存在环路直接抛出 `ValueError`。
- `to_mermaid()`：导出 Mermaid 流程图文本。

---

### 2.4 维度守恒律检查器 (`checker.py`)

#### `ShapeTypeChecker(env)`
- `check(diagram) -> bool`：沿拓扑序前向推算每个算子的真实输出类型，并执行特化守恒律校验：
  - `ContractionVertex`：校验缩并轴 $K_1 = K_2$；
  - `ResidualVertex`：校验旁路 $x$ 与分支 $\mathcal{F}(x)$ 的 Shape 兼容量；
  - 校验最终导出的算子类型与传播子显式声明的 `tensor_type` 兼容量。

---

### 2.5 活跃性显存与 Action 物理成本模型 (`cost.py`)

#### `CostModel(env)`
- `evaluate(diagram) -> PerformanceReport`：
  - **FLOPs 浮点运算量**：基于 MatMul、SDPA Attention 物理公式精确累加；
  - **HBM 流量**：累加所有算子的读写 bytes 传输；
  - **活跃显存峰值（Liveness Analysis Peak Memory）**：追踪每个节点张量的生成步（Production Step）与最终消亡步（Last Consumption Step），在拓扑执行的每个时刻步 $t$ 求存活张量的显存和，取最大值。

---

### 2.6 拓扑突变重写引擎 (`rewriter.py`)

#### `RewriteEngine`
- `fuse_attention_pattern(diagram)`：检测 `QK^T` 缩并 $\to$ Softmax $\to$ `PV` 缩并三节点序列，安全删除这 3 个旧节点与内部边，挂载新 `AttentionVertex` 并重连 Q/K/V 和下游节点。
- `cancel_double_transpose(diagram)`：检测连续两次置换 $T_{p2}(T_{p1}(x))$，若组合置换 $p_1[p_2[i]] == i$，擦除两个置换节点，直接贯通前驱与后继。

---

### 2.7 数学公式双向转换器 (`formula.py`)

#### `FormulaMapper`
- `einsum_to_diagram(einsum_str)`：解析 Einstein 表示法（如 `"ik,kj->ij; ij,jl->il"`）生成 Diagram。
- `attention_formula_to_diagram(B, S, D, num_heads, head_dim)`：一键生成完整 Transformer Attention 模块 AST。
- `diagram_to_latex(diagram)`：将 AST 反编译导出为 LaTeX 数学公式（如 $y = \text{LayerNorm}(\dots)$）。
- `diagram_to_einsum_chain(diagram)`：导出算子链的 Einsum 表示法。

---

### 2.8 物理费曼图渲染器 (`visualizer.py`)

#### `FeynmanVisualizer`
- `to_tikz(diagram)`：导出符合物理学术论文标准的 LaTeX `tikz-feynman` 源代码（实线为费米子/状态线，波浪线为玻色/Attention 相互作用线，虚线为残差旁路）。
- `to_svg(diagram, width, height)`：生成颜色编码的独立 SVG 矢量图。
- `to_html(diagram)`：生成包含 SVG 矢量图与 TikZ 源码的交互式 HTML 文件。

---

### 2.9 Python DSL 零损序列化器 (`codegen.py`)

#### `FDIRCodeGen`
- `ast_to_code(diagram) -> str`：将 Diagram AST 序列化为干净的可执行 Python DSL 代码。
- `code_to_ast(code_str) -> Diagram`：在安全命名空间中执行 Python DSL 代码，逆向构建 Diagram AST。

---

### 2.10 硬件下发编译器 (`lowering*.py`)

1. **`TorchLowering` (`lowering.py`)**：编译生成动态求值的 PyTorch `nn.Module`，支持 LayerNorm 与 RMSNorm 前向推理。
2. **`TileIRLowering` (`lowering_tileir.py`)**：遵照 NVIDIA [CUDA Tile IR 规范](https://docs.nvidia.com/cuda/tile-ir/)，下发为 `cuda::tile::load`、`cuda::tile::mma`、`cuda::tile::store` 模板 C++ 代码。
3. **`TritonLowering` (`lowering_triton.py`)**：下发为 `@triton.jit` Python 代码与完整的 Launcher 启动引导代码。

---

### 2.11 双重性能评估引擎 (`evaluation.py`)

#### `DualEvaluator(env, hardware)`
输出包含 Model Capacity 与 GPU Roofline Model 的联合报告 `DualPerformanceReport`：
- **`ModelPerformanceReport`**：参数量、激活量、最长拓扑深度、感应野范围（Receptive Field）、参数效率、序列复杂度 $O(S^2 \cdot D)$。
- **`InfraPerformanceReport`**：基于 GPU Hardware Spec（默认 H100）计算算术强度 FLOPs/Byte、计算时间 $T_{\text{compute}}$、显存时间 $T_{\text{memory}}$、计算瓶颈诊断与算力/带宽利用率。

---

## 3. 上层智能 Agent 子系统 `Feynman.agent` 详细 API

### 3.1 闭环设计环境接口 (`design_agent.py`)

#### `DesignAgentInterface`
- `observe() -> Observation`：获取当前计算图拓扑摘要、双重性能报告及历史动作日志。
- `mutate(action: MutationAction) -> Observation`：执行原子突变动作并更新环境。
- `get_available_mutations() -> List[MutationAction]`：根据当前拓扑推荐可选突变动作。
- 支持突变动作类型 `MutationType`：`ADD_CONTRACTION`, `ADD_ATTENTION`, `ADD_RESIDUAL_BYPASS`, `ADD_NORM_LAYER`, `REMOVE_VERTEX`, `FUSE_ATTENTION`, `MODIFY_TILE_CONFIG`, `SWAP_NORM_TYPE`。

---

### 3.2 DeepSeek LLM API 客户端 (`llm_client.py`)

#### `DeepSeekClient(api_key, base_url, model="deepseek-chat")`
- `load_env_file()`：自动读取项目根目录下的 `.env` 文件（包含 `DEEPSEEK_API_KEY` 与 `DEEPSEEK_BASE_URL`）。
- `chat_completion(messages, temperature, max_tokens)`：使用 Python 标准库 `urllib.request` 封装的高效 OpenAI 兼容 HTTP 客户端。

---

### 3.3 DeepSeek 驱动的自主架构设计 Agent (`fdir_agent.py`)

#### `LLMDesignAgent(agent_env, llm_client)`
- `run_optimization_step(verbose=True)`：自动构建 Prompt，将计算图的算力/显存瓶颈输入给 DeepSeek LLM，解析 DeepSeek 返回的 JSON 决策对象（包含推理理由 `reasoning` 与结构突变 `action`），在 `DesignAgentInterface` 上执行突变并返回更新后的系统状态。

---

## 4. 常见问题与抗鲁棒性设计

1. **为什么将 `agent/` 与 `fdir/` 分离？**
   - `fdir/` 是纯粹的 AI 编译器与 IR 基础设施，不依赖任何 API 或大模型网络连接；
   - `agent/` 是上层的智能决策层。解耦后，`fdir/` 可以被任意搜索算法（如 MCTS、强化学习、贝叶斯优化）独立调用，也可以接入不同的 LLM 供应商。
2. **符号维度在静态检查时报错的原因？**
   - 如果两个不同的符号字符串（如 `"D"` 与 `"H"`）没有在环境映射表 `env` 中显式赋值相同整数，静态检查器会自动判定它们不兼容，以避免隐式维度不匹配导致的运行时底层 Crash。
