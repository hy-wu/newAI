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

- **持久研究待办 (Theoretical Research Roadmap): 费曼图原生映射 (Primal) 与对偶映射 (Dual) 的统一性探讨**  
  - **Primal 原生表征（当前标准图）**：计算过程为相互作用顶点 (Vertex)，张量/物理状态为传播子 (Propagator Line)。适用于高层拓扑推导与算子融合。
  - **Dual 对偶表征（内存主导视角）**：内存 Buffer / SRAM Tiling 块为节点 (Vertex)，计算过程/数据流传输为有向线 (Propagator Line)。在显存带宽受限 (Memory-bound) 和 SRAM 调度场景下评估 Dataflow Dual Graph 表示法的物理合理性与优化效率。

---

## 2. 更新日志 (Changelog)

### 2026-07-14 二维视觉排布与 Markdown 公式格式优化
- **二维分层与自适应标签排布**：扩展了 `visualizer.py` 中的横向与纵向间距（横向间距 4.5，纵向间距 2.8），并按 Y 轴正负位置自适应决定 TikZ 节点的 label 标注方位（`above` / `below`），消除节点标签文本重叠碰撞问题。
- ** Markdown 公式导出**：将公式与下标分析导出的文件修改为 Markdown 格式 (`formula_math.md`)。

### 2026-07-14 架构解耦与 GPU 硬件性能分析集成
- **物理硬件性能测量**：增加 `profiler_runner.py` 模块，利用 PyTorch CUDA Events 和 max_memory_allocated 测量物理显卡上的实际耗时和显存；
- **模块解耦分离**：将智能决策和突变逻辑移入 `Feynman.agent` 独立包，底层编译器部分移入 `Feynman.fdir` 包；
- **DeepSeek API 对接**：集成 `DeepSeekClient` 与 `LLMDesignAgent`，跑通了包含真实 GPU 执行数据反馈的闭环优化流程；
- **分析制品生成器**：新增 `generate_analysis_artifacts.py` 示例程序，一键编译导出 FDIR 分析制品（包括 LaTeX .tex / .pdf 费曼图、SVG 矢量图、HTML 网页、`formula_math.md` Markdown 数学公式描述、DSL 代码、CUDA Tile IR 和 Triton Kernel 汇编与性能指标文件）。
- **词汇优化**：全量清理说明文档中的夸大词汇（如“全量”、“严格”、“完美”、“极致”、“零损”），转为更客观的陈述。
