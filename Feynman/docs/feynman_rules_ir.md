# Feynman Rules for Deep Learning Infrastructure: Specification (FDIR)

## 1. Vision & Overview

The **Feynman Diagrammatic Intermediate Representation (FDIR)** bridges mathematical neural network formulations (such as Tensor Contractions, Multi-Head Attention, and Residual Bypasses) down to executable PyTorch/`torch.fx` graph modules and GPU compiler targets.

Inspired by Quantum Field Theory (QFT), FDIR formalizes neural computation through a strict set of **Feynman Rules**:
- **Tensors as Fields & Propagators**: Internal channels transmit state vectors/tensors with explicit shapes, dynamic/static bounds, Dtypes, and memory layouts.
- **Operators as Vertices**: Primitive computational nodes (linear projections, contraction nodes, attention interactions, pointwise activations, layer normalizations).
- **Contraction as Conservation Laws**: Gauge invariance and index dimension matching enforce exact shape contracts at every vertex leg.
- **Graph Transformation as Gauge Rewrites**: Semantics-preserving graph rewrites (e.g. contraction reassociation, operator fusion, FlashAttention pattern folding).
- **Execution Cost as Action Metric**: FLOPs, HBM memory traffic (Bytes), peak activation footprint, and kernel launch overhead evaluate alternative graph topologies.

---

## 2. The DL Feynman Rules Syntax

### 2.1 External Lines (Interfaces)
- **Input Tensor Line**: $x \sim T^{(B, S, D)} \in \mathbb{R}^{B \times S \times D}$
- **Output Tensor Line**: $y \sim T^{(B, S, D_{\text{out}})}$

### 2.2 Vertices (Interactions)
1. **Contraction Vertex ($V_{\text{contract}}$)**:
   $$\mathcal{V}_{\text{contract}}(A_{ik}, B_{kj}) \Rightarrow C_{ij} = \sum_{k} A_{ik} B_{kj}$$
   *Conservation Law*: Matching index $k$ requires $d_k^{(A)} == d_k^{(B)}$.
2. **Attention Interaction Vertex ($V_{\text{attn}}$)**:
   $$\mathcal{V}_{\text{attn}}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
   Encapsulates the rank-4 tensor interaction channel across batch $B$, head $H$, and sequence length $S$.
3. **Pointwise Activation Vertex ($V_{\text{act}}$)**:
   $$y_i = \sigma(x_i) \quad \text{where } \sigma \in \{\text{ReLU, GELU, Softmax, Sigmoid}\}$$
4. **Residual Bypass Propagator & Addition Vertex ($V_{\text{res}}$)**:
   $$y = x \oplus \mathcal{F}(x)$$
   *Conservation Law*: Shape of bypass propagator $x$ must strictly equal output shape of interaction loop $\mathcal{F}(x)$.

---

## 3. Formal Rewrite Rules & Equivalence Saturation

1. **Linear Projection Fusion**:
   $$\text{MatMul}(X, W) + b \longleftrightarrow \text{LinearVertex}(X, W, b)$$
2. **Double Transpose Cancellation**:
   $$\mathcal{T}(\mathcal{T}(X)) \longleftrightarrow X$$
3. **Contraction Reassociation**:
   $$(A \cdot B) \cdot C \longleftrightarrow A \cdot (B \cdot C)$$
4. **Attention Pattern Folding**:
   $$\text{MatMul}(Q, K^T) \to \text{Softmax} \to \text{MatMul}(\cdot, V) \longleftrightarrow \text{AttentionVertex}(Q, K, V)$$

---

## 4. Operational Cost Model (Action Integral)

For a diagram $\mathcal{D}$, total action cost $\mathcal{S}(\mathcal{D})$ is defined as:

$$\mathcal{S}(\mathcal{D}) = \alpha \cdot \text{FLOPs}(\mathcal{D}) + \beta \cdot \text{HBM\_Traffic\_Bytes}(\mathcal{D}) + \gamma \cdot \text{Launch\_Overhead}(\mathcal{D})$$

Where:
- $\text{FLOPs}$: Multiply-accumulate operations counted over contracted dimensions.
- $\text{HBM\_Traffic\_Bytes}$: Total volume of memory reads & writes across off-chip memory.
- $\text{Launch\_Overhead}$: Fixed launch penalty per discrete vertex execution step.
