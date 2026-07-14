# FDIR 数学公式与 Tensor Einstein Subscripts 分析报告

## 1. 显式 LaTeX 数学公式

$$
y = \text{LayerNorm}\left(\mathbf{x} + \text{Softmax}\left(\frac{\mathbf{x} \cdot \mathbf{W_q} \mathbf{x} \cdot \mathbf{W_k}^T}{\sqrt{d_k}}\right) \mathbf{x} \cdot \mathbf{W_v}\right)
$$

## 2. Einstein 矩阵缩并下标链 (Einstein Notation Chain)

```text
4256768,768768->4256768; 4256768,768768->4256768; 4256768,768768->4256768
```
