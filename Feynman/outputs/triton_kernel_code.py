# ====================================================================
# Auto-Generated Triton JIT GPU Kernel for FDIR Diagram: 'transformer_attention_formula'
# Block Config: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
# ====================================================================
import triton
import triton.language as tl
import torch

@triton.jit
def transformer_attention_formula_triton_kernel(
    ptr_x,
    ptr_W_q,
    ptr_W_k,
    ptr_W_v,
    ptr_out,
    stride_batch, stride_seq, stride_dim,
    N_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 32,
):
    # Program and Tile Indexing
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Vertex: Input (x)
    ptr_in_x = ptr_x + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    val_x = tl.load(ptr_in_x)

    # Vertex: Input (W_q)
    ptr_in_W_q = ptr_W_q + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    val_W_q = tl.load(ptr_in_W_q)

    # Vertex: Input (W_k)
    ptr_in_W_k = ptr_W_k + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    val_W_k = tl.load(ptr_in_W_k)

    # Vertex: Input (W_v)
    ptr_in_W_v = ptr_W_v + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    val_W_v = tl.load(ptr_in_W_v)

    # Vertex: Contraction (proj_Q)
    acc_proj_Q = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_proj_Q += tl.dot(val_x, val_W_q)
    val_proj_Q = acc_proj_Q.to(tl.float16)

    # Vertex: Contraction (proj_K)
    acc_proj_K = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_proj_K += tl.dot(val_x, val_W_k)
    val_proj_K = acc_proj_K.to(tl.float16)

    # Vertex: Contraction (proj_V)
    acc_proj_V = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_proj_V += tl.dot(val_x, val_W_v)
    val_proj_V = acc_proj_V.to(tl.float16)

    # Vertex: Attention (sdpa_attention)
    # Fused FlashAttention Online Softmax Block
    scores_sdpa_attention = tl.dot(val_proj_Q, tl.trans(val_proj_K)) * (1.0 / 8.0)
    m_i = tl.max(scores_sdpa_attention, axis=1)
    p_i = tl.exp(scores_sdpa_attention - m_i[:, None])
    val_sdpa_attention = tl.dot(p_i.to(tl.float16), val_proj_V)

    # Vertex: ResidualAdd (residual_add)
    val_residual_add = val_x + val_sdpa_attention

    # Vertex: Norm (layer_norm)
    mean_layer_norm = tl.sum(val_residual_add, axis=1) / BLOCK_N
    var_layer_norm = tl.sum((val_residual_add - mean_layer_norm[:, None]) ** 2, axis=1) / BLOCK_N
    val_layer_norm = (val_residual_add - mean_layer_norm[:, None]) / tl.sqrt(var_layer_norm[:, None] + 1e-5)

    # Vertex: Output (out)
    ptr_out_out = ptr_out + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    tl.store(ptr_out_out, val_layer_norm)


def launch_transformer_attention_formula_triton(*tensors):
    """Launcher function for transformer_attention_formula_triton_kernel."""
    x_in = tensors[0]
    B, S, D = x_in.shape
    out_tensor = torch.empty_like(x_in)
    grid = (triton.cdiv(S, 128), triton.cdiv(D, 128), B)
    
    transformer_attention_formula_triton_kernel[grid](
        tensors[0],
        tensors[1],
        tensors[2],
        tensors[3],
        out_tensor,
        x_in.stride(0), x_in.stride(1), x_in.stride(2),
        N_SIZE=D,
        num_warps=4,
        num_stages=3,
    )
    return out_tensor