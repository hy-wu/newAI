# ====================================================================
# Auto-Generated Triton JIT GPU Kernel for FDIR Diagram: 'llama3_2layer_architecture'
# Block Config: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
# ====================================================================
import triton
import triton.language as tl
import torch

@triton.jit
def llama3_2layer_architecture_triton_kernel(
    ptr_x_input,
    ptr_layer_0_W_q,
    ptr_layer_0_W_k,
    ptr_layer_0_W_v,
    ptr_layer_0_W_o,
    ptr_layer_0_W_gate,
    ptr_layer_0_W_up,
    ptr_layer_0_W_down,
    ptr_layer_1_W_q,
    ptr_layer_1_W_k,
    ptr_layer_1_W_v,
    ptr_layer_1_W_o,
    ptr_layer_1_W_gate,
    ptr_layer_1_W_up,
    ptr_layer_1_W_down,
    ptr_logits_out,
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

    # Vertex: Input (x_input)
    ptr_in_x_input = ptr_x_input + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    val_x_input = tl.load(ptr_in_x_input)

    # Vertex: Input (layer_0_W_q)
    ptr_in_layer_0_W_q = ptr_layer_0_W_q + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    val_layer_0_W_q = tl.load(ptr_in_layer_0_W_q)

    # Vertex: Input (layer_0_W_k)
    ptr_in_layer_0_W_k = ptr_layer_0_W_k + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    val_layer_0_W_k = tl.load(ptr_in_layer_0_W_k)

    # Vertex: Input (layer_0_W_v)
    ptr_in_layer_0_W_v = ptr_layer_0_W_v + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    val_layer_0_W_v = tl.load(ptr_in_layer_0_W_v)

    # Vertex: Input (layer_0_W_o)
    ptr_in_layer_0_W_o = ptr_layer_0_W_o + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    val_layer_0_W_o = tl.load(ptr_in_layer_0_W_o)

    # Vertex: Input (layer_0_W_gate)
    ptr_in_layer_0_W_gate = ptr_layer_0_W_gate + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    val_layer_0_W_gate = tl.load(ptr_in_layer_0_W_gate)

    # Vertex: Input (layer_0_W_up)
    ptr_in_layer_0_W_up = ptr_layer_0_W_up + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    val_layer_0_W_up = tl.load(ptr_in_layer_0_W_up)

    # Vertex: Input (layer_0_W_down)
    ptr_in_layer_0_W_down = ptr_layer_0_W_down + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    val_layer_0_W_down = tl.load(ptr_in_layer_0_W_down)

    # Vertex: Input (layer_1_W_q)
    ptr_in_layer_1_W_q = ptr_layer_1_W_q + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    val_layer_1_W_q = tl.load(ptr_in_layer_1_W_q)

    # Vertex: Input (layer_1_W_k)
    ptr_in_layer_1_W_k = ptr_layer_1_W_k + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    val_layer_1_W_k = tl.load(ptr_in_layer_1_W_k)

    # Vertex: Input (layer_1_W_v)
    ptr_in_layer_1_W_v = ptr_layer_1_W_v + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    val_layer_1_W_v = tl.load(ptr_in_layer_1_W_v)

    # Vertex: Input (layer_1_W_o)
    ptr_in_layer_1_W_o = ptr_layer_1_W_o + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    val_layer_1_W_o = tl.load(ptr_in_layer_1_W_o)

    # Vertex: Input (layer_1_W_gate)
    ptr_in_layer_1_W_gate = ptr_layer_1_W_gate + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    val_layer_1_W_gate = tl.load(ptr_in_layer_1_W_gate)

    # Vertex: Input (layer_1_W_up)
    ptr_in_layer_1_W_up = ptr_layer_1_W_up + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    val_layer_1_W_up = tl.load(ptr_in_layer_1_W_up)

    # Vertex: Input (layer_1_W_down)
    ptr_in_layer_1_W_down = ptr_layer_1_W_down + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    val_layer_1_W_down = tl.load(ptr_in_layer_1_W_down)

    # Vertex: Norm (layer_0_rms_norm1)
    mean_sq_layer_0_rms_norm1 = tl.sum(val_x_input * val_x_input, axis=1) / BLOCK_N
    inv_rms_layer_0_rms_norm1 = 1.0 / tl.sqrt(mean_sq_layer_0_rms_norm1 + 1e-5)
    val_layer_0_rms_norm1 = val_x_input * inv_rms_layer_0_rms_norm1[:, None]

    # Vertex: Contraction (layer_0_proj_Q)
    acc_layer_0_proj_Q = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_layer_0_proj_Q += tl.dot(val_layer_0_rms_norm1, val_layer_0_W_q)
    val_layer_0_proj_Q = acc_layer_0_proj_Q.to(tl.float16)

    # Vertex: Contraction (layer_0_proj_K)
    acc_layer_0_proj_K = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_layer_0_proj_K += tl.dot(val_layer_0_rms_norm1, val_layer_0_W_k)
    val_layer_0_proj_K = acc_layer_0_proj_K.to(tl.float16)

    # Vertex: Contraction (layer_0_proj_V)
    acc_layer_0_proj_V = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_layer_0_proj_V += tl.dot(val_layer_0_rms_norm1, val_layer_0_W_v)
    val_layer_0_proj_V = acc_layer_0_proj_V.to(tl.float16)

    # Vertex: Attention (layer_0_attention)
    # Fused FlashAttention Online Softmax Block
    scores_layer_0_attention = tl.dot(val_layer_0_proj_Q, tl.trans(val_layer_0_proj_K)) * (1.0 / 8.0)
    m_i = tl.max(scores_layer_0_attention, axis=1)
    p_i = tl.exp(scores_layer_0_attention - m_i[:, None])
    val_layer_0_attention = tl.dot(p_i.to(tl.float16), val_layer_0_proj_V)

    # Vertex: Contraction (layer_0_proj_O)
    acc_layer_0_proj_O = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_layer_0_proj_O += tl.dot(val_layer_0_attention, val_layer_0_W_o)
    val_layer_0_proj_O = acc_layer_0_proj_O.to(tl.float16)

    # Vertex: ResidualAdd (layer_0_res_attn)
    val_layer_0_res_attn = val_x_input + val_layer_0_proj_O

    # Vertex: Norm (layer_0_rms_norm2)
    mean_sq_layer_0_rms_norm2 = tl.sum(val_layer_0_res_attn * val_layer_0_res_attn, axis=1) / BLOCK_N
    inv_rms_layer_0_rms_norm2 = 1.0 / tl.sqrt(mean_sq_layer_0_rms_norm2 + 1e-5)
    val_layer_0_rms_norm2 = val_layer_0_res_attn * inv_rms_layer_0_rms_norm2[:, None]

    # Vertex: Contraction (layer_0_proj_gate)
    acc_layer_0_proj_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_layer_0_proj_gate += tl.dot(val_layer_0_rms_norm2, val_layer_0_W_gate)
    val_layer_0_proj_gate = acc_layer_0_proj_gate.to(tl.float16)

    # Vertex: Contraction (layer_0_proj_up)
    acc_layer_0_proj_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_layer_0_proj_up += tl.dot(val_layer_0_rms_norm2, val_layer_0_W_up)
    val_layer_0_proj_up = acc_layer_0_proj_up.to(tl.float16)

    # Vertex: Pointwise (layer_0_silu)
    val_layer_0_silu = val_layer_0_proj_gate * 0.5 * (1.0 + tl.erf(val_layer_0_proj_gate * 0.70710678))

    # Vertex: Pointwise (layer_0_swiglu_mul)
    val_layer_0_swiglu_mul = val_layer_0_silu * val_layer_0_proj_up

    # Vertex: Contraction (layer_0_proj_down)
    acc_layer_0_proj_down = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_layer_0_proj_down += tl.dot(val_layer_0_swiglu_mul, val_layer_0_W_down)
    val_layer_0_proj_down = acc_layer_0_proj_down.to(tl.float16)

    # Vertex: ResidualAdd (layer_0_res_ffn)
    val_layer_0_res_ffn = val_layer_0_res_attn + val_layer_0_proj_down

    # Vertex: Norm (layer_1_rms_norm1)
    mean_sq_layer_1_rms_norm1 = tl.sum(val_layer_0_res_ffn * val_layer_0_res_ffn, axis=1) / BLOCK_N
    inv_rms_layer_1_rms_norm1 = 1.0 / tl.sqrt(mean_sq_layer_1_rms_norm1 + 1e-5)
    val_layer_1_rms_norm1 = val_layer_0_res_ffn * inv_rms_layer_1_rms_norm1[:, None]

    # Vertex: Contraction (layer_1_proj_Q)
    acc_layer_1_proj_Q = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_layer_1_proj_Q += tl.dot(val_layer_1_rms_norm1, val_layer_1_W_q)
    val_layer_1_proj_Q = acc_layer_1_proj_Q.to(tl.float16)

    # Vertex: Contraction (layer_1_proj_K)
    acc_layer_1_proj_K = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_layer_1_proj_K += tl.dot(val_layer_1_rms_norm1, val_layer_1_W_k)
    val_layer_1_proj_K = acc_layer_1_proj_K.to(tl.float16)

    # Vertex: Contraction (layer_1_proj_V)
    acc_layer_1_proj_V = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_layer_1_proj_V += tl.dot(val_layer_1_rms_norm1, val_layer_1_W_v)
    val_layer_1_proj_V = acc_layer_1_proj_V.to(tl.float16)

    # Vertex: Attention (layer_1_attention)
    # Fused FlashAttention Online Softmax Block
    scores_layer_1_attention = tl.dot(val_layer_1_proj_Q, tl.trans(val_layer_1_proj_K)) * (1.0 / 8.0)
    m_i = tl.max(scores_layer_1_attention, axis=1)
    p_i = tl.exp(scores_layer_1_attention - m_i[:, None])
    val_layer_1_attention = tl.dot(p_i.to(tl.float16), val_layer_1_proj_V)

    # Vertex: Contraction (layer_1_proj_O)
    acc_layer_1_proj_O = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_layer_1_proj_O += tl.dot(val_layer_1_attention, val_layer_1_W_o)
    val_layer_1_proj_O = acc_layer_1_proj_O.to(tl.float16)

    # Vertex: ResidualAdd (layer_1_res_attn)
    val_layer_1_res_attn = val_layer_0_res_ffn + val_layer_1_proj_O

    # Vertex: Norm (layer_1_rms_norm2)
    mean_sq_layer_1_rms_norm2 = tl.sum(val_layer_1_res_attn * val_layer_1_res_attn, axis=1) / BLOCK_N
    inv_rms_layer_1_rms_norm2 = 1.0 / tl.sqrt(mean_sq_layer_1_rms_norm2 + 1e-5)
    val_layer_1_rms_norm2 = val_layer_1_res_attn * inv_rms_layer_1_rms_norm2[:, None]

    # Vertex: Contraction (layer_1_proj_gate)
    acc_layer_1_proj_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_layer_1_proj_gate += tl.dot(val_layer_1_rms_norm2, val_layer_1_W_gate)
    val_layer_1_proj_gate = acc_layer_1_proj_gate.to(tl.float16)

    # Vertex: Contraction (layer_1_proj_up)
    acc_layer_1_proj_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_layer_1_proj_up += tl.dot(val_layer_1_rms_norm2, val_layer_1_W_up)
    val_layer_1_proj_up = acc_layer_1_proj_up.to(tl.float16)

    # Vertex: Pointwise (layer_1_silu)
    val_layer_1_silu = val_layer_1_proj_gate * 0.5 * (1.0 + tl.erf(val_layer_1_proj_gate * 0.70710678))

    # Vertex: Pointwise (layer_1_swiglu_mul)
    val_layer_1_swiglu_mul = val_layer_1_silu * val_layer_1_proj_up

    # Vertex: Contraction (layer_1_proj_down)
    acc_layer_1_proj_down = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_layer_1_proj_down += tl.dot(val_layer_1_swiglu_mul, val_layer_1_W_down)
    val_layer_1_proj_down = acc_layer_1_proj_down.to(tl.float16)

    # Vertex: ResidualAdd (layer_1_res_ffn)
    val_layer_1_res_ffn = val_layer_1_res_attn + val_layer_1_proj_down

    # Vertex: Norm (final_rms_norm)
    mean_sq_final_rms_norm = tl.sum(val_layer_1_res_ffn * val_layer_1_res_ffn, axis=1) / BLOCK_N
    inv_rms_final_rms_norm = 1.0 / tl.sqrt(mean_sq_final_rms_norm + 1e-5)
    val_final_rms_norm = val_layer_1_res_ffn * inv_rms_final_rms_norm[:, None]

    # Vertex: Output (logits_out)
    ptr_out_logits_out = ptr_logits_out + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim
    tl.store(ptr_out_logits_out, val_final_rms_norm)


def launch_llama3_2layer_architecture_triton(*tensors):
    """Launcher function for llama3_2layer_architecture_triton_kernel."""
    x_in = tensors[0]
    B, S, D = x_in.shape
    out_tensor = torch.empty_like(x_in)
    grid = (triton.cdiv(S, 128), triton.cdiv(D, 128), B)
    
    llama3_2layer_architecture_triton_kernel[grid](
        tensors[0],
        tensors[1],
        tensors[2],
        tensors[3],
        tensors[4],
        tensors[5],
        tensors[6],
        tensors[7],
        tensors[8],
        tensors[9],
        tensors[10],
        tensors[11],
        tensors[12],
        tensors[13],
        tensors[14],
        out_tensor,
        x_in.stride(0), x_in.stride(1), x_in.stride(2),
        N_SIZE=D,
        num_warps=4,
        num_stages=3,
    )
    return out_tensor