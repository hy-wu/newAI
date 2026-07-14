// ====================================================================
// Auto-Generated NVIDIA CUDA Tile IR Code for FDIR: 'llama3_2layer_architecture'
// Specification: https://docs.nvidia.com/cuda/tile-ir/
// Tile Sizing: TILE_M=128, TILE_N=128, TILE_K=32
// ====================================================================
#include <cuda/tile>
#include <cuda_fp16.h>

using namespace cuda::tile;

__global__ void __launch_bounds__(128)
llama3_2layer_architecture_cuda_tile_kernel(
    const half* __restrict__ ptr_x_input,
    const half* __restrict__ ptr_layer_0_W_q,
    const half* __restrict__ ptr_layer_0_W_k,
    const half* __restrict__ ptr_layer_0_W_v,
    const half* __restrict__ ptr_layer_0_W_o,
    const half* __restrict__ ptr_layer_0_W_gate,
    const half* __restrict__ ptr_layer_0_W_up,
    const half* __restrict__ ptr_layer_0_W_down,
    const half* __restrict__ ptr_layer_1_W_q,
    const half* __restrict__ ptr_layer_1_W_k,
    const half* __restrict__ ptr_layer_1_W_v,
    const half* __restrict__ ptr_layer_1_W_o,
    const half* __restrict__ ptr_layer_1_W_gate,
    const half* __restrict__ ptr_layer_1_W_up,
    const half* __restrict__ ptr_layer_1_W_down,
    half* __restrict__ ptr_logits_out
) {
    // 1. Thread and Tile Block Indexing
    int tile_idx_m = blockIdx.x;
    int tile_idx_n = blockIdx.y;
    int batch_idx  = blockIdx.z;

    // 2. Tile Registers & Execution pipeline
    // Vertex: Input (x_input)
    tile<half, 128, 128> tile_x_input;
    load(tile_x_input, ptr_x_input + batch_idx * offset, layout::row_major());
    // Vertex: Input (layer_0_W_q)
    tile<half, 128, 128> tile_layer_0_W_q;
    load(tile_layer_0_W_q, ptr_layer_0_W_q + batch_idx * offset, layout::row_major());
    // Vertex: Input (layer_0_W_k)
    tile<half, 128, 128> tile_layer_0_W_k;
    load(tile_layer_0_W_k, ptr_layer_0_W_k + batch_idx * offset, layout::row_major());
    // Vertex: Input (layer_0_W_v)
    tile<half, 128, 128> tile_layer_0_W_v;
    load(tile_layer_0_W_v, ptr_layer_0_W_v + batch_idx * offset, layout::row_major());
    // Vertex: Input (layer_0_W_o)
    tile<half, 128, 128> tile_layer_0_W_o;
    load(tile_layer_0_W_o, ptr_layer_0_W_o + batch_idx * offset, layout::row_major());
    // Vertex: Input (layer_0_W_gate)
    tile<half, 128, 128> tile_layer_0_W_gate;
    load(tile_layer_0_W_gate, ptr_layer_0_W_gate + batch_idx * offset, layout::row_major());
    // Vertex: Input (layer_0_W_up)
    tile<half, 128, 128> tile_layer_0_W_up;
    load(tile_layer_0_W_up, ptr_layer_0_W_up + batch_idx * offset, layout::row_major());
    // Vertex: Input (layer_0_W_down)
    tile<half, 128, 128> tile_layer_0_W_down;
    load(tile_layer_0_W_down, ptr_layer_0_W_down + batch_idx * offset, layout::row_major());
    // Vertex: Input (layer_1_W_q)
    tile<half, 128, 128> tile_layer_1_W_q;
    load(tile_layer_1_W_q, ptr_layer_1_W_q + batch_idx * offset, layout::row_major());
    // Vertex: Input (layer_1_W_k)
    tile<half, 128, 128> tile_layer_1_W_k;
    load(tile_layer_1_W_k, ptr_layer_1_W_k + batch_idx * offset, layout::row_major());
    // Vertex: Input (layer_1_W_v)
    tile<half, 128, 128> tile_layer_1_W_v;
    load(tile_layer_1_W_v, ptr_layer_1_W_v + batch_idx * offset, layout::row_major());
    // Vertex: Input (layer_1_W_o)
    tile<half, 128, 128> tile_layer_1_W_o;
    load(tile_layer_1_W_o, ptr_layer_1_W_o + batch_idx * offset, layout::row_major());
    // Vertex: Input (layer_1_W_gate)
    tile<half, 128, 128> tile_layer_1_W_gate;
    load(tile_layer_1_W_gate, ptr_layer_1_W_gate + batch_idx * offset, layout::row_major());
    // Vertex: Input (layer_1_W_up)
    tile<half, 128, 128> tile_layer_1_W_up;
    load(tile_layer_1_W_up, ptr_layer_1_W_up + batch_idx * offset, layout::row_major());
    // Vertex: Input (layer_1_W_down)
    tile<half, 128, 128> tile_layer_1_W_down;
    load(tile_layer_1_W_down, ptr_layer_1_W_down + batch_idx * offset, layout::row_major());
    // Vertex: Norm (layer_0_rms_norm1)
    tile<half, 128, 128> tile_layer_0_rms_norm1 = tile_rmsnorm(tile_x_input);
    // Vertex: Contraction (layer_0_proj_Q)
    tile<float, 128, 128> acc_layer_0_proj_Q = 0.0f;
    #pragma unroll
    for (int k_tile = 0; k_tile < K_TOTAL; k_tile += 32) {
        tile<half, 128, 32> tile_a = load(tile_layer_0_rms_norm1);
        tile<half, 32, 128> tile_b = load(tile_layer_0_W_q, layout::row_major());
        mma(acc_layer_0_proj_Q, tile_a, tile_b, acc_layer_0_proj_Q); // Tensor Core HW MMA
    }
    tile<half, 128, 128> tile_layer_0_proj_Q = cast<half>(acc_layer_0_proj_Q);
    // Vertex: Contraction (layer_0_proj_K)
    tile<float, 128, 128> acc_layer_0_proj_K = 0.0f;
    #pragma unroll
    for (int k_tile = 0; k_tile < K_TOTAL; k_tile += 32) {
        tile<half, 128, 32> tile_a = load(tile_layer_0_rms_norm1);
        tile<half, 32, 128> tile_b = load(tile_layer_0_W_k, layout::row_major());
        mma(acc_layer_0_proj_K, tile_a, tile_b, acc_layer_0_proj_K); // Tensor Core HW MMA
    }
    tile<half, 128, 128> tile_layer_0_proj_K = cast<half>(acc_layer_0_proj_K);
    // Vertex: Contraction (layer_0_proj_V)
    tile<float, 128, 128> acc_layer_0_proj_V = 0.0f;
    #pragma unroll
    for (int k_tile = 0; k_tile < K_TOTAL; k_tile += 32) {
        tile<half, 128, 32> tile_a = load(tile_layer_0_rms_norm1);
        tile<half, 32, 128> tile_b = load(tile_layer_0_W_v, layout::row_major());
        mma(acc_layer_0_proj_V, tile_a, tile_b, acc_layer_0_proj_V); // Tensor Core HW MMA
    }
    tile<half, 128, 128> tile_layer_0_proj_V = cast<half>(acc_layer_0_proj_V);
    // Vertex: Attention (layer_0_attention)
    // Fused FlashAttention Tile Interaction Loop
    tile<float, 128, 128> acc_layer_0_attention = 0.0f;
    tile<float, 128, 1> max_score = -inf;
    tile<float, 128, 1> sum_exp = 0.0f;
    for (int s2 = 0; s2 < S_LEN; s2 += 128) {
        tile<half, 128, 32> tile_q = load(tile_layer_0_proj_Q);
        tile<half, 32, 128> tile_k = load(tile_layer_0_proj_K, layout::col_major());
        tile<float, 128, 128> S_tile = 0.0f;
        mma(S_tile, tile_q, tile_k, S_tile);
        S_tile = scale(S_tile, 1.0f / sqrtf(64.0f));
        tile<float, 128, 1> new_max = reduce_max(S_tile, dim=1);
        tile<float, 128, 128> P_tile = exp(S_tile - new_max);
        tile<half, 128, 32> tile_v = load(tile_layer_0_proj_V);
        mma(acc_layer_0_attention, cast<half>(P_tile), tile_v, acc_layer_0_attention);
    }
    tile<half, 128, 128> tile_layer_0_attention = cast<half>(acc_layer_0_attention);
    // Vertex: Contraction (layer_0_proj_O)
    tile<float, 128, 128> acc_layer_0_proj_O = 0.0f;
    #pragma unroll
    for (int k_tile = 0; k_tile < K_TOTAL; k_tile += 32) {
        tile<half, 128, 32> tile_a = load(tile_layer_0_attention);
        tile<half, 32, 128> tile_b = load(tile_layer_0_W_o, layout::row_major());
        mma(acc_layer_0_proj_O, tile_a, tile_b, acc_layer_0_proj_O); // Tensor Core HW MMA
    }
    tile<half, 128, 128> tile_layer_0_proj_O = cast<half>(acc_layer_0_proj_O);
    // Vertex: ResidualAdd (layer_0_res_attn)
    tile<half, 128, 128> tile_layer_0_res_attn = add(tile_x_input, tile_layer_0_proj_O);
    // Vertex: Norm (layer_0_rms_norm2)
    tile<half, 128, 128> tile_layer_0_rms_norm2 = tile_rmsnorm(tile_layer_0_res_attn);
    // Vertex: Contraction (layer_0_proj_gate)
    tile<float, 128, 128> acc_layer_0_proj_gate = 0.0f;
    #pragma unroll
    for (int k_tile = 0; k_tile < K_TOTAL; k_tile += 32) {
        tile<half, 128, 32> tile_a = load(tile_layer_0_rms_norm2);
        tile<half, 32, 128> tile_b = load(tile_layer_0_W_gate, layout::row_major());
        mma(acc_layer_0_proj_gate, tile_a, tile_b, acc_layer_0_proj_gate); // Tensor Core HW MMA
    }
    tile<half, 128, 128> tile_layer_0_proj_gate = cast<half>(acc_layer_0_proj_gate);
    // Vertex: Contraction (layer_0_proj_up)
    tile<float, 128, 128> acc_layer_0_proj_up = 0.0f;
    #pragma unroll
    for (int k_tile = 0; k_tile < K_TOTAL; k_tile += 32) {
        tile<half, 128, 32> tile_a = load(tile_layer_0_rms_norm2);
        tile<half, 32, 128> tile_b = load(tile_layer_0_W_up, layout::row_major());
        mma(acc_layer_0_proj_up, tile_a, tile_b, acc_layer_0_proj_up); // Tensor Core HW MMA
    }
    tile<half, 128, 128> tile_layer_0_proj_up = cast<half>(acc_layer_0_proj_up);
    // Vertex: Pointwise (layer_0_silu)
    tile<half, 128, 128> tile_layer_0_silu = apply(op::gelu(), tile_layer_0_proj_gate);
    // Vertex: Pointwise (layer_0_swiglu_mul)
    tile<half, 128, 128> tile_layer_0_swiglu_mul = apply(op::mul(), tile_layer_0_silu, tile_layer_0_proj_up);
    // Vertex: Contraction (layer_0_proj_down)
    tile<float, 128, 128> acc_layer_0_proj_down = 0.0f;
    #pragma unroll
    for (int k_tile = 0; k_tile < K_TOTAL; k_tile += 32) {
        tile<half, 128, 32> tile_a = load(tile_layer_0_swiglu_mul);
        tile<half, 32, 128> tile_b = load(tile_layer_0_W_down, layout::row_major());
        mma(acc_layer_0_proj_down, tile_a, tile_b, acc_layer_0_proj_down); // Tensor Core HW MMA
    }
    tile<half, 128, 128> tile_layer_0_proj_down = cast<half>(acc_layer_0_proj_down);
    // Vertex: ResidualAdd (layer_0_res_ffn)
    tile<half, 128, 128> tile_layer_0_res_ffn = add(tile_layer_0_res_attn, tile_layer_0_proj_down);
    // Vertex: Norm (layer_1_rms_norm1)
    tile<half, 128, 128> tile_layer_1_rms_norm1 = tile_rmsnorm(tile_layer_0_res_ffn);
    // Vertex: Contraction (layer_1_proj_Q)
    tile<float, 128, 128> acc_layer_1_proj_Q = 0.0f;
    #pragma unroll
    for (int k_tile = 0; k_tile < K_TOTAL; k_tile += 32) {
        tile<half, 128, 32> tile_a = load(tile_layer_1_rms_norm1);
        tile<half, 32, 128> tile_b = load(tile_layer_1_W_q, layout::row_major());
        mma(acc_layer_1_proj_Q, tile_a, tile_b, acc_layer_1_proj_Q); // Tensor Core HW MMA
    }
    tile<half, 128, 128> tile_layer_1_proj_Q = cast<half>(acc_layer_1_proj_Q);
    // Vertex: Contraction (layer_1_proj_K)
    tile<float, 128, 128> acc_layer_1_proj_K = 0.0f;
    #pragma unroll
    for (int k_tile = 0; k_tile < K_TOTAL; k_tile += 32) {
        tile<half, 128, 32> tile_a = load(tile_layer_1_rms_norm1);
        tile<half, 32, 128> tile_b = load(tile_layer_1_W_k, layout::row_major());
        mma(acc_layer_1_proj_K, tile_a, tile_b, acc_layer_1_proj_K); // Tensor Core HW MMA
    }
    tile<half, 128, 128> tile_layer_1_proj_K = cast<half>(acc_layer_1_proj_K);
    // Vertex: Contraction (layer_1_proj_V)
    tile<float, 128, 128> acc_layer_1_proj_V = 0.0f;
    #pragma unroll
    for (int k_tile = 0; k_tile < K_TOTAL; k_tile += 32) {
        tile<half, 128, 32> tile_a = load(tile_layer_1_rms_norm1);
        tile<half, 32, 128> tile_b = load(tile_layer_1_W_v, layout::row_major());
        mma(acc_layer_1_proj_V, tile_a, tile_b, acc_layer_1_proj_V); // Tensor Core HW MMA
    }
    tile<half, 128, 128> tile_layer_1_proj_V = cast<half>(acc_layer_1_proj_V);
    // Vertex: Attention (layer_1_attention)
    // Fused FlashAttention Tile Interaction Loop
    tile<float, 128, 128> acc_layer_1_attention = 0.0f;
    tile<float, 128, 1> max_score = -inf;
    tile<float, 128, 1> sum_exp = 0.0f;
    for (int s2 = 0; s2 < S_LEN; s2 += 128) {
        tile<half, 128, 32> tile_q = load(tile_layer_1_proj_Q);
        tile<half, 32, 128> tile_k = load(tile_layer_1_proj_K, layout::col_major());
        tile<float, 128, 128> S_tile = 0.0f;
        mma(S_tile, tile_q, tile_k, S_tile);
        S_tile = scale(S_tile, 1.0f / sqrtf(64.0f));
        tile<float, 128, 1> new_max = reduce_max(S_tile, dim=1);
        tile<float, 128, 128> P_tile = exp(S_tile - new_max);
        tile<half, 128, 32> tile_v = load(tile_layer_1_proj_V);
        mma(acc_layer_1_attention, cast<half>(P_tile), tile_v, acc_layer_1_attention);
    }
    tile<half, 128, 128> tile_layer_1_attention = cast<half>(acc_layer_1_attention);
    // Vertex: Contraction (layer_1_proj_O)
    tile<float, 128, 128> acc_layer_1_proj_O = 0.0f;
    #pragma unroll
    for (int k_tile = 0; k_tile < K_TOTAL; k_tile += 32) {
        tile<half, 128, 32> tile_a = load(tile_layer_1_attention);
        tile<half, 32, 128> tile_b = load(tile_layer_1_W_o, layout::row_major());
        mma(acc_layer_1_proj_O, tile_a, tile_b, acc_layer_1_proj_O); // Tensor Core HW MMA
    }
    tile<half, 128, 128> tile_layer_1_proj_O = cast<half>(acc_layer_1_proj_O);
    // Vertex: ResidualAdd (layer_1_res_attn)
    tile<half, 128, 128> tile_layer_1_res_attn = add(tile_layer_0_res_ffn, tile_layer_1_proj_O);
    // Vertex: Norm (layer_1_rms_norm2)
    tile<half, 128, 128> tile_layer_1_rms_norm2 = tile_rmsnorm(tile_layer_1_res_attn);
    // Vertex: Contraction (layer_1_proj_gate)
    tile<float, 128, 128> acc_layer_1_proj_gate = 0.0f;
    #pragma unroll
    for (int k_tile = 0; k_tile < K_TOTAL; k_tile += 32) {
        tile<half, 128, 32> tile_a = load(tile_layer_1_rms_norm2);
        tile<half, 32, 128> tile_b = load(tile_layer_1_W_gate, layout::row_major());
        mma(acc_layer_1_proj_gate, tile_a, tile_b, acc_layer_1_proj_gate); // Tensor Core HW MMA
    }
    tile<half, 128, 128> tile_layer_1_proj_gate = cast<half>(acc_layer_1_proj_gate);
    // Vertex: Contraction (layer_1_proj_up)
    tile<float, 128, 128> acc_layer_1_proj_up = 0.0f;
    #pragma unroll
    for (int k_tile = 0; k_tile < K_TOTAL; k_tile += 32) {
        tile<half, 128, 32> tile_a = load(tile_layer_1_rms_norm2);
        tile<half, 32, 128> tile_b = load(tile_layer_1_W_up, layout::row_major());
        mma(acc_layer_1_proj_up, tile_a, tile_b, acc_layer_1_proj_up); // Tensor Core HW MMA
    }
    tile<half, 128, 128> tile_layer_1_proj_up = cast<half>(acc_layer_1_proj_up);
    // Vertex: Pointwise (layer_1_silu)
    tile<half, 128, 128> tile_layer_1_silu = apply(op::gelu(), tile_layer_1_proj_gate);
    // Vertex: Pointwise (layer_1_swiglu_mul)
    tile<half, 128, 128> tile_layer_1_swiglu_mul = apply(op::mul(), tile_layer_1_silu, tile_layer_1_proj_up);
    // Vertex: Contraction (layer_1_proj_down)
    tile<float, 128, 128> acc_layer_1_proj_down = 0.0f;
    #pragma unroll
    for (int k_tile = 0; k_tile < K_TOTAL; k_tile += 32) {
        tile<half, 128, 32> tile_a = load(tile_layer_1_swiglu_mul);
        tile<half, 32, 128> tile_b = load(tile_layer_1_W_down, layout::row_major());
        mma(acc_layer_1_proj_down, tile_a, tile_b, acc_layer_1_proj_down); // Tensor Core HW MMA
    }
    tile<half, 128, 128> tile_layer_1_proj_down = cast<half>(acc_layer_1_proj_down);
    // Vertex: ResidualAdd (layer_1_res_ffn)
    tile<half, 128, 128> tile_layer_1_res_ffn = add(tile_layer_1_res_attn, tile_layer_1_proj_down);
    // Vertex: Norm (final_rms_norm)
    tile<half, 128, 128> tile_final_rms_norm = tile_rmsnorm(tile_layer_1_res_ffn);
    // Vertex: Output (logits_out)
    store(ptr_logits_out + batch_idx * offset, tile_final_rms_norm, layout::row_major());
}