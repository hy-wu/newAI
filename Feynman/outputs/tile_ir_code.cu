// ====================================================================
// Auto-Generated NVIDIA CUDA Tile IR Code for FDIR: 'transformer_attention_formula'
// Specification: https://docs.nvidia.com/cuda/tile-ir/
// Tile Sizing: TILE_M=128, TILE_N=128, TILE_K=32
// ====================================================================
#include <cuda/tile>
#include <cuda_fp16.h>

using namespace cuda::tile;

__global__ void __launch_bounds__(128)
transformer_attention_formula_cuda_tile_kernel(
    const half* __restrict__ ptr_x,
    const half* __restrict__ ptr_W_q,
    const half* __restrict__ ptr_W_k,
    const half* __restrict__ ptr_W_v,
    half* __restrict__ ptr_out
) {
    // 1. Thread and Tile Block Indexing
    int tile_idx_m = blockIdx.x;
    int tile_idx_n = blockIdx.y;
    int batch_idx  = blockIdx.z;

    // 2. Tile Registers & Execution pipeline
    // Vertex: Input (x)
    tile<half, 128, 128> tile_x;
    load(tile_x, ptr_x + batch_idx * offset, layout::row_major());
    // Vertex: Input (W_q)
    tile<half, 128, 128> tile_W_q;
    load(tile_W_q, ptr_W_q + batch_idx * offset, layout::row_major());
    // Vertex: Input (W_k)
    tile<half, 128, 128> tile_W_k;
    load(tile_W_k, ptr_W_k + batch_idx * offset, layout::row_major());
    // Vertex: Input (W_v)
    tile<half, 128, 128> tile_W_v;
    load(tile_W_v, ptr_W_v + batch_idx * offset, layout::row_major());
    // Vertex: Contraction (proj_Q)
    tile<float, 128, 128> acc_proj_Q = 0.0f;
    #pragma unroll
    for (int k_tile = 0; k_tile < K_TOTAL; k_tile += 32) {
        tile<half, 128, 32> tile_a = load(tile_x);
        tile<half, 32, 128> tile_b = load(tile_W_q, layout::row_major());
        mma(acc_proj_Q, tile_a, tile_b, acc_proj_Q); // Tensor Core HW MMA
    }
    tile<half, 128, 128> tile_proj_Q = cast<half>(acc_proj_Q);
    // Vertex: Contraction (proj_K)
    tile<float, 128, 128> acc_proj_K = 0.0f;
    #pragma unroll
    for (int k_tile = 0; k_tile < K_TOTAL; k_tile += 32) {
        tile<half, 128, 32> tile_a = load(tile_x);
        tile<half, 32, 128> tile_b = load(tile_W_k, layout::row_major());
        mma(acc_proj_K, tile_a, tile_b, acc_proj_K); // Tensor Core HW MMA
    }
    tile<half, 128, 128> tile_proj_K = cast<half>(acc_proj_K);
    // Vertex: Contraction (proj_V)
    tile<float, 128, 128> acc_proj_V = 0.0f;
    #pragma unroll
    for (int k_tile = 0; k_tile < K_TOTAL; k_tile += 32) {
        tile<half, 128, 32> tile_a = load(tile_x);
        tile<half, 32, 128> tile_b = load(tile_W_v, layout::row_major());
        mma(acc_proj_V, tile_a, tile_b, acc_proj_V); // Tensor Core HW MMA
    }
    tile<half, 128, 128> tile_proj_V = cast<half>(acc_proj_V);
    // Vertex: Attention (sdpa_attention)
    // Fused FlashAttention Tile Interaction Loop
    tile<float, 128, 128> acc_sdpa_attention = 0.0f;
    tile<float, 128, 1> max_score = -inf;
    tile<float, 128, 1> sum_exp = 0.0f;
    for (int s2 = 0; s2 < S_LEN; s2 += 128) {
        tile<half, 128, 32> tile_q = load(tile_proj_Q);
        tile<half, 32, 128> tile_k = load(tile_proj_K, layout::col_major());
        tile<float, 128, 128> S_tile = 0.0f;
        mma(S_tile, tile_q, tile_k, S_tile);
        S_tile = scale(S_tile, 1.0f / sqrtf(64.0f));
        tile<float, 128, 1> new_max = reduce_max(S_tile, dim=1);
        tile<float, 128, 128> P_tile = exp(S_tile - new_max);
        tile<half, 128, 32> tile_v = load(tile_proj_V);
        mma(acc_sdpa_attention, cast<half>(P_tile), tile_v, acc_sdpa_attention);
    }
    tile<half, 128, 128> tile_sdpa_attention = cast<half>(acc_sdpa_attention);
    // Vertex: ResidualAdd (residual_add)
    tile<half, 128, 128> tile_residual_add = add(tile_x, tile_sdpa_attention);
    // Vertex: Norm (layer_norm)
    tile<half, 128, 128> tile_layer_norm = tile_layernorm(tile_residual_add);
    // Vertex: Output (out)
    store(ptr_out + batch_idx * offset, tile_layer_norm, layout::row_major());
}