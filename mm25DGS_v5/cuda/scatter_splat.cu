// Phase A stub: scatter splat kernel. Full implementation in Phase D.
// See md/v4_cuda_kernel_plan.md.

#include <cuda_runtime.h>
#include <torch/types.h>

#include "utils.cuh"

namespace mm25v5 {

__global__ void scatter_splat_kernel_phaseA(
    const float* __restrict__ contrib_real,
    const float* __restrict__ contrib_imag,
    const int64_t* __restrict__ flat_idx,
    float* __restrict__ rp_real,
    float* __restrict__ rp_imag,
    int64_t n_items,
    int64_t out_size)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n_items) return;
    int64_t idx = flat_idx[i];
    if (idx < 0 || idx >= out_size) return;
    atomicAdd(rp_real + idx, contrib_real[i]);
    atomicAdd(rp_imag + idx, contrib_imag[i]);
}

void launch_scatter_splat(
    const float* contrib_real, const float* contrib_imag,
    const int64_t* flat_idx,
    float* rp_real, float* rp_imag,
    int64_t n_items, int64_t out_size, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    const int grid = (int)((n_items + BLOCK - 1) / BLOCK);
    if (grid > 0) {
        scatter_splat_kernel_phaseA<<<grid, BLOCK, 0, stream>>>(
            contrib_real, contrib_imag, flat_idx,
            rp_real, rp_imag, n_items, out_size);
    }
}

} // namespace mm25v5
