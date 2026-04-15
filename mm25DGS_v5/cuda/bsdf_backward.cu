// Phase A stub: fused BSDF backward kernel.
// Full implementation in Phase C (analytical gradients for 6 material params).
// See md/v4_cuda_kernel_plan.md.

#include <cuda_runtime.h>
#include <torch/types.h>

#include "utils.cuh"

namespace mm25v5 {

__global__ void zero_fill_kernel_bw(float* __restrict__ ptr, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) ptr[i] = 0.0f;
}

void launch_bsdf_backward_stub(
    float* grad_raw_mat, int64_t n_elem, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    const int grid = (int)((n_elem + BLOCK - 1) / BLOCK);
    if (grid > 0) {
        zero_fill_kernel_bw<<<grid, BLOCK, 0, stream>>>(grad_raw_mat, n_elem);
    }
}

} // namespace mm25v5
