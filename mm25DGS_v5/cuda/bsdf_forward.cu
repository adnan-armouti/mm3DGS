// Phase A stub: fused BSDF forward kernel.
//
// Exposes a single entry point `launch_bsdf_forward` that the bindings.cpp
// wrapper calls. Full implementation in Phase B (md/v4_cuda_kernel_plan.md).
//
// Stub behavior: validates launch configuration and zero-fills the outputs
// so the extension can be loaded + tested end-to-end (build / import /
// call path) without the real kernel yet. The Python autograd wrapper
// falls back to the PyTorch reference whenever the stub is active, so this
// does not affect numerical results during Phase A.

#include <cuda_runtime.h>
#include <torch/types.h>

#include "utils.cuh"

namespace mm25v5 {

// Zero-fill kernel used by the Phase A stub.
__global__ void zero_fill_kernel(float* __restrict__ ptr, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) ptr[i] = 0.0f;
}

// Phase A entry: zero-fills outputs, returns. Phase B will replace this
// with the real fused BSDF kernel.
void launch_bsdf_forward_stub(
    float* f_cos, int64_t n_elem, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    const int grid = (int)((n_elem + BLOCK - 1) / BLOCK);
    if (grid > 0) {
        zero_fill_kernel<<<grid, BLOCK, 0, stream>>>(f_cos, n_elem);
    }
}

} // namespace mm25v5
