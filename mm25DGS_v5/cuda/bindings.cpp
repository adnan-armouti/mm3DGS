// pybind11 bindings for the mm25DGS_v5 CUDA extension.
//
// Exposes thin wrappers that: (1) check tensors are CUDA + contiguous +
// the right dtype, (2) call into the .cu launcher, (3) return outputs.
//
// Phase A only exposes a build-verification entry point and a working
// scatter_splat. The BSDF forward/backward are Phase B / C stubs.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

namespace mm25v5 {

void launch_bsdf_forward_stub(
    float* f_cos, int64_t n_elem, cudaStream_t stream);

void launch_bsdf_backward_stub(
    float* grad_raw_mat, int64_t n_elem, cudaStream_t stream);

void launch_scatter_splat(
    const float* contrib_real, const float* contrib_imag,
    const int64_t* flat_idx,
    float* rp_real, float* rp_imag,
    int64_t n_items, int64_t out_size, cudaStream_t stream);

} // namespace mm25v5

#define CHECK_CUDA(x)       TORCH_CHECK(x.is_cuda(),         #x " must be a CUDA tensor")
#define CHECK_CONTIG(x)     TORCH_CHECK(x.is_contiguous(),   #x " must be contiguous")
#define CHECK_FLOAT32(x)    TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INT64(x)      TORCH_CHECK(x.scalar_type() == at::kLong, #x " must be int64")

// Phase A: smoke test — zero-fill a float tensor on the current stream.
// Used by tests to confirm the extension loads and launches a kernel.
torch::Tensor bsdf_forward_stub(torch::Tensor f_cos) {
    CHECK_CUDA(f_cos);
    CHECK_CONTIG(f_cos);
    CHECK_FLOAT32(f_cos);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_bsdf_forward_stub(
        f_cos.data_ptr<float>(), f_cos.numel(), stream.stream());
    return f_cos;
}

torch::Tensor bsdf_backward_stub(torch::Tensor grad_raw_mat) {
    CHECK_CUDA(grad_raw_mat);
    CHECK_CONTIG(grad_raw_mat);
    CHECK_FLOAT32(grad_raw_mat);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_bsdf_backward_stub(
        grad_raw_mat.data_ptr<float>(), grad_raw_mat.numel(), stream.stream());
    return grad_raw_mat;
}

// Phase A: working atomic scatter-add. Takes flattened contributions and
// scatters them into rp_real / rp_imag. Equivalent to
//   rp_real.view(-1).scatter_add_(0, flat_idx, contrib_real)
//   rp_imag.view(-1).scatter_add_(0, flat_idx, contrib_imag)
// but issues only one kernel launch for both and reads flat_idx once.
void scatter_splat(
    torch::Tensor contrib_real, torch::Tensor contrib_imag,
    torch::Tensor flat_idx,
    torch::Tensor rp_real, torch::Tensor rp_imag)
{
    CHECK_CUDA(contrib_real); CHECK_CONTIG(contrib_real); CHECK_FLOAT32(contrib_real);
    CHECK_CUDA(contrib_imag); CHECK_CONTIG(contrib_imag); CHECK_FLOAT32(contrib_imag);
    CHECK_CUDA(flat_idx);     CHECK_CONTIG(flat_idx);     CHECK_INT64(flat_idx);
    CHECK_CUDA(rp_real);      CHECK_CONTIG(rp_real);      CHECK_FLOAT32(rp_real);
    CHECK_CUDA(rp_imag);      CHECK_CONTIG(rp_imag);      CHECK_FLOAT32(rp_imag);

    TORCH_CHECK(contrib_real.numel() == contrib_imag.numel(),
                "contrib_real and contrib_imag must have same numel");
    TORCH_CHECK(flat_idx.numel() == contrib_real.numel(),
                "flat_idx must match contrib numel");
    TORCH_CHECK(rp_real.numel() == rp_imag.numel(),
                "rp_real and rp_imag must have same numel");

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_scatter_splat(
        contrib_real.data_ptr<float>(),
        contrib_imag.data_ptr<float>(),
        flat_idx.data_ptr<int64_t>(),
        rp_real.data_ptr<float>(),
        rp_imag.data_ptr<float>(),
        contrib_real.numel(),
        rp_real.numel(),
        stream.stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "mm25DGS_v5 CUDA kernels (Phase A: stubs + working scatter_splat)";
    m.def("bsdf_forward_stub", &bsdf_forward_stub,
          "Phase A stub for BSDF forward (zero-fills the output).");
    m.def("bsdf_backward_stub", &bsdf_backward_stub,
          "Phase A stub for BSDF backward (zero-fills the output).");
    m.def("scatter_splat", &scatter_splat,
          "Atomic scatter splat into (rp_real, rp_imag). Equivalent to "
          "two scatter_add_ calls, but with a single kernel launch.");
}
