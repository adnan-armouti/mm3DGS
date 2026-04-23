// pybind11 bindings for the v7 CUDA extension (step5_doppler fused).
//
// Exposes step5_doppler_fused forward + backward. Everything else (BSDF,
// v5 step5_fused) stays in the mm25DGS_v5 extension — v7 imports those
// by name, see mm25DGS_v7/cuda/__init__.py.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

namespace mm25v7 {

void launch_step5_doppler_fused_forward(
    const float* w_full, const float* phi_base, const float* n_peak,
    const float* A_vec, const float* t_off,
    const float* psf_real, const float* psf_imag,
    float* rp_real, float* rp_imag,
    int M, int n_chirps, int n_tx, int n_rx, int K,
    int spread, int n_grid, float w_threshold,
    cudaStream_t stream);

void launch_step5_doppler_fused_backward(
    const float* grad_rp_real, const float* grad_rp_imag,
    const float* w_full, const float* phi_base, const float* n_peak,
    const float* A_vec, const float* t_off,
    const float* psf_real, const float* psf_imag,
    float* grad_w,
    int M, int n_chirps, int n_tx, int n_rx, int K,
    int spread, int n_grid, float w_threshold,
    cudaStream_t stream);

} // namespace mm25v7

#define CHECK_CUDA(x) TORCH_CHECK((x).device().is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).dtype() == torch::kFloat32, #x " must be float32")


std::vector<torch::Tensor> step5_doppler_fused_forward(
    torch::Tensor w_full, torch::Tensor phi_base, torch::Tensor n_peak,
    torch::Tensor A_vec, torch::Tensor t_off,
    torch::Tensor psf_real, torch::Tensor psf_imag,
    int64_t K, double w_threshold)
{
    CHECK_CUDA(w_full);    CHECK_CONTIG(w_full);    CHECK_FLOAT32(w_full);
    CHECK_CUDA(phi_base);  CHECK_CONTIG(phi_base);  CHECK_FLOAT32(phi_base);
    CHECK_CUDA(n_peak);    CHECK_CONTIG(n_peak);    CHECK_FLOAT32(n_peak);
    CHECK_CUDA(A_vec);     CHECK_CONTIG(A_vec);     CHECK_FLOAT32(A_vec);
    CHECK_CUDA(t_off);     CHECK_CONTIG(t_off);     CHECK_FLOAT32(t_off);
    CHECK_CUDA(psf_real);  CHECK_CONTIG(psf_real);  CHECK_FLOAT32(psf_real);
    CHECK_CUDA(psf_imag);  CHECK_CONTIG(psf_imag);  CHECK_FLOAT32(psf_imag);

    TORCH_CHECK(w_full.dim()   == 3, "w_full must be (M, n_tx, n_rx)");
    TORCH_CHECK(phi_base.dim() == 3, "phi_base must be (M, n_tx, n_rx)");
    TORCH_CHECK(n_peak.dim()   == 3, "n_peak must be (M, n_tx, n_rx)");
    TORCH_CHECK(A_vec.dim()    == 1, "A_vec must be (M,)");
    TORCH_CHECK(t_off.dim()    == 2, "t_off must be (n_chirps, n_tx)");

    const int M        = (int)w_full.size(0);
    const int n_tx     = (int)w_full.size(1);
    const int n_rx     = (int)w_full.size(2);
    const int n_chirps = (int)t_off.size(0);
    const int spread   = (int)psf_real.size(0);
    const int n_grid   = (int)psf_real.size(1);

    TORCH_CHECK(A_vec.size(0)    == M,    "A_vec size mismatch");
    TORCH_CHECK(t_off.size(1)    == n_tx, "t_off size mismatch");

    auto opts = torch::TensorOptions()
        .dtype(torch::kFloat32).device(w_full.device());
    auto rp_real = torch::zeros({n_chirps, n_tx, n_rx, (int64_t)K}, opts);
    auto rp_imag = torch::zeros({n_chirps, n_tx, n_rx, (int64_t)K}, opts);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v7::launch_step5_doppler_fused_forward(
        w_full.data_ptr<float>(), phi_base.data_ptr<float>(),
        n_peak.data_ptr<float>(),
        A_vec.data_ptr<float>(), t_off.data_ptr<float>(),
        psf_real.data_ptr<float>(), psf_imag.data_ptr<float>(),
        rp_real.data_ptr<float>(), rp_imag.data_ptr<float>(),
        M, n_chirps, n_tx, n_rx, (int)K, spread, n_grid, (float)w_threshold,
        stream.stream());
    return {rp_real, rp_imag};
}


torch::Tensor step5_doppler_fused_backward(
    torch::Tensor grad_rp_real, torch::Tensor grad_rp_imag,
    torch::Tensor w_full, torch::Tensor phi_base, torch::Tensor n_peak,
    torch::Tensor A_vec, torch::Tensor t_off,
    torch::Tensor psf_real, torch::Tensor psf_imag,
    double w_threshold)
{
    CHECK_CUDA(grad_rp_real); CHECK_CONTIG(grad_rp_real); CHECK_FLOAT32(grad_rp_real);
    CHECK_CUDA(grad_rp_imag); CHECK_CONTIG(grad_rp_imag); CHECK_FLOAT32(grad_rp_imag);
    CHECK_CUDA(w_full);       CHECK_CONTIG(w_full);       CHECK_FLOAT32(w_full);
    CHECK_CUDA(phi_base);     CHECK_CONTIG(phi_base);     CHECK_FLOAT32(phi_base);
    CHECK_CUDA(n_peak);       CHECK_CONTIG(n_peak);       CHECK_FLOAT32(n_peak);
    CHECK_CUDA(A_vec);        CHECK_CONTIG(A_vec);        CHECK_FLOAT32(A_vec);
    CHECK_CUDA(t_off);        CHECK_CONTIG(t_off);        CHECK_FLOAT32(t_off);

    const int M        = (int)w_full.size(0);
    const int n_tx     = (int)w_full.size(1);
    const int n_rx     = (int)w_full.size(2);
    const int n_chirps = (int)t_off.size(0);
    const int K        = (int)grad_rp_real.size(3);
    const int spread   = (int)psf_real.size(0);
    const int n_grid   = (int)psf_real.size(1);

    auto opts = torch::TensorOptions()
        .dtype(torch::kFloat32).device(w_full.device());
    auto grad_w = torch::empty({M, n_tx, n_rx}, opts);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v7::launch_step5_doppler_fused_backward(
        grad_rp_real.data_ptr<float>(), grad_rp_imag.data_ptr<float>(),
        w_full.data_ptr<float>(), phi_base.data_ptr<float>(),
        n_peak.data_ptr<float>(),
        A_vec.data_ptr<float>(), t_off.data_ptr<float>(),
        psf_real.data_ptr<float>(), psf_imag.data_ptr<float>(),
        grad_w.data_ptr<float>(),
        M, n_chirps, n_tx, n_rx, K, spread, n_grid, (float)w_threshold,
        stream.stream());
    return grad_w;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "mm25DGS_v7 CUDA kernels (step5_doppler fused)";
    m.def("step5_doppler_fused_forward", &step5_doppler_fused_forward,
          "v7 fused step5 + Doppler splat. Takes (w_full, phi_base, "
          "n_peak, A_vec, t_off, psf_real, psf_imag, K, w_threshold); "
          "returns (rp_real, rp_imag) of shape (n_chirps, n_tx, n_rx, K).");
    m.def("step5_doppler_fused_backward", &step5_doppler_fused_backward,
          "Backward of step5_doppler_fused_forward. Returns grad_w of "
          "shape (M, n_tx, n_rx); phi_base, n_peak, A_vec, t_off are "
          "treated as non-differentiable in the training path.");
}
