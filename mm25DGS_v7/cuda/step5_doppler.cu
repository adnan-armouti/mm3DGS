// v7 Doppler fused Step-5 splat: replaces the 16× Python loop around
// ``mm25DGS_v5/cuda::step5_fused`` with ONE forward + ONE backward
// kernel, amortising the (M, n_tx, n_rx) BSDF path across all chirps.
//
// Per plan md/mm25dgs_v7_speed_and_ceiling.md §B2a.
//
// Inputs (all fp32, contiguous):
//   w_full     : (M, n_tx, n_rx)   — step 4 weight (= C_radar·√f_cos·α_tx·α_rx)
//   phi_base   : (M, n_tx, n_rx)   — carrier phase (2π·f0·(τ_tx+τ_rx))
//   n_peak     : (M, n_tx, n_rx)   — fractional range-bin peak
//   A          : (M,)              — Doppler factor = −(4π/λ)·⟨û, v_ego⟩
//   t_off      : (n_chirps, n_tx)  — time offset m·T_c + k(i)·T_a
//   psf_real   : (spread, n_grid)  — Hann PSF table, real part
//   psf_imag   : (spread, n_grid)
// Outputs:
//   rp_real    : (n_chirps, n_tx, n_rx, K)  — scattered, atomic
//   rp_imag    : (n_chirps, n_tx, n_rx, K)
//
// Per-path phase: phi = phi_base + A * t_off[chirp, tx_i]. n_peak and
// w_full are shared across chirps; the carrier sin/cos changes per chirp.
//
// Parallelism: one thread per (M, n_tx, n_rx) path; each thread loops
// internally over n_chirps. This amortises the per-path loads
// (w_full, phi_base, n_peak, PSF samples, bin) once across all chirps.
// Atomic contention on ``rp_*`` is identical to the single-chirp kernel:
// no cross-chirp contention (separate slices); intra-chirp contention
// unchanged.

#include <cuda_runtime.h>
#include <math.h>

namespace mm25v7 {

// Launch-time constants — must match the Python helper
// (_precompute_doppler_factors) and HannPSFTable. All are fine up to 64.
// N_CHIRPS_MAX bounds the shared-memory staging for t_off[chirp, tx].
constexpr int N_CHIRPS_MAX = 32;
constexpr int N_TX_MAX     = 32;

// -------------------------------------------------------------------------
// Forward
// -------------------------------------------------------------------------

__global__ void step5_doppler_fused_forward_kernel(
    const float* __restrict__ w_full,       // (M, n_tx, n_rx)
    const float* __restrict__ phi_base,     // (M, n_tx, n_rx)
    const float* __restrict__ n_peak,       // (M, n_tx, n_rx)
    const float* __restrict__ A_vec,        // (M,)
    const float* __restrict__ t_off,        // (n_chirps, n_tx)
    const float* __restrict__ psf_real,     // (spread, n_grid)
    const float* __restrict__ psf_imag,     // (spread, n_grid)
    float* __restrict__ rp_real,            // (n_chirps, n_tx, n_rx, K)
    float* __restrict__ rp_imag,
    int M, int n_chirps, int n_tx, int n_rx, int K,
    int spread, int n_grid,
    float w_threshold)
{
    // Shared-memory staging of t_off[chirp, tx] — (n_chirps · n_tx) floats.
    // 32·32·4 = 4 KiB max; normal case 16·12·4 = 768 B.
    __shared__ float s_t_off[N_CHIRPS_MAX * N_TX_MAX];
    const int total_t_off = n_chirps * n_tx;
    for (int i = threadIdx.x; i < total_t_off; i += blockDim.x) {
        s_t_off[i] = t_off[i];
    }
    __syncthreads();

    const int64_t total = (int64_t)M * n_tx * n_rx;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const float w = w_full[idx];
    if (w < w_threshold) return;  // all chirps of this culled path contribute 0

    const int r  = (int)(idx % n_rx);
    const int t  = (int)((idx / n_rx) % n_tx);
    const int p  = (int)(idx / ((int64_t)n_tx * n_rx));

    const float phi0     = phi_base[idx];
    const float np_peak  = n_peak[idx];
    const float A        = A_vec[p];

    const int n_floor = (int)floorf(np_peak);
    const float n_frac = np_peak - (float)n_floor;

    // PSF table index + interp (shared across chirps)
    const float idx_f = n_frac * (float)n_grid;
    int idx_lo = (int)floorf(idx_f);
    if (idx_lo < 0)         idx_lo = 0;
    if (idx_lo >= n_grid)   idx_lo = n_grid - 1;
    int idx_hi = idx_lo + 1;
    if (idx_hi >= n_grid)   idx_hi = n_grid - 1;
    const float frac = idx_f - (float)idx_lo;
    const float inv_frac = 1.0f - frac;
    const int half_spread = spread >> 1;

    // Precompute PSF samples (shared across chirps)
    float psf_r_d[15];
    float psf_i_d[15];
    #pragma unroll
    for (int d = 0; d < 15; ++d) {
        if (d >= spread) break;
        const int psf_off = d * n_grid;
        const float pr_lo = psf_real[psf_off + idx_lo];
        const float pr_hi = psf_real[psf_off + idx_hi];
        const float pi_lo = psf_imag[psf_off + idx_lo];
        const float pi_hi = psf_imag[psf_off + idx_hi];
        psf_r_d[d] = pr_lo * inv_frac + pr_hi * frac;
        psf_i_d[d] = pi_lo * inv_frac + pi_hi * frac;
    }

    // Precompute bin offsets (shared across chirps)
    int bins[15];
    #pragma unroll
    for (int d = 0; d < 15; ++d) {
        if (d >= spread) break;
        int bin = (n_floor + (d - half_spread)) % K;
        if (bin < 0) bin += K;
        bins[d] = bin;
    }

    // Per-chirp splat loop
    const int64_t stride_chirp = (int64_t)n_tx * n_rx * K;
    for (int m = 0; m < n_chirps; ++m) {
        const float t_m     = s_t_off[m * n_tx + t];
        const float phi     = phi0 + A * t_m;
        float s, c;
        sincosf(phi, &s, &c);
        const float carrier_real = w * c;
        const float carrier_imag = w * s;

        float* rp_r_m = rp_real + m * stride_chirp + (t * n_rx + r) * K;
        float* rp_i_m = rp_imag + m * stride_chirp + (t * n_rx + r) * K;

        #pragma unroll
        for (int d = 0; d < 15; ++d) {
            if (d >= spread) break;
            const float psf_r = psf_r_d[d];
            const float psf_i = psf_i_d[d];
            const float contrib_re = carrier_real * psf_r - carrier_imag * psf_i;
            const float contrib_im = carrier_real * psf_i + carrier_imag * psf_r;
            atomicAdd(&rp_r_m[bins[d]], contrib_re);
            atomicAdd(&rp_i_m[bins[d]], contrib_im);
        }
    }
}

void launch_step5_doppler_fused_forward(
    const float* w_full, const float* phi_base, const float* n_peak,
    const float* A_vec, const float* t_off,
    const float* psf_real, const float* psf_imag,
    float* rp_real, float* rp_imag,
    int M, int n_chirps, int n_tx, int n_rx, int K,
    int spread, int n_grid, float w_threshold,
    cudaStream_t stream)
{
    const int64_t total = (int64_t)M * n_tx * n_rx;
    constexpr int BLOCK = 256;
    const int grid = (int)((total + BLOCK - 1) / BLOCK);
    if (grid <= 0) return;
    step5_doppler_fused_forward_kernel<<<grid, BLOCK, 0, stream>>>(
        w_full, phi_base, n_peak, A_vec, t_off,
        psf_real, psf_imag, rp_real, rp_imag,
        M, n_chirps, n_tx, n_rx, K, spread, n_grid, w_threshold);
}

// -------------------------------------------------------------------------
// Backward — only grad_w_full is emitted (matches step5_fused_backward
// convention: phi_base, n_peak, A, t_off are all non-differentiable in
// the training path; phi is produced via detached inputs).
//
// Chain rule per (chirp, p, t, r):
//   contrib_re[m,d] = w·(c_m·psf_r − s_m·psf_i)
//   contrib_im[m,d] = w·(c_m·psf_i + s_m·psf_r)
//   grad_w += Σ_m Σ_d  grad_rp_re[m,t,r,bin_d]·(c_m·psf_r − s_m·psf_i)
//                    + grad_rp_im[m,t,r,bin_d]·(c_m·psf_i + s_m·psf_r)
// -------------------------------------------------------------------------

__global__ void step5_doppler_fused_backward_kernel(
    const float* __restrict__ grad_rp_real, // (n_chirps, n_tx, n_rx, K)
    const float* __restrict__ grad_rp_imag,
    const float* __restrict__ w_full,
    const float* __restrict__ phi_base,
    const float* __restrict__ n_peak,
    const float* __restrict__ A_vec,
    const float* __restrict__ t_off,
    const float* __restrict__ psf_real,
    const float* __restrict__ psf_imag,
    float* __restrict__ grad_w,             // (M, n_tx, n_rx)
    int M, int n_chirps, int n_tx, int n_rx, int K,
    int spread, int n_grid,
    float w_threshold)
{
    __shared__ float s_t_off[N_CHIRPS_MAX * N_TX_MAX];
    const int total_t_off = n_chirps * n_tx;
    for (int i = threadIdx.x; i < total_t_off; i += blockDim.x) {
        s_t_off[i] = t_off[i];
    }
    __syncthreads();

    const int64_t total = (int64_t)M * n_tx * n_rx;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const float w = w_full[idx];
    if (w < w_threshold) {
        grad_w[idx] = 0.0f;
        return;
    }

    const int r = (int)(idx % n_rx);
    const int t = (int)((idx / n_rx) % n_tx);
    const int p = (int)(idx / ((int64_t)n_tx * n_rx));

    const float phi0     = phi_base[idx];
    const float np_peak  = n_peak[idx];
    const float A        = A_vec[p];

    const int n_floor = (int)floorf(np_peak);
    const float n_frac = np_peak - (float)n_floor;

    const float idx_f = n_frac * (float)n_grid;
    int idx_lo = (int)floorf(idx_f);
    if (idx_lo < 0)         idx_lo = 0;
    if (idx_lo >= n_grid)   idx_lo = n_grid - 1;
    int idx_hi = idx_lo + 1;
    if (idx_hi >= n_grid)   idx_hi = n_grid - 1;
    const float frac = idx_f - (float)idx_lo;
    const float inv_frac = 1.0f - frac;
    const int half_spread = spread >> 1;

    float psf_r_d[15];
    float psf_i_d[15];
    #pragma unroll
    for (int d = 0; d < 15; ++d) {
        if (d >= spread) break;
        const int psf_off = d * n_grid;
        const float pr_lo = psf_real[psf_off + idx_lo];
        const float pr_hi = psf_real[psf_off + idx_hi];
        const float pi_lo = psf_imag[psf_off + idx_lo];
        const float pi_hi = psf_imag[psf_off + idx_hi];
        psf_r_d[d] = pr_lo * inv_frac + pr_hi * frac;
        psf_i_d[d] = pi_lo * inv_frac + pi_hi * frac;
    }

    int bins[15];
    #pragma unroll
    for (int d = 0; d < 15; ++d) {
        if (d >= spread) break;
        int bin = (n_floor + (d - half_spread)) % K;
        if (bin < 0) bin += K;
        bins[d] = bin;
    }

    const int64_t stride_chirp = (int64_t)n_tx * n_rx * K;
    float g_w = 0.0f;
    for (int m = 0; m < n_chirps; ++m) {
        const float t_m = s_t_off[m * n_tx + t];
        const float phi = phi0 + A * t_m;
        float s, c;
        sincosf(phi, &s, &c);

        const float* g_rp_r_m = grad_rp_real + m * stride_chirp + (t * n_rx + r) * K;
        const float* g_rp_i_m = grad_rp_imag + m * stride_chirp + (t * n_rx + r) * K;

        #pragma unroll
        for (int d = 0; d < 15; ++d) {
            if (d >= spread) break;
            const float psf_r = psf_r_d[d];
            const float psf_i = psf_i_d[d];
            const float dre_dw = c * psf_r - s * psf_i;
            const float dim_dw = c * psf_i + s * psf_r;
            const float g_rp_re = g_rp_r_m[bins[d]];
            const float g_rp_im = g_rp_i_m[bins[d]];
            g_w += g_rp_re * dre_dw + g_rp_im * dim_dw;
        }
    }
    grad_w[idx] = g_w;
}

void launch_step5_doppler_fused_backward(
    const float* grad_rp_real, const float* grad_rp_imag,
    const float* w_full, const float* phi_base, const float* n_peak,
    const float* A_vec, const float* t_off,
    const float* psf_real, const float* psf_imag,
    float* grad_w,
    int M, int n_chirps, int n_tx, int n_rx, int K,
    int spread, int n_grid, float w_threshold,
    cudaStream_t stream)
{
    const int64_t total = (int64_t)M * n_tx * n_rx;
    constexpr int BLOCK = 256;
    const int grid = (int)((total + BLOCK - 1) / BLOCK);
    if (grid <= 0) return;
    step5_doppler_fused_backward_kernel<<<grid, BLOCK, 0, stream>>>(
        grad_rp_real, grad_rp_imag,
        w_full, phi_base, n_peak, A_vec, t_off, psf_real, psf_imag,
        grad_w, M, n_chirps, n_tx, n_rx, K, spread, n_grid, w_threshold);
}

} // namespace mm25v7
