// Device-side helpers shared by the mm25DGS_v5 CUDA kernels.
//
// Scope:
//  - Complex-number ops on float2 (the plan keeps everything in float/float2
//    registers; no std::complex, no thrust::complex).
//  - Vector-3 helpers (dot, cross, normalize).
//  - Device-side reparam for the 6 raw material parameters (matches the
//    reparameterize_torch function in mmir/bsdf_torch.py).
//  - itu_slab_fresnel port (scalar version, single path).
//
// All functions are __device__ __forceinline__ to encourage the compiler
// to keep intermediates in registers across the fused kernel.

#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

namespace mm25v5 {

// --- scalar constants -------------------------------------------------------

constexpr float C_LIGHT = 299792458.0f;
constexpr float PI      = 3.14159265358979323846f;
constexpr float TWO_PI  = 6.28318530717958647692f;
constexpr float INV_PI  = 0.31830988618379067154f;
// WAVELENGTH matches mmir/bsdf_torch.py:WAVELENGTH = 3.9e-3 (77 GHz carrier).
constexpr float WAVELENGTH = 3.9e-3f;
constexpr float K_WAVE     = TWO_PI / WAVELENGTH;     // ~1611 rad/m

// --- float2 complex ops -----------------------------------------------------

__device__ __forceinline__ float2 cadd(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ float2 csub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}
__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y,
                       a.x * b.y + a.y * b.x);
}
__device__ __forceinline__ float2 cscale(float2 a, float s) {
    return make_float2(a.x * s, a.y * s);
}
__device__ __forceinline__ float cabs_sq(float2 a) {
    return a.x * a.x + a.y * a.y;
}
// complex division a / b
__device__ __forceinline__ float2 cdiv(float2 a, float2 b) {
    float denom = b.x * b.x + b.y * b.y;
    float inv = 1.0f / fmaxf(denom, 1e-30f);
    return make_float2((a.x * b.x + a.y * b.y) * inv,
                       (a.y * b.x - a.x * b.y) * inv);
}
// complex sqrt (principal branch), using polar form.
__device__ __forceinline__ float2 csqrt(float2 a) {
    float r = sqrtf(a.x * a.x + a.y * a.y);
    float re = sqrtf(0.5f * (r + a.x));
    float im = sqrtf(fmaxf(0.5f * (r - a.x), 0.0f));
    if (a.y < 0.0f) im = -im;
    return make_float2(re, im);
}
// complex exp
__device__ __forceinline__ float2 cexp(float2 a) {
    float er = expf(a.x);
    float s, c;
    __sincosf(a.y, &s, &c);
    return make_float2(er * c, er * s);
}

// --- vec3 helpers -----------------------------------------------------------

__device__ __forceinline__ float vdot(const float ax, const float ay, const float az,
                                       const float bx, const float by, const float bz) {
    return ax * bx + ay * by + az * bz;
}
__device__ __forceinline__ void vcross(const float ax, const float ay, const float az,
                                        const float bx, const float by, const float bz,
                                        float& ox, float& oy, float& oz) {
    ox = ay * bz - az * by;
    oy = az * bx - ax * bz;
    oz = ax * by - ay * bx;
}

// --- material reparam ------------------------------------------------------
//
// Matches mm25DGS_v5/rasterizer.py:reparameterize_torch exactly:
//   raw[0] -> eps_real  = 1 + softplus(raw)
//   raw[1] -> eps_imag  = exp(clamp(raw, -7, 16))
//   raw[2] -> sigma_h   = exp(clamp(raw, -16, -7))
//   raw[3] -> l_c       = exp(clamp(raw, -10, 2))
//   raw[4] -> tau_base  = 0.05 + 0.9 * sigmoid(raw)
//   raw[5] -> thickness = exp(clamp(raw, -7, 2))

__device__ __forceinline__ float softplus(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__device__ __forceinline__ void reparam6(
    const float r0, const float r1, const float r2,
    const float r3, const float r4, const float r5,
    float& eps_real, float& eps_imag,
    float& sigma_h, float& l_c,
    float& tau_base, float& thickness)
{
    eps_real  = 1.0f + softplus(r0);
    eps_imag  = expf(clampf(r1, -7.0f, 16.0f));
    sigma_h   = expf(clampf(r2, -16.0f, -7.0f));
    l_c       = expf(clampf(r3, -10.0f, 2.0f));
    tau_base  = 0.05f + 0.9f * sigmoid(r4);
    thickness = expf(clampf(r5, -7.0f, 2.0f));
}

// --- itu_slab_fresnel port -------------------------------------------------
//
// Ported from mmir/bsdf_torch.py:itu_slab_fresnel. Computes the reflection
// coefficients R_TE, R_TM of a dielectric slab of thickness d with complex
// permittivity eps_r + i*eps_i, at incidence cosine cos_i. Assumes lossy
// medium (the Python path adds a small imaginary component for stability).
//
// Returns (R_TE, R_TM) as two float2 values via output pointers.

__device__ __forceinline__ void itu_slab_fresnel(
    float eps_real, float eps_imag, float cos_i, float thickness,
    float2& R_TE, float2& R_TM)
{
    // Match mmir/bsdf_torch.py:itu_slab_fresnel exactly.
    // Python: eta = complex(eps_real, -eps_imag)  (negative imag convention)
    //         cos_i clamped to [1e-6, 1]
    //         sin2 = 1 - cos_i^2
    //         a = csqrt(eta - sin2)
    //         r_te = (cos_i - a) / (cos_i + a + 1e-10)
    //         r_tm = (eta*cos_i - a) / (eta*cos_i + a + 1e-10)
    //         q = (2π/λ) * d * a
    //         ej2q = exp(-2j*q)
    //         R = r*(1 - ej2q) / (1 - r^2 * ej2q + 1e-10)
    cos_i = clampf(cos_i, 1e-6f, 1.0f);
    float sin2 = 1.0f - cos_i * cos_i;

    float2 eta = make_float2(eps_real, -eps_imag);
    float2 inside = csub(eta, make_float2(sin2, 0.0f));
    float2 a = csqrt(inside);

    float2 cos_i_c = make_float2(cos_i, 0.0f);
    float2 eps = make_float2(1e-10f, 0.0f);

    float2 num_te = csub(cos_i_c, a);
    float2 den_te = cadd(cadd(cos_i_c, a), eps);
    float2 r_te = cdiv(num_te, den_te);

    float2 eta_cos_i = cscale(eta, cos_i);
    float2 num_tm = csub(eta_cos_i, a);
    float2 den_tm = cadd(cadd(eta_cos_i, a), eps);
    float2 r_tm = cdiv(num_tm, den_tm);

    // q = (2π/λ) * d * a   (complex)
    float2 q = cscale(a, K_WAVE * thickness);
    // ej2q = exp(-2j * q) = exp(2*q.y + j*(-2*q.x))
    float2 ej2q = cexp(make_float2(2.0f * q.y, -2.0f * q.x));

    float2 one = make_float2(1.0f, 0.0f);
    float2 one_minus_ej2q = csub(one, ej2q);

    // R_TE = r_te * (1 - ej2q) / (1 - r_te^2 * ej2q + 1e-10)
    float2 r_te_sq = cmul(r_te, r_te);
    float2 num_TE = cmul(r_te, one_minus_ej2q);
    float2 den_TE = cadd(csub(one, cmul(r_te_sq, ej2q)), eps);
    R_TE = cdiv(num_TE, den_TE);

    float2 r_tm_sq = cmul(r_tm, r_tm);
    float2 num_TM = cmul(r_tm, one_minus_ej2q);
    float2 den_TM = cadd(csub(one, cmul(r_tm_sq, ej2q)), eps);
    R_TM = cdiv(num_TM, den_TM);
}

} // namespace mm25v5
