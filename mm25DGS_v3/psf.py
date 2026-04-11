"""Range PSF for Hann-windowed FFT.

The range FFT in adc_to_ra_image applies:
    1. Hann window: W[k] = 0.5 - 0.5*cos(2πk/K) for k=0..K-1
    2. FFT: F[n] = Σ_k (x[k] * W[k]) * exp(-j2πnk/K)

For a single sinusoid x[k] = exp(j*ω*k), the FFT output is a shifted
Dirichlet kernel (DFT of the Hann window). The Hann window's DFT is
the sum of 3 shifted Dirichlet kernels:
    PSF_hann(δ) = 0.5*D(δ) - 0.25*D(δ-1) - 0.25*D(δ+1)
where D(δ) = sin(πδ)/sin(πδ/K) * exp(-jπδ(K-1)/K) is the Dirichlet kernel.
"""

import math
import torch
from torch import Tensor

PI = math.pi
TWO_PI = 2.0 * PI


def dirichlet_kernel(delta: Tensor, K: int) -> Tensor:
    """Dirichlet kernel (periodic sinc): D(δ) = sin(πδ) / sin(πδ/K).

    At integer δ, this equals K (by L'Hôpital). We handle this with
    a smooth approximation to avoid division by zero.

    Returns complex values including the phase shift exp(-jπδ(K-1)/K).
    """
    # Magnitude: |D(δ)| = sin(πδ) / sin(πδ/K)
    pi_delta = PI * delta
    pi_delta_K = PI * delta / K

    # Near-integer handling: when |sin(πδ/K)| < ε, D → K
    sin_num = torch.sin(pi_delta)
    sin_den = torch.sin(pi_delta_K)
    near_integer = sin_den.abs() < 1e-8
    sin_den_safe = torch.where(near_integer, torch.ones_like(sin_den), sin_den)
    magnitude = torch.where(near_integer, torch.full_like(delta, float(K)), sin_num / sin_den_safe)

    # Phase: exp(-jπδ(K-1)/K)
    phase = -PI * delta * (K - 1) / K
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)

    return torch.complex(real, imag)


def hann_psf(delta: Tensor, K: int) -> Tensor:
    """Range PSF for Hann-windowed FFT.

    PSF_hann(δ) = 0.5 * D(δ) - 0.25 * D(δ-1) - 0.25 * D(δ+1)

    where D is the Dirichlet kernel of length K.

    Args:
        delta: (*,) fractional range bin offsets (real-valued)
        K: FFT length (number of ADC samples)

    Returns:
        (*,) complex PSF values
    """
    return (0.5 * dirichlet_kernel(delta, K)
            - 0.25 * dirichlet_kernel(delta - 1.0, K)
            - 0.25 * dirichlet_kernel(delta + 1.0, K))
