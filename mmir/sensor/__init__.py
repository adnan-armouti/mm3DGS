"""FMCW radar sensor model.

Provides the radar front-end: antenna array geometry, element gain patterns,
and the FMCW ADC phasor accumulator that converts ray-traced paths into
complex baseband samples.

Modules
-------
config              TX/RX array layout, FMCWConfig dataclass
element_patterns    Antenna element gain patterns (isotropic, cosine, phased-array)
adc_accumulator     Vectorized FMCW ADC phasor accumulation
"""
