"""
Visualize I/Q data from FMCW radar single chip frame.

This script loads radar I/Q data and creates publication-quality plots
showing the In-phase (I) and Quadrature (Q) components for the first
TX-RX pair and first chirp across all range bins.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Set publication-quality defaults
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['grid.linewidth'] = 0.6
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['ytick.major.pad'] = 8
mpl.rcParams['axes.labelpad'] = 10

# File paths
data_path = Path(__file__).resolve().parents[3] / 'data' / 'seq_1_frame_185' / 'radar' / 'cascaded_frame_185.npy'
output_dir = Path(__file__).resolve().parents[3] / 'output'

# Load radar data
print(f"Loading radar data from {data_path}...")
radar_data = np.load(data_path)
print(f"Data shape: {radar_data.shape}")
print(f"Data dtype: {radar_data.dtype}")

# TX-RX pairs to visualize
tx_rx_pairs = [
    (0, 0),  # TX0-RX0
    (0, 1),  # TX0-RX1
]

chirp_idx = 0

# Create plots for each TX-RX pair
for tx_idx, rx_idx in tx_rx_pairs:
    # Extract data for this TX-RX pair
    iq_signal = radar_data[chirp_idx, tx_idx, rx_idx, :]
    n_range_bins = len(iq_signal)

    print(f"\nProcessing TX{tx_idx}-RX{rx_idx}:")
    print(f"  Chirp: {chirp_idx}")
    print(f"  Range bins: {n_range_bins}")

    # Extract I and Q components
    i_component = np.real(iq_signal)
    q_component = np.imag(iq_signal)

    # Calculate magnitude and phase for statistics
    magnitude = np.abs(iq_signal)
    phase = np.angle(iq_signal)

    # Create range bin axis
    range_bins = np.arange(n_range_bins)

    # Create publication-quality figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot I and Q components
    ax.plot(range_bins, i_component, label='In-phase (I)', color='#2E86AB', alpha=0.95, linewidth=2.0)
    ax.plot(range_bins, q_component, label='Quadrature (Q)', color='#A23B72', alpha=0.95, linewidth=2.0)

    # Labels (no title)
    ax.set_xlabel('Range Bin', fontweight='normal')
    ax.set_ylabel('Amplitude', fontweight='normal')

    # Legend
    ax.legend(loc='upper right', framealpha=0.98, edgecolor='black', fancybox=False)

    # Grid and horizontal line
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
    ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.4, linestyle='-')

    # Remove left margin and set x-axis to start at 0
    ax.set_xlim(left=0, right=n_range_bins-1)

    # Set x-axis and y-axis ticks with regular spacing
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(MultipleLocator(12.5))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(25))

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = output_dir / f'iq_visualization_tx{tx_idx}_rx{rx_idx}.png'
    print(f"Saving figure to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved successfully!")

    # Print statistics
    print(f"Data Statistics for TX{tx_idx}-RX{rx_idx}:")
    print(f"  I component - Min: {i_component.min():.4e}, Max: {i_component.max():.4e}, Mean: {i_component.mean():.4e}")
    print(f"  Q component - Min: {q_component.min():.4e}, Max: {q_component.max():.4e}, Mean: {q_component.mean():.4e}")
    print(f"  Magnitude   - Min: {magnitude.min():.4e}, Max: {magnitude.max():.4e}, Mean: {magnitude.mean():.4e}")
    print(f"  Phase       - Min: {phase.min():.4f}, Max: {phase.max():.4f}")

    # Close the figure to free memory
    plt.close()

print("\n✓ All visualizations complete!")
