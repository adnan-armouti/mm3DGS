"""
Visualize MMWCAS antenna beam patterns.

Generates both line plots and polar plots for E-plane and H-plane patterns.
Saves output figures to notebooks/output/ directory.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def plot_combined_line_patterns(patterns_dict, title_prefix="", save_path=None):
    """
    Create combined line plot showing both E-plane and H-plane side by side.

    Args:
        patterns_dict: dict of {label: pattern_array}
        title_prefix: prefix for plot title
        save_path: path to save figure
    """
    angles = np.linspace(-180, 180, 361)
    colors = plt.cm.tab10(np.linspace(0, 1, len(patterns_dict)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # E-plane (left subplot)
    for (label, pattern), color in zip(patterns_dict.items(), colors):
        ax1.plot(angles, pattern[:, 0], label=label, linewidth=2, color=color)

    ax1.set_title("E-plane (Elevation)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Angle (degrees)", fontsize=12)
    ax1.set_ylabel("Gain (dBi)", fontsize=12)
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-180, 180])

    # H-plane (right subplot)
    for (label, pattern), color in zip(patterns_dict.items(), colors):
        ax2.plot(angles, pattern[:, 1], label=label, linewidth=2, color=color)

    ax2.set_title("H-plane (Azimuth)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Angle (degrees)", fontsize=12)
    ax2.set_ylabel("Gain (dBi)", fontsize=12)
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-180, 180])

    # Overall title
    fig.suptitle(f"{title_prefix}Antenna Patterns", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_combined_polar_patterns(patterns_dict, title_prefix="", save_path=None):
    """
    Create combined polar plot showing both E-plane and H-plane side by side (dB scale).

    Args:
        patterns_dict: dict of {label: pattern_array}
        title_prefix: prefix for plot title
        save_path: path to save figure
    """
    angles = np.linspace(-180, 180, 361)
    angles_rad = np.deg2rad(angles)
    colors = plt.cm.tab10(np.linspace(0, 1, len(patterns_dict)))

    fig = plt.figure(figsize=(18, 8))

    # E-plane polar plot (left)
    ax1 = fig.add_subplot(121, projection='polar')

    # Find global min and max for E-plane
    all_gains_e = np.concatenate([pattern[:, 0] for pattern in patterns_dict.values()])
    min_db_e = np.floor(all_gains_e.min() / 5) * 5

    for (label, pattern), color in zip(patterns_dict.items(), colors):
        gain_db = pattern[:, 0]
        gain_shifted = gain_db - min_db_e
        ax1.plot(angles_rad, gain_shifted, label=label, linewidth=2, color=color)

    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_title("E-plane (Elevation)", fontsize=14, fontweight='bold', pad=20)

    # Set angular ticks every 30 degrees (in radians for polar plots)
    ax1.set_xticks(np.deg2rad(np.arange(0, 360, 30)))

    # Set radial ticks every 5 dB
    max_db_e = np.ceil(all_gains_e.max() / 5) * 5
    r_range_e = max_db_e - min_db_e
    r_ticks_e = np.arange(0, r_range_e + 5, 5)
    ax1.set_yticks(r_ticks_e)
    ax1.set_yticklabels([f'{tick + min_db_e:.0f}' for tick in r_ticks_e])

    # Position radial labels at 0 degrees (top, between quadrants)
    ax1.set_rlabel_position(0)

    ax1.grid(True, alpha=0.3)

    # H-plane polar plot (right)
    ax2 = fig.add_subplot(122, projection='polar')

    # Find global min and max for H-plane
    all_gains_h = np.concatenate([pattern[:, 1] for pattern in patterns_dict.values()])
    min_db_h = np.floor(all_gains_h.min() / 5) * 5

    for (label, pattern), color in zip(patterns_dict.items(), colors):
        gain_db = pattern[:, 1]
        gain_shifted = gain_db - min_db_h
        ax2.plot(angles_rad, gain_shifted, label=label, linewidth=2, color=color)

    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_title("H-plane (Azimuth)", fontsize=14, fontweight='bold', pad=20)

    # Set angular ticks every 30 degrees (in radians for polar plots)
    ax2.set_xticks(np.deg2rad(np.arange(0, 360, 30)))

    # Set radial ticks every 5 dB
    max_db_h = np.ceil(all_gains_h.max() / 5) * 5
    r_range_h = max_db_h - min_db_h
    r_ticks_h = np.arange(0, r_range_h + 5, 5)
    ax2.set_yticks(r_ticks_h)
    ax2.set_yticklabels([f'{tick + min_db_h:.0f}' for tick in r_ticks_h])

    # Position radial labels at 0 degrees (top, between quadrants)
    ax2.set_rlabel_position(0)

    ax2.grid(True, alpha=0.3)

    # Shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=min(len(patterns_dict), 4), fontsize=11)

    fig.suptitle(f"{title_prefix}Antenna Patterns (Polar - dB Scale)",
                 fontsize=16, fontweight='bold', y=0.92)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def draw_beam_pattern_thumbnail(polar_ax, tx_pattern_path, rx_pattern_path):
    """Draw TX/RX E-plane and H-plane beam patterns on a polar axes.

    Shows full 360° circle with 4 lines: TX E-plane, TX H-plane,
    RX E-plane, RX H-plane.

    Args:
        polar_ax: Matplotlib polar Axes to draw into.
        tx_pattern_path: Path to TX antenna pattern .npy file.
        rx_pattern_path: Path to RX antenna pattern .npy file.
    """
    tx_pat = np.load(tx_pattern_path)
    rx_pat = np.load(rx_pattern_path)

    angles_rad = np.deg2rad(np.linspace(-180, 180, 361))

    # Column 0 = E-plane (elevation), Column 1 = H-plane (azimuth)
    tx_e = tx_pat[:, 0]
    tx_h = tx_pat[:, 1]
    rx_e = rx_pat[:, 0]
    rx_h = rx_pat[:, 1]

    # Global min for dB shift (all 4 traces)
    all_gains = np.concatenate([tx_e, tx_h, rx_e, rx_h])
    min_db = np.floor(all_gains.min() / 5) * 5

    polar_ax.plot(angles_rad, tx_e - min_db, color="#c62828", lw=0.5,
                  label="TX El", alpha=0.85)
    polar_ax.plot(angles_rad, tx_h - min_db, color="#c62828", lw=0.5,
                  label="TX Az", alpha=0.85, ls="--")
    polar_ax.plot(angles_rad, rx_e - min_db, color="#1565c0", lw=0.5,
                  label="RX El", alpha=0.85)
    polar_ax.plot(angles_rad, rx_h - min_db, color="#1565c0", lw=0.5,
                  label="RX Az", alpha=0.85, ls="--")

    polar_ax.set_theta_zero_location("N")
    polar_ax.set_theta_direction(-1)

    # Radial ticks in dB — only show ticks that fit inside the data range
    max_shifted = all_gains.max() - min_db
    r_ticks = np.arange(0, max_shifted, 10)
    polar_ax.set_yticks(r_ticks)
    polar_ax.set_yticklabels([f"{t + min_db:.0f}" for t in r_ticks])
    polar_ax.set_rlim(0, max_shifted)

    polar_ax.set_rlabel_position(0)
    polar_ax.tick_params(axis='y', labelsize=2.5, pad=-1)
    polar_ax.tick_params(axis='x', labelsize=2.5, pad=-7)
    polar_ax.grid(True, lw=0.15, alpha=0.4)

    # Thin border instead of thick black line
    polar_ax.spines["polar"].set_linewidth(0.3)

    # Background color matching innermost tile (#ebebeb)
    polar_ax.set_facecolor("#ebebeb")


def main():
    # Set up paths
    base_path = Path(__file__).resolve().parents[3]
    pattern_path = base_path / "assets/antenna_pattern/MMWCAS"
    output_path = base_path / "notebooks/output"

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("MMWCAS Antenna Pattern Visualization")
    print("="*70)

    # Load RX patterns at 76 GHz
    print("\nLoading RX patterns...")
    rx_patterns = {}
    for rx in ['rx1', 'rx4', 'rx8']:
        pattern_file = pattern_path / f"{rx}_76.npy"
        rx_patterns[rx.upper()] = np.load(pattern_file)
        print(f"  Loaded: {rx}_76.npy")

    # Load TX patterns at multiple frequencies
    print("\nLoading TX patterns...")
    tx_patterns = {}
    freq_files = [
        ('76', 'tx1_76.npy'),
        ('78.5', 'tx1_78_5.npy'),
        ('81', 'tx1_81.npy')
    ]
    for freq, filename in freq_files:
        pattern_file = pattern_path / filename
        tx_patterns[f'TX1 @ {freq} GHz'] = np.load(pattern_file)
        print(f"  Loaded: {filename}")

    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)

    # ========== RX Antenna Plots ==========
    print("\n[1/4] RX Antennas - Line Plots (E & H planes)")
    plot_combined_line_patterns(
        rx_patterns,
        title_prefix="RX ",
        save_path=output_path / "rx_patterns_line.png"
    )

    print("[2/4] RX Antennas - Polar Plots (E & H planes)")
    plot_combined_polar_patterns(
        rx_patterns,
        title_prefix="RX ",
        save_path=output_path / "rx_patterns_polar.png"
    )

    # ========== TX Antenna Plots ==========
    print("[3/4] TX Frequencies - Line Plots (E & H planes)")
    plot_combined_line_patterns(
        tx_patterns,
        title_prefix="TX ",
        save_path=output_path / "tx_patterns_line.png"
    )

    print("[4/4] TX Frequencies - Polar Plots (E & H planes)")
    plot_combined_polar_patterns(
        tx_patterns,
        title_prefix="TX ",
        save_path=output_path / "tx_patterns_polar.png"
    )

    print("\n" + "="*70)
    print("Visualization Complete!")
    print("="*70)
    print(f"All plots saved to: {output_path}")
    print("\nGenerated files:")
    for file in ['rx_patterns_line.png', 'rx_patterns_polar.png',
                 'tx_patterns_line.png', 'tx_patterns_polar.png']:
        filepath = output_path / file
        if filepath.exists():
            print(f"  - {file}")


if __name__ == "__main__":
    main()
