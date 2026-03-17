#!/usr/bin/env python3
"""
Generate a uniform virtual antenna array (using the provided function EXACTLY),
plot the 2D layout, map TX/RX onto the old board plane (orthogonal to boresight)
with HALF-WAVELENGTH spacing, translate the new board so its geometric center
matches the old board's center, export an updated config JSON with new TX/RX
elements (keeping all other settings), and visualize old vs new configs in 3D.
"""

import argparse
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import trimesh

# ---------------------------
# EXACT function (do not modify)
# ---------------------------
def generate_uniform_virtual_antennas(num_antennas, spacing=1.0):
    """
    Generate uniform virtual antenna positions based on a single antenna count.
    
    Args:
        num_antennas: Total number of transmitters and receivers (must be even)
                     num_tx = num_rx = num_antennas
        spacing: Spacing between virtual antenna positions (default: 1.0)
        
    Returns:
        tuple: (virtual_antennas, tx_locations, rx_locations)
            - virtual_antennas: set of (x, y) tuples
            - tx_locations: list of (x, y) tuples for transmitters
            - rx_locations: list of (x, y) tuples for receivers
            
    Raises:
        AssertionError: If num_antennas is not even
    """
    
    # Step 1: Assert that input is an even number
    assert num_antennas % 2 == 0, f"Number of antennas ({num_antennas}) must be an even number"
    
    # Set num_tx = num_rx = num_antennas
    total_tx = num_antennas
    total_rx = num_antennas
    
    # Step 2: Divide by 2 (receivers form two rows and transmitters form two columns)
    tx_per_column = total_tx // 2
    rx_per_row = total_rx // 2
    
    # Step 3: Establish transmitter and receiver locations
    tx_locations = []
    rx_locations = []
    
    # Transmitter positions (two columns: left and right)
    # Left column: x=0.5, y from 1 to tx_per_column
    # Right column: x=tx_per_column+0.5, y from 1 to tx_per_column
    for y in (np.arange(-(tx_per_column-1)/2, (tx_per_column)/2, 1)):
        # Left column
        tx_locations.append((-(tx_per_column/2), y))
        # Right column (positioned to create uniform spacing)
        tx_locations.append(((tx_per_column/2), y))
    
    # Receiver positions (two rows: bottom and top)
    # Bottom row: y=0, x from 0.5 to rx_per_row-0.5
    # Top row: y=rx_per_row, x from 0.5 to rx_per_row-0.5
    for x in (np.arange(-(rx_per_row-1)/2, (rx_per_row)/2, 1)):
        # Bottom row
        rx_locations.append((x, -(tx_per_column/2)))
        # Top row (positioned to create uniform spacing)
        rx_locations.append((x, (tx_per_column/2)))
    
    # Step 4: Map to uniformly spaced virtual antenna positions
    virtual_antennas = []
    
    for rx_loc in rx_locations:
        for tx_loc in tx_locations:
            vx_x = rx_loc[0] + tx_loc[0]
            vx_y = rx_loc[1] + tx_loc[1]
            virtual_antennas.append((vx_x, vx_y))
    
    return np.array(virtual_antennas), np.array(tx_locations), np.array(rx_locations)

# ---------------------------
# Utilities
# ---------------------------

def calculate_wavelength(carrier_frequency_hz: float) -> float:
    c = 299792458.0
    return c / carrier_frequency_hz

def load_config(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def get_geometric_center_and_boresight(cfg: dict):
    tx = np.array([t['pos_mm'] for t in cfg['tx_array']]) / 1000.0
    rx = np.array([r['pos_mm'] for r in cfg['rx_array']]) / 1000.0
    all_pos = np.vstack([tx, rx])
    center = np.mean(all_pos, axis=0)
    # Assume common boresight for all
    boresight = np.array(cfg['tx_array'][0]['boresight'], dtype=float)
    boresight /= np.linalg.norm(boresight)
    return center, boresight

def plane_basis_from_normal(n: np.ndarray):
    # Find a vector least aligned with n
    axes = [np.array([1.0,0.0,0.0]), np.array([0.0,1.0,0.0]), np.array([0.0,0.0,1.0])]
    dots = [abs(np.dot(n, a)) for a in axes]
    v1 = axes[int(np.argmin(dots))]
    # Project to plane and normalize
    v1 = v1 - np.dot(v1, n) * n
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(n, v1)
    v2 /= np.linalg.norm(v2)
    return v1, v2

def map_2d_to_plane(points_2d, center3d_m: np.ndarray, basis_v1: np.ndarray, basis_v2: np.ndarray, scale_m: float):
    mapped = []
    for (x, y) in points_2d:
        # Fix the orientation: swap basis vectors to preserve intended layout
        # TX column (vertical) should map to Y direction, RX row (horizontal) should map to X direction
        mapped.append(center3d_m + scale_m * (y * basis_v1 + x * basis_v2))
    return np.array(mapped)

# ---------------------------
# Callable API
# ---------------------------

def generate_dense_config(cascaded_config_path: str, output_path: str,
                          num_antennas: int = 100, verbose: bool = True) -> str:
    """Generate a dense 100×100 virtual array config from a cascaded config.

    Takes the cascaded config's board plane (center + boresight), generates a
    uniform TX/RX layout with half-wavelength spacing, and writes a new JSON
    config with the dense array but all other settings (FMCW params, etc.) preserved.

    Args:
        cascaded_config_path: Path to the input cascaded config JSON.
        output_path: Path where the dense config JSON will be written.
        num_antennas: Number of TX and RX elements (default 100 each).
        verbose: Print progress messages.

    Returns:
        The output_path (for chaining).
    """
    cfg = load_config(cascaded_config_path)
    center_m, boresight_n = get_geometric_center_and_boresight(cfg)
    v1, v2 = plane_basis_from_normal(boresight_n)

    wavelength_m = calculate_wavelength(cfg['carrierFrequency'])
    half_wavelength_m = 0.5 * wavelength_m

    _, tx_locations, rx_locations = generate_uniform_virtual_antennas(num_antennas)

    tx_3d = map_2d_to_plane(tx_locations, center_m, v1, v2, half_wavelength_m)
    rx_3d = map_2d_to_plane(rx_locations, center_m, v1, v2, half_wavelength_m)

    # Translate so new board center matches old
    new_center = np.mean(np.vstack([tx_3d, rx_3d]), axis=0)
    delta = center_m - new_center
    tx_3d = tx_3d + delta
    rx_3d = rx_3d + delta

    # Build output config (deep copy, replace arrays)
    new_cfg = json.loads(json.dumps(cfg))

    new_cfg['tx_array'] = [
        {
            "name": f"TX{i+1}",
            "pos_mm": (pos_m * 1000.0).tolist(),
            "boresight": boresight_n.tolist(),
            "radius": 1.0,
            "polarization": "V",
        }
        for i, pos_m in enumerate(tx_3d)
    ]
    new_cfg['rx_array'] = [
        {
            "name": f"RX{i+1}",
            "pos_mm": (pos_m * 1000.0).tolist(),
            "boresight": boresight_n.tolist(),
            "power_dBm": 10.0,
            "polarization": "V",
        }
        for i, pos_m in enumerate(rx_3d)
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(new_cfg, f, indent=2)

    if verbose:
        print(f"[generate_dense_config] Saved dense {num_antennas}×{num_antennas} config")
        print(f"  Input: {cascaded_config_path}")
        print(f"  Output: {output_path}")
        print(f"  λ/2 spacing: {half_wavelength_m*1000:.3f} mm")

    return output_path


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate dense antenna array config from sparse config"
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to input config file (e.g., .../configs/cascaded_frame_135.json)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable remote visualization via SSH tunnel"
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="Path to mesh file (.ply or .obj) for visualization context"
    )
    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")

    # Extract frame index from filename
    filename = os.path.basename(args.config_path)
    match = re.search(r'_frame_(\d+)\.json$', filename)
    if not match:
        raise ValueError(
            f"Input filename must end with '_frame_{{idx}}.json', got: {filename}"
        )
    frame_idx = match.group(1)

    # Generate output path in the same directory
    config_dir = os.path.dirname(args.config_path)
    output_filename = f"dense_frame_{frame_idx}.json"
    output_path = os.path.join(config_dir, output_filename)

    print(f"Input config: {args.config_path}")
    print(f"Output config: {output_path}")
    print(f"Frame index: {frame_idx}")

    # 1) Generate virtual antenna array (units: wavelengths)
    num_antennas = 100
    virtual_antennas, tx_locations, rx_locations = generate_uniform_virtual_antennas(num_antennas)

    # 2) Print and plot the transmitter and receiver positions in 2D
    tx_x_coords = [pos[0] for pos in tx_locations]
    tx_y_coords = [pos[1] for pos in tx_locations]
    rx_x_coords = [pos[0] for pos in rx_locations]
    rx_y_coords = [pos[1] for pos in rx_locations]
    
    print(f"TX positions: {tx_locations}")
    print(f"RX positions: {rx_locations}")
    print(f"virtual_antennas: {virtual_antennas}")

    # Plot TX and RX positions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot 1: Transmitter and Receiver positions
    ax1.scatter(tx_x_coords, tx_y_coords, s=150, c='red', alpha=0.8, edgecolors='black', label='Transmitters', marker='^')
    ax1.scatter(rx_x_coords, rx_y_coords, s=150, c='blue', alpha=0.8, edgecolors='black', label='Receivers', marker='s')
    ax1.set_xlabel('X Position', fontsize=30)
    ax1.set_ylabel('Y Position', fontsize=30)
    ax1.set_title('a) Transmitter and Receiver Positions', fontsize=30)
    ax1.tick_params(axis='both', which='major', labelsize=30)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.legend(fontsize=20)

    # Set limits for TX/RX plot
    all_x = tx_x_coords + rx_x_coords
    all_y = tx_y_coords + rx_y_coords
    max_coord_txrx = max(max(all_x), max(all_y))
    # ax1.set_xlim(-2, max_coord_txrx + 2)
    # ax1.set_ylim(-2, max_coord_txrx + 2)
    # ax1.set_xticks(np.arange(-2, max_coord_txrx + 3, 2))
    # ax1.set_yticks(np.arange(-2, max_coord_txrx + 3, 2))

    # Plot 2: Virtual antenna array layout
    vx_x_coords = [pos[0] for pos in virtual_antennas]
    vx_y_coords = [pos[1] for pos in virtual_antennas]

    ax2.scatter(vx_x_coords, vx_y_coords, s=100, c='green', alpha=0.7, edgecolors='black')
    ax2.set_xlabel('X Position', fontsize=30)
    ax2.set_ylabel('Y Position', fontsize=30)
    # ax2.set_title('Virtual Antenna Array Layout', fontsize=30)
    ax2.tick_params(axis='both', which='major', labelsize=30)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    max_coord = max(max(vx_x_coords), max(vx_y_coords))
    # ax2.set_xlim(-5, max_coord + 5)
    # ax2.set_ylim(-5, max_coord + 5)
    # ax2.set_xticks(np.arange(-5, max_coord + 6, 5))
    # ax2.set_yticks(np.arange(-5, max_coord + 6, 5))
    
    plt.tight_layout()
    plot_output_path = os.path.join(config_dir, f'tx_rx_and_virtual_array_layout_frame_{frame_idx}.png')
    plt.savefig(plot_output_path, dpi=300)
    print(f"Saved 2D layout plot: {plot_output_path}")
    plt.close()  # Close to avoid display issues in headless environments

    # 3) Map this antenna array layout to the same board plane as the old config
    cfg = load_config(args.config_path)
    center_m, boresight_n = get_geometric_center_and_boresight(cfg)
    v1, v2 = plane_basis_from_normal(boresight_n)

    wavelength_m = calculate_wavelength(cfg['carrierFrequency'])
    half_wavelength_m = 0.5 * wavelength_m
    print(f"Using half-wavelength spacing: {half_wavelength_m*1000:.3f} mm")

    # 3a) Create physical space plot (in mm)
    # Convert wavelength units to mm
    tx_x_coords_mm = [pos[0] * half_wavelength_m * 1000 for pos in tx_locations]
    tx_y_coords_mm = [pos[1] * half_wavelength_m * 1000 for pos in tx_locations]
    rx_x_coords_mm = [pos[0] * half_wavelength_m * 1000 for pos in rx_locations]
    rx_y_coords_mm = [pos[1] * half_wavelength_m * 1000 for pos in rx_locations]
    vx_x_coords_mm = [pos[0] * half_wavelength_m * 1000 for pos in virtual_antennas]
    vx_y_coords_mm = [pos[1] * half_wavelength_m * 1000 for pos in virtual_antennas]

    fig_mm, (ax1_mm, ax2_mm) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot 1: Transmitter and Receiver positions in mm
    ax1_mm.scatter(tx_x_coords_mm, tx_y_coords_mm, s=150, c='red', alpha=0.8, edgecolors='black', label='Transmitters', marker='^')
    ax1_mm.scatter(rx_x_coords_mm, rx_y_coords_mm, s=150, c='blue', alpha=0.8, edgecolors='black', label='Receivers', marker='s')
    ax1_mm.set_xlabel('X Position (mm)', fontsize=20)
    ax1_mm.set_ylabel('Y Position (mm)', fontsize=20)
    # ax1_mm.set_title(f'Transmitter and Receiver Positions (Physical Space)\n(Two columns of TX, Two rows of RX) - lambda/2 = {half_wavelength_m*1000:.3f} mm', fontsize=20)
    ax1_mm.tick_params(axis='both', which='major', labelsize=20)
    ax1_mm.grid(True, alpha=0.3)
    ax1_mm.set_aspect('equal')
    ax1_mm.legend(fontsize=20)

    # Plot 2: Virtual antenna array layout in mm
    ax2_mm.scatter(vx_x_coords_mm, vx_y_coords_mm, s=100, c='green', alpha=0.7, edgecolors='black')
    ax2_mm.set_xlabel('X Position (mm)', fontsize=20)
    ax2_mm.set_ylabel('Y Position (mm)', fontsize=20)
    # ax2_mm.set_title(f'Virtual Antenna Array Layout (Physical Space)\n(Uniform grid spacing) - lambda/2 = {half_wavelength_m*1000:.3f} mm', fontsize=20)
    ax2_mm.tick_params(axis='both', which='major', labelsize=20)
    ax2_mm.grid(True, alpha=0.3)
    ax2_mm.set_aspect('equal')

    plt.tight_layout()
    plot_output_path_mm = os.path.join(config_dir, f'tx_rx_and_virtual_array_layout_frame_{frame_idx}_physical_mm.png')
    plt.savefig(plot_output_path_mm, dpi=200)
    print(f"Saved 2D layout plot (physical space): {plot_output_path_mm}")
    plt.close()

    # tx_locations and rx_locations are in wavelengths; map to meters on board plane (lambda/2 spacing)
    tx_3d = map_2d_to_plane(tx_locations, center_m, v1, v2, half_wavelength_m)
    rx_3d = map_2d_to_plane(rx_locations, center_m, v1, v2, half_wavelength_m)

    # Compute new board's current geometric center (before translation)
    new_center = np.mean(np.vstack([tx_3d, rx_3d]), axis=0)
    delta = center_m - new_center
    # Translate all new antennas collectively so centers match exactly
    tx_3d = tx_3d + delta
    rx_3d = rx_3d + delta

    # Confirm orthogonality and centering
    tx_dot = np.abs(((tx_3d - center_m) @ boresight_n)).max()
    rx_dot = np.abs(((rx_3d - center_m) @ boresight_n)).max()
    print(f"Max |(TX - center)*n| after centering: {tx_dot:.3e}")
    print(f"Max |(RX - center)*n| after centering: {rx_dot:.3e}")
    print(f"Old center (m): {center_m}")
    print(f"New center (m): {np.mean(np.vstack([tx_3d, rx_3d]), axis=0)}")

    # 3b) Export an updated config file with new TX/RX sets
    new_cfg = json.loads(json.dumps(cfg))  # copy

    # Build new tx_array with common boresight
    new_tx_array = []
    for i, pos_m in enumerate(tx_3d):
        new_tx_array.append({
            "name": f"TX{i+1}",
            "pos_mm": (pos_m * 1000.0).tolist(),
            "boresight": boresight_n.tolist(),
            "radius": 1.0,
            "polarization": "V"
        })

    # Build new rx_array with common boresight
    new_rx_array = []
    for i, pos_m in enumerate(rx_3d):
        new_rx_array.append({
            "name": f"RX{i+1}",
            "pos_mm": (pos_m * 1000.0).tolist(),
            "boresight": boresight_n.tolist(),
            "power_dBm": 10.0,
            "polarization": "V"
        })

    new_cfg['tx_array'] = new_tx_array
    new_cfg['rx_array'] = new_rx_array

    with open(output_path, 'w') as f:
        json.dump(new_cfg, f, indent=2)
    print(f"Saved updated config with centered board and lambda/2 spacing: {output_path}")

    # 4) Visualize old and new config in 3D using trimesh along with mesh for context
    if args.visualize:
        import sys
        # sys.path resolved via package install
        from mmir.remote_viz import show_scene

        # Determine mesh path with the following priority:
        # 1. User-provided --mesh argument
        # 2. mesh.ply in parent directory of config
        # 3. mesh.obj in parent directory of config
        # 4. No mesh (create empty scene)
        mesh_path = None
        if args.mesh:
            # User provided explicit mesh path
            if os.path.exists(args.mesh):
                mesh_path = args.mesh
            else:
                print(f"Warning: User-provided mesh file not found: {args.mesh}")
        else:
            # Try to find mesh automatically in parent directory
            for mesh_filename in ['mesh.ply', 'mesh.obj']:
                candidate_path = os.path.join(config_dir, '..', mesh_filename)
                if os.path.exists(candidate_path):
                    mesh_path = candidate_path
                    break

        if mesh_path:
            print(f"Loading mesh from: {mesh_path}")
            mesh = trimesh.load(mesh_path)
            scene = mesh.scene()
        else:
            print(f"Warning: No mesh file found, creating scene without mesh")
            scene = trimesh.Scene()

        # Helpers to add spheres
        def create_colored_sphere(center, radius, color):
            sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
            sphere.visual.vertex_colors = color
            sphere.apply_translation(center)
            return sphere

        # Load old config positions (meters)
        old_tx_m = np.array([t['pos_mm'] for t in cfg['tx_array']]) / 1000.0
        old_rx_m = np.array([r['pos_mm'] for r in cfg['rx_array']]) / 1000.0

        # Add old RX (blue) and TX (red)
        for pos in old_rx_m:
            scene.add_geometry(create_colored_sphere(pos, radius=0.01, color=[0, 0, 255, 255]))
        for pos in old_tx_m:
            scene.add_geometry(create_colored_sphere(pos, radius=0.01, color=[255, 0, 0, 255]))

        # Add new RX (green) and TX (orange)
        for pos in rx_3d:
            scene.add_geometry(create_colored_sphere(pos, radius=0.006, color=[0, 255, 0, 255]))
        for pos in tx_3d:
            scene.add_geometry(create_colored_sphere(pos, radius=0.006, color=[255, 165, 0, 255]))

        # Add axes at board center
        origin = center_m
        axis_len = 0.5
        def create_axis_arrow(start, end, color):
            path = trimesh.load_path(np.array([start, end]))
            path.colors = np.tile(np.array(color + [255]), (len(path.entities), 1))
            return path
        scene.add_geometry(create_axis_arrow(origin, origin + axis_len*np.array([1,0,0]), [255,0,0]))
        scene.add_geometry(create_axis_arrow(origin, origin + axis_len*np.array([0,1,0]), [0,255,0]))
        scene.add_geometry(create_axis_arrow(origin, origin + axis_len*np.array([0,0,1]), [0,0,255]))

        # Show scene using remote visualization
        print("Showing 3D scene via remote visualization. Legend: Red=Old TX, Blue=Old RX, Orange=New TX, Green=New RX")
        show_scene(scene)
    else:
        print("Visualization disabled. Use --visualize flag to enable remote visualization.")
