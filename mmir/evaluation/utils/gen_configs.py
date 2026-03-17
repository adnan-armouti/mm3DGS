import json
import math
import numpy as np

def _norm(v, eps: float = 1e-9):
    """L2-normalise `v` along its last dimension."""
    return v / (np.linalg.norm(v) + eps)

def _axis_angle_to_matrix(w):
    """
    Rodrigues' formula - converts a 3-vector axis-angle `w` (rad) to a 3x3
    rotation matrix. Works with numpy arrays.
    """
    theta = np.linalg.norm(w)
    if theta < 1e-9:
        return np.eye(3)
    
    k = w / theta  # unit axis
    kx, ky, kz = k[0], k[1], k[2]
    
    # Skew-symmetric matrix
    K = np.array([
        [0, -kz,  ky],
        [kz,  0, -kx],
        [-ky, kx,  0]
    ])
    
    eye = np.eye(3)
    return eye + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def _calibrated_geometry_corrected(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    rx_bor: np.ndarray,
    translation: np.ndarray,
    rot_vec: np.ndarray,
) -> tuple:
    """Lightweight version of calibrated_geometry_xyz adapted for numpy.
    Args
    ----
    translation : (3,) XYZ *in board frame*
    rot_vec     : (3,) axis-angle *global frame* (rad)
    """
    # 1) canonical board basis from the shared boresight ----------------------
    bs0 = rx_bor[0]
    y0 = _norm(bs0)
    z_try = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(y0, z_try)) > 0.99:  # near collinear
        z_try = np.array([0.0, 1.0, 0.0])
    x0 = _norm(np.cross(y0, z_try))
    z0 = _norm(np.cross(x0, y0))

    # 2) rotate the entire board in *global* coords ---------------------------
    R = _axis_angle_to_matrix(rot_vec)
    y_hat = _norm(R @ y0)

    # 3) new orthonormal board frame after rotation --------------------------
    z_try2 = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(y_hat, z_try2)) > 0.99:
        z_try2 = np.array([0.0, 1.0, 0.0])
    x_hat = _norm(np.cross(y_hat, z_try2))
    z_hat = _norm(np.cross(x_hat, y_hat))

    # 4) re-express every antenna in the rotated frame -----------------------
    all_p = np.vstack([rx_pos, tx_pos])
    board_c = np.mean(all_p, axis=0, keepdims=True)
    delta = all_p - board_c
    rel_x = delta @ x0 - np.mean(delta @ x0)
    rel_z = delta @ z0 - np.mean(delta @ z0)

    recon = board_c + rel_x.reshape(-1, 1) * x_hat + rel_z.reshape(-1, 1) * z_hat

    # 5) apply translation in rotated frame ----------------------------------
    recon = recon + translation[0] * x_hat + translation[1] * y_hat + translation[2] * z_hat

    # 6) split back into RX/TX ----------------------------------------------
    n_rx = rx_pos.shape[0]
    rx_new, tx_new = recon[:n_rx], recon[n_rx:]
    rx_bor_new = np.tile(y_hat, (n_rx, 1))
    return tx_new, rx_new, rx_bor_new

def generate_azimuth_rotated_config(base_config: dict, azimuth_angle_deg: float) -> dict:
    """
    Generate a new radar config rotated by the specified azimuth angle.
    Uses drift-free transformation logic from renderer_core.py to maintain
    the board center fixed in world coordinates.
    
    Args:
        base_config: Original radar config dict
        azimuth_angle_deg: Azimuth rotation angle in degrees (positive/negative)
    
    Returns:
        New config dict with rotated antenna positions and boresights
    """
    # Convert angle to radians
    azimuth_angle_rad = math.radians(azimuth_angle_deg)
    
    # Create a deep copy of the base config
    rotated_config = json.loads(json.dumps(base_config))
    
    # Extract TX and RX positions and boresights
    tx_positions = []
    rx_positions = []
    rx_boresights = []
    
    # Collect TX positions (convert from mm to m)
    for tx in base_config["tx_array"]:
        pos_mm = np.array(tx["pos_mm"])
        pos_m = pos_mm / 1000.0  # Convert to meters
        tx_positions.append(pos_m)
    
    # Collect RX positions and boresights (convert from mm to m)
    for rx in base_config["rx_array"]:
        pos_mm = np.array(rx["pos_mm"])
        pos_m = pos_mm / 1000.0  # Convert to meters
        rx_positions.append(pos_m)
        
        boresight = np.array(rx["boresight"])
        rx_boresights.append(boresight)
    
    # Convert to numpy arrays
    tx_pos = np.array(tx_positions)  # Shape: (n_tx, 3)
    rx_pos = np.array(rx_positions)  # Shape: (n_rx, 3)
    rx_bor = np.array(rx_boresights)  # Shape: (n_rx, 3)
    
    # Apply drift-free transformation
    # rot_vec is [0, 0, azimuth_angle_rad] for Z-axis rotation
    rot_vec = np.array([0.0, 0.0, azimuth_angle_rad])
    translation = np.array([0.0, 0.0, 0.0])  # No translation, just rotation
    
    tx_new, rx_new, rx_bor_new = _calibrated_geometry_corrected(
        tx_pos, rx_pos, rx_bor, translation, rot_vec
    )
    
    # Update TX array positions and boresights
    for i, tx in enumerate(rotated_config["tx_array"]):
        # Convert back to mm
        pos_mm_new = (tx_new[i] * 1000.0).tolist()
        tx["pos_mm"] = pos_mm_new
        
        # Note: TX boresights remain unchanged in this implementation
        # as they are typically aligned with the board normal
    
    # Update RX array positions and boresights
    for i, rx in enumerate(rotated_config["rx_array"]):
        # Convert back to mm
        pos_mm_new = (rx_new[i] * 1000.0).tolist()
        rx["pos_mm"] = pos_mm_new
        
        # Update boresight
        boresight_new = rx_bor_new[i].tolist()
        rx["boresight"] = boresight_new
    
    return rotated_config