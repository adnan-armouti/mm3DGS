"""
DrJit quaternion utilities for differentiable rigid-body transformations.

Provides quaternion rotation for 6DOF pose optimization of the radar board.
All operations are fully differentiable through DrJit AD.

Convention:
- Quaternion layout: (x, y, z, w) with w = scalar part
- Euler angles: ZYX intrinsic order (yaw -> pitch -> roll)
- Right-hand coordinate system
"""

import drjit as dr
import mitsuba as mi


def euler_to_quaternion(pitch, roll, yaw):
    """
    Convert Euler angles (ZYX intrinsic) to unit quaternion.

    Args:
        pitch: Rotation about X axis (mi.Float scalar)
        roll:  Rotation about Y axis (mi.Float scalar)
        yaw:   Rotation about Z axis (mi.Float scalar)

    Returns:
        Tuple (qx, qy, qz, qw) of mi.Float scalars.
    """
    half_p = pitch * 0.5
    half_r = roll * 0.5
    half_y = yaw * 0.5

    sp = dr.sin(half_p)
    cp = dr.cos(half_p)
    sr = dr.sin(half_r)
    cr = dr.cos(half_r)
    sy = dr.sin(half_y)
    cy = dr.cos(half_y)

    # ZYX composition: q = qz * qy * qx
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * cr * sp - sy * sr * cp
    qy = cy * sr * cp + sy * cr * sp
    qz = sy * cr * cp - cy * sr * sp

    return (qx, qy, qz, qw)


def quaternion_rotate(q, point):
    """
    Rotate a point by a unit quaternion.

    Uses the formula: p' = q * p * q^-1
    Optimized form avoiding full quaternion multiplication.

    Args:
        q: Tuple (qx, qy, qz, qw) of mi.Float
        point: mi.Point3f or mi.Vector3f to rotate

    Returns:
        Rotated mi.Point3f (or mi.Vector3f)
    """
    qx, qy, qz, qw = q

    # Cross product: t = 2 * (q_vec x p)
    tx = mi.Float(2.0) * (qy * point.z - qz * point.y)
    ty = mi.Float(2.0) * (qz * point.x - qx * point.z)
    tz = mi.Float(2.0) * (qx * point.y - qy * point.x)

    # p' = p + qw * t + (q_vec x t)
    rx = point.x + qw * tx + (qy * tz - qz * ty)
    ry = point.y + qw * ty + (qz * tx - qx * tz)
    rz = point.z + qw * tz + (qx * ty - qy * tx)

    return type(point)(rx, ry, rz)


def transform_positions(base_positions, pose_params, translate=True):
    """
    Apply rigid-body transformation to array positions.

    Args:
        base_positions: mi.Point3f [N] original positions (frozen)
        pose_params: dict with keys:
            'pitch', 'roll', 'yaw': mi.Float scalars (radians, grad-enabled)
            'tx', 'ty', 'tz': mi.Float scalars (meters, grad-enabled)
        translate: If True (default), apply both rotation and translation.
            If False, apply rotation only (useful for direction vectors like boresights).

    Returns:
        mi.Point3f [N] transformed positions (live, differentiable)
    """
    # Build rotation quaternion from Euler angles
    q = euler_to_quaternion(
        pose_params['pitch'],
        pose_params['roll'],
        pose_params['yaw'],
    )

    # Rotate positions about the origin
    rotated = quaternion_rotate(q, base_positions)

    if not translate:
        return rotated

    # Translate
    result = mi.Point3f(
        rotated.x + pose_params['tx'],
        rotated.y + pose_params['ty'],
        rotated.z + pose_params['tz'],
    )

    return result


def create_pose_params(pitch=0.0, roll=0.0, yaw=0.0, tx=0.0, ty=0.0, tz=0.0,
                       enable_grad=True):
    """
    Create a pose parameter dictionary with optional gradient tracking.

    Args:
        pitch, roll, yaw: Initial rotation angles in radians
        tx, ty, tz: Initial translations in meters
        enable_grad: Whether to enable DrJit gradient tracking

    Returns:
        dict of mi.Float scalars
    """
    params = {
        'pitch': mi.Float(pitch),
        'roll': mi.Float(roll),
        'yaw': mi.Float(yaw),
        'tx': mi.Float(tx),
        'ty': mi.Float(ty),
        'tz': mi.Float(tz),
    }
    if enable_grad:
        for v in params.values():
            dr.enable_grad(v)
    return params
