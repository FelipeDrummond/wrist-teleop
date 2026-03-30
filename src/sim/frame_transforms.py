"""Watch frame <-> robot base frame coordinate transforms.

Coordinate frame conventions:

    Watch Frame (CoreMotion):        Robot Base Frame (robosuite):
      +X: towards crown               +X: forward (away from robot base)
      +Y: along band (up arm)         +Y: left
      +Z: out of screen               +Z: up

The nominal transform assumes the operator faces the robot with the watch
on their left wrist, screen facing up:
    Watch +X (crown)       -> Robot -Y (right)
    Watch +Y (up arm)      -> Robot +X (forward)
    Watch +Z (out screen)  -> Robot +Z (up)

Quaternion conventions:
    Apple Watch / CoreMotion: (w, x, y, z) scalar-first
    robosuite / scipy:        (x, y, z, w) scalar-last
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# shape: (3, 3)
NOMINAL_WATCH_TO_BASE: np.ndarray = np.array(
    [
        [0.0, 1.0, 0.0],  # robot X = watch Y
        [-1.0, 0.0, 0.0],  # robot Y = -watch X
        [0.0, 0.0, 1.0],  # robot Z = watch Z
    ],
    dtype=np.float64,
)


def transform_position_delta(
    delta_watch: np.ndarray,
    rotation: np.ndarray = NOMINAL_WATCH_TO_BASE,
) -> np.ndarray:
    """Rotate a position delta from watch frame to robot base frame.

    Args:
        delta_watch: (3,) position delta in watch frame [meters].
        rotation: (3,3) rotation matrix from watch to base frame.

    Returns:
        (3,) position delta in robot base frame.
    """
    return (rotation @ delta_watch).astype(np.float64)


def transform_orientation_quat(
    quat_delta_wxyz: np.ndarray,
    rotation: np.ndarray = NOMINAL_WATCH_TO_BASE,
) -> np.ndarray:
    """Rotate a quaternion delta from watch frame to robot base frame.

    The frame rotation is applied as conjugation: R_base = R_tf * R_watch * R_tf^T

    Args:
        quat_delta_wxyz: (4,) quaternion delta in watch frame, (w,x,y,z).
        rotation: (3,3) rotation matrix from watch to base frame.

    Returns:
        (4,) quaternion delta in robot base frame, (w,x,y,z).
    """
    r_tf = Rotation.from_matrix(rotation)
    r_watch = Rotation.from_quat(quat_wxyz_to_xyzw(quat_delta_wxyz))
    r_base = r_tf * r_watch * r_tf.inv()
    return quat_xyzw_to_wxyz(r_base.as_quat())


def quat_to_rotvec(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a (w,x,y,z) quaternion to axis-angle rotation vector.

    Args:
        quat_wxyz: (4,) quaternion in (w,x,y,z) convention.

    Returns:
        (3,) rotation vector (axis * angle in radians).
    """
    return Rotation.from_quat(quat_wxyz_to_xyzw(quat_wxyz)).as_rotvec().astype(np.float64)


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from (w,x,y,z) to (x,y,z,w)."""
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from (x,y,z,w) to (w,x,y,z)."""
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
