"""Unit tests for coordinate frame transforms.

Tests use known input/output pairs for axis-by-axis verification
as required by PROJECT.md testing strategy.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.sim.frame_transforms import (
    NOMINAL_WATCH_TO_BASE,
    transform_position_delta,
    transform_orientation_quat,
    quat_to_rotvec,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
)


class TestNominalRotationMatrix:
    def test_is_proper_rotation(self):
        det = np.linalg.det(NOMINAL_WATCH_TO_BASE)
        np.testing.assert_allclose(det, 1.0, atol=1e-10)

    def test_is_orthogonal(self):
        product = NOMINAL_WATCH_TO_BASE @ NOMINAL_WATCH_TO_BASE.T
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)


class TestPositionDeltaTransform:
    """Axis-by-axis tests for watch → robot base position mapping."""

    def test_watch_x_maps_to_robot_neg_y(self):
        delta = np.array([1.0, 0.0, 0.0])
        result = transform_position_delta(delta)
        np.testing.assert_allclose(result, [0.0, -1.0, 0.0], atol=1e-10)

    def test_watch_y_maps_to_robot_x(self):
        delta = np.array([0.0, 1.0, 0.0])
        result = transform_position_delta(delta)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-10)

    def test_watch_z_maps_to_robot_z(self):
        delta = np.array([0.0, 0.0, 1.0])
        result = transform_position_delta(delta)
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-10)

    def test_zero_delta_stays_zero(self):
        delta = np.array([0.0, 0.0, 0.0])
        result = transform_position_delta(delta)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-10)

    def test_combined_delta(self):
        delta = np.array([1.0, 2.0, 3.0])
        result = transform_position_delta(delta)
        # robot X = watch Y = 2.0, robot Y = -watch X = -1.0, robot Z = watch Z = 3.0
        np.testing.assert_allclose(result, [2.0, -1.0, 3.0], atol=1e-10)

    def test_custom_rotation(self):
        identity = np.eye(3)
        delta = np.array([1.0, 0.0, 0.0])
        result = transform_position_delta(delta, rotation=identity)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-10)

    def test_output_dtype_is_float64(self):
        delta = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = transform_position_delta(delta)
        assert result.dtype == np.float64


class TestOrientationQuatTransform:
    """Tests for quaternion frame rotation."""

    def test_identity_quat_stays_identity(self):
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])  # (w,x,y,z)
        result = transform_orientation_quat(q_identity)
        # Identity rotation should remain identity regardless of frame
        np.testing.assert_allclose(np.abs(result), [1.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_identity_rotation_matrix_preserves_quat(self):
        # 90-degree rotation about watch Z
        q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])  # (w,x,y,z)
        result = transform_orientation_quat(q, rotation=np.eye(3))
        np.testing.assert_allclose(result, q, atol=1e-10)

    def test_watch_z_rotation_maps_to_robot_z_rotation(self):
        # Watch Z = Robot Z, so a rotation about watch Z should map to
        # a rotation about robot Z
        angle = np.pi / 6  # 30 degrees
        q_watch = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        result = transform_orientation_quat(q_watch)
        rotvec = quat_to_rotvec(result)
        # Should be primarily about robot Z axis
        assert abs(rotvec[2]) > abs(rotvec[0]) + abs(rotvec[1])

    def test_output_is_unit_quaternion(self):
        q = np.array([np.cos(0.3), np.sin(0.3), 0.0, 0.0])
        result = transform_orientation_quat(q)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-10)


class TestQuatConversions:
    def test_wxyz_to_xyzw(self):
        q = np.array([1.0, 2.0, 3.0, 4.0])  # w,x,y,z
        result = quat_wxyz_to_xyzw(q)
        np.testing.assert_array_equal(result, [2.0, 3.0, 4.0, 1.0])

    def test_xyzw_to_wxyz(self):
        q = np.array([2.0, 3.0, 4.0, 1.0])  # x,y,z,w
        result = quat_xyzw_to_wxyz(q)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])

    def test_roundtrip_wxyz(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_array_equal(quat_xyzw_to_wxyz(quat_wxyz_to_xyzw(q)), q)

    def test_roundtrip_xyzw(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_array_equal(quat_wxyz_to_xyzw(quat_xyzw_to_wxyz(q)), q)

    def test_output_dtype_is_float64(self):
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert quat_wxyz_to_xyzw(q).dtype == np.float64
        assert quat_xyzw_to_wxyz(q).dtype == np.float64


class TestQuatToRotvec:
    def test_identity_gives_zero_rotvec(self):
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        result = quat_to_rotvec(q_identity)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-10)

    def test_90deg_about_z(self):
        angle = np.pi / 2
        q = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])  # (w,x,y,z)
        result = quat_to_rotvec(q)
        np.testing.assert_allclose(result, [0.0, 0.0, np.pi / 2], atol=1e-10)

    def test_90deg_about_x(self):
        angle = np.pi / 2
        q = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        result = quat_to_rotvec(q)
        np.testing.assert_allclose(result, [np.pi / 2, 0.0, 0.0], atol=1e-10)

    def test_small_rotation(self):
        angle = 0.01  # ~0.57 degrees
        q = np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])
        result = quat_to_rotvec(q)
        np.testing.assert_allclose(result, [0.0, angle, 0.0], atol=1e-6)

    def test_180deg_about_y(self):
        q = np.array([0.0, 0.0, 1.0, 0.0])  # (w,x,y,z) = 180 about Y
        result = quat_to_rotvec(q)
        np.testing.assert_allclose(np.linalg.norm(result), np.pi, atol=1e-10)
        # Direction should be Y-axis
        np.testing.assert_allclose(
            np.abs(result / np.linalg.norm(result)), [0.0, 1.0, 0.0], atol=1e-10
        )

    def test_output_dtype_is_float64(self):
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert quat_to_rotvec(q).dtype == np.float64
