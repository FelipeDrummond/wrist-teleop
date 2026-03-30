"""Acceptance tests for WAT-37: EE command interface + coordinate frame alignment.

Tests:
  1. Position deltas in world frame correctly translate to robot motion
  2. Orientation quaternion commands produce correct EE rotation
  3. Workspace clamping prevents out-of-range commands
  4. Gain multipliers allow per-axis tuning
  5. Config loading and EECommand factory
"""

from __future__ import annotations

import numpy as np
import pytest

from src.sim.env_setup import EnvConfig, make_env
from src.sim.ee_interface import EECommand, EECommandConfig, EECommandInterface


@pytest.fixture(scope="module")
def env():
    e = make_env(config=EnvConfig(render_during_teleop=False), seed=42)
    yield e
    e.close()


@pytest.fixture(scope="module")
def interface(env):
    config = EECommandConfig(
        frame_rotation=np.eye(3),  # identity: skip frame transform for directional tests
    )
    return EECommandInterface(env, config=config)


def _apply_command(interface, command, steps=15):
    """Apply a command for N steps and return first and last obs."""
    obs_first = interface.reset()
    obs_last = obs_first
    for _ in range(steps):
        obs_last, _, done, _ = interface.step(command)
        if done:
            break
    return obs_first, obs_last


class TestPositionDeltasTranslateCorrectly:
    """AC-1: Position deltas in world frame correctly translate to robot motion."""

    def test_positive_x_moves_ee_in_x(self, interface):
        cmd = EECommand(
            position_delta=np.array([0.05, 0.0, 0.0]),
            orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper=0.0,
        )
        obs_first, obs_last = _apply_command(interface, cmd)
        dx = obs_last["robot0_eef_pos"][0] - obs_first["robot0_eef_pos"][0]
        assert dx > 1e-4, f"EE did not move in +X: dx={dx}"

    def test_positive_y_moves_ee_in_y(self, interface):
        cmd = EECommand(
            position_delta=np.array([0.0, 0.05, 0.0]),
            orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper=0.0,
        )
        obs_first, obs_last = _apply_command(interface, cmd)
        dy = obs_last["robot0_eef_pos"][1] - obs_first["robot0_eef_pos"][1]
        assert dy > 1e-4, f"EE did not move in +Y: dy={dy}"

    def test_positive_z_moves_ee_in_z(self, interface):
        cmd = EECommand(
            position_delta=np.array([0.0, 0.0, 0.05]),
            orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper=0.0,
        )
        obs_first, obs_last = _apply_command(interface, cmd)
        dz = obs_last["robot0_eef_pos"][2] - obs_first["robot0_eef_pos"][2]
        assert dz > 1e-4, f"EE did not move in +Z: dz={dz}"

    def test_zero_command_minimal_movement(self, interface):
        cmd = EECommand.zero()
        obs_first, obs_last = _apply_command(interface, cmd, steps=5)
        disp = np.linalg.norm(
            obs_last["robot0_eef_pos"] - obs_first["robot0_eef_pos"]
        )
        # OSC controller has settling dynamics; allow small drift
        assert disp < 0.08, f"EE moved too much on zero command: {disp}"


class TestOrientationCommands:
    """AC-2: Orientation quaternion commands produce correct EE rotation."""

    def test_rotation_about_z_changes_orientation(self, interface):
        angle = 0.3  # ~17 degrees
        cmd = EECommand(
            position_delta=np.zeros(3),
            orientation_quat=np.array(
                [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
            ),
            gripper=0.0,
        )
        obs_first, obs_last = _apply_command(interface, cmd, steps=20)
        quat_diff = np.linalg.norm(
            obs_last["robot0_eef_quat"] - obs_first["robot0_eef_quat"]
        )
        assert quat_diff > 1e-3, f"Orientation did not change: diff={quat_diff}"

    def test_identity_quaternion_minimal_rotation(self, interface):
        cmd = EECommand(
            position_delta=np.zeros(3),
            orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper=0.0,
        )
        obs_first, obs_last = _apply_command(interface, cmd, steps=5)
        quat_diff = np.linalg.norm(
            obs_last["robot0_eef_quat"] - obs_first["robot0_eef_quat"]
        )
        assert quat_diff < 0.05, f"Orientation changed too much: diff={quat_diff}"


class TestWorkspaceClamping:
    """AC-3: Workspace clamping prevents out-of-range commands."""

    def test_ee_stays_within_upper_bounds(self, interface):
        bounds = interface._config.workspace_bounds
        cmd = EECommand(
            position_delta=np.array([0.05, 0.05, 0.05]),
            orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper=0.0,
        )
        interface.reset()
        for _ in range(50):
            obs, _, done, _ = interface.step(cmd)
            if done:
                interface.reset()
        pos = obs["robot0_eef_pos"]
        assert pos[0] <= bounds["x"][1] + 0.02, f"X exceeded upper bound: {pos[0]}"
        assert pos[1] <= bounds["y"][1] + 0.02, f"Y exceeded upper bound: {pos[1]}"
        assert pos[2] <= bounds["z"][1] + 0.02, f"Z exceeded upper bound: {pos[2]}"

    def test_ee_stays_within_lower_bounds(self, interface):
        bounds = interface._config.workspace_bounds
        cmd = EECommand(
            position_delta=np.array([-0.05, -0.05, -0.05]),
            orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper=0.0,
        )
        interface.reset()
        for _ in range(50):
            obs, _, done, _ = interface.step(cmd)
            if done:
                interface.reset()
        pos = obs["robot0_eef_pos"]
        assert pos[0] >= bounds["x"][0] - 0.02, f"X below lower bound: {pos[0]}"
        assert pos[1] >= bounds["y"][0] - 0.02, f"Y below lower bound: {pos[1]}"
        assert pos[2] >= bounds["z"][0] - 0.02, f"Z below lower bound: {pos[2]}"


class TestGainMultipliers:
    """AC-4: Gain multipliers allow per-axis tuning."""

    def test_higher_gain_produces_larger_displacement(self):
        env_lo = make_env(config=EnvConfig(render_during_teleop=False), seed=99)
        env_hi = make_env(config=EnvConfig(render_during_teleop=False), seed=99)
        try:
            iface_lo = EECommandInterface(
                env_lo,
                EECommandConfig(
                    position_gains=np.array([0.3, 0.3, 0.3]),
                    frame_rotation=np.eye(3),
                ),
            )
            iface_hi = EECommandInterface(
                env_hi,
                EECommandConfig(
                    position_gains=np.array([1.0, 1.0, 1.0]),
                    frame_rotation=np.eye(3),
                ),
            )

            cmd = EECommand(
                position_delta=np.array([0.05, 0.0, 0.0]),
                orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                gripper=0.0,
            )

            obs_lo_first, obs_lo_last = _apply_command(iface_lo, cmd, steps=10)
            obs_hi_first, obs_hi_last = _apply_command(iface_hi, cmd, steps=10)

            disp_lo = np.linalg.norm(
                obs_lo_last["robot0_eef_pos"] - obs_lo_first["robot0_eef_pos"]
            )
            disp_hi = np.linalg.norm(
                obs_hi_last["robot0_eef_pos"] - obs_hi_first["robot0_eef_pos"]
            )
            assert disp_hi > disp_lo, (
                f"Higher gain should produce larger displacement: hi={disp_hi}, lo={disp_lo}"
            )
        finally:
            env_lo.close()
            env_hi.close()

    def test_zero_gain_suppresses_axis(self):
        env = make_env(config=EnvConfig(render_during_teleop=False), seed=77)
        try:
            iface = EECommandInterface(
                env,
                EECommandConfig(
                    position_gains=np.array([0.0, 1.0, 1.0]),
                    frame_rotation=np.eye(3),
                ),
            )
            cmd = EECommand(
                position_delta=np.array([0.05, 0.0, 0.0]),
                orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                gripper=0.0,
            )
            obs_first, obs_last = _apply_command(iface, cmd, steps=10)
            dx = abs(
                obs_last["robot0_eef_pos"][0] - obs_first["robot0_eef_pos"][0]
            )
            assert dx < 0.005, f"X axis should be suppressed with zero gain: dx={dx}"
        finally:
            env.close()


class TestConfigAndFactory:
    """Config loading and EECommand helpers."""

    def test_ee_command_zero(self):
        cmd = EECommand.zero()
        np.testing.assert_array_equal(cmd.position_delta, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(cmd.orientation_quat, [1.0, 0.0, 0.0, 0.0])
        assert cmd.gripper == 0.0

    def test_ee_command_config_defaults(self):
        cfg = EECommandConfig()
        np.testing.assert_array_equal(cfg.position_gains, [1.0, 1.0, 1.0])
        assert cfg.orientation_gain == 1.0
        assert cfg.max_position_delta == 0.05
        assert cfg.max_orientation_delta == 0.5

    def test_ee_command_config_from_yaml(self, tmp_path):
        yaml_content = """
control:
  ee_interface:
    position_gains: [0.8, 0.8, 0.5]
    orientation_gain: 0.7
    max_position_delta: 0.04
    max_orientation_delta: 0.4
    workspace_bounds:
      x: [-0.3, 0.3]
      y: [-0.3, 0.3]
      z: [0.85, 1.1]
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)
        cfg = EECommandConfig.from_yaml(config_file)
        np.testing.assert_allclose(cfg.position_gains, [0.8, 0.8, 0.5])
        assert cfg.orientation_gain == 0.7
        assert cfg.max_position_delta == 0.04
        assert cfg.max_orientation_delta == 0.4
        assert cfg.workspace_bounds["x"] == [-0.3, 0.3]
        assert cfg.workspace_bounds["z"] == [0.85, 1.1]

    def test_action_output_is_7d(self, interface):
        interface.reset()
        cmd = EECommand.zero()
        action = interface._transform_command(cmd)
        assert action.shape == (7,)

    def test_action_values_in_range(self, interface):
        interface.reset()
        cmd = EECommand(
            position_delta=np.array([0.1, -0.1, 0.05]),
            orientation_quat=np.array([0.99, 0.0, 0.0, 0.14]),
            gripper=0.8,
        )
        action = interface._transform_command(cmd)
        assert np.all(action >= -1.0) and np.all(action <= 1.0), (
            f"Action out of [-1,1] range: {action}"
        )
