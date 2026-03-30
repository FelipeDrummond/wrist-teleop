"""End-effector command interface for robosuite OSC_POSE controller.

Translates EECommand (position delta + orientation quaternion + gripper)
into the 7D normalized action vector expected by robosuite's OSC_POSE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from src.sim.frame_transforms import (
    NOMINAL_WATCH_TO_BASE,
    transform_position_delta,
    transform_orientation_quat,
    quat_to_rotvec,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EECommand:
    """A single end-effector command in world (watch) frame.

    Attributes:
        position_delta: (3,) [dx, dy, dz] in world frame, meters.
        orientation_quat: (4,) [w, x, y, z] delta quaternion (scalar-first).
        gripper: Gripper command, -1.0 (open) to 1.0 (close).
    """

    position_delta: np.ndarray  # shape: (3,)
    orientation_quat: np.ndarray  # shape: (4,) wxyz
    gripper: float

    @classmethod
    def zero(cls) -> EECommand:
        """No-op command: zero position/orientation delta, neutral gripper."""
        return cls(
            position_delta=np.zeros(3, dtype=np.float64),
            orientation_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            gripper=0.0,
        )


@dataclass(frozen=True)
class EECommandConfig:
    """Configuration for the EE command interface.

    Attributes:
        position_gains: (3,) per-axis gain multipliers [x, y, z].
        orientation_gain: Scalar gain for orientation.
        workspace_bounds: Dict with 'x', 'y', 'z' keys, each [min, max].
        frame_rotation: (3,3) rotation from watch frame to robot base frame.
        max_position_delta: Max position delta per step [meters]. From OSC config.
        max_orientation_delta: Max orientation delta per step [radians]. From OSC config.
    """

    position_gains: np.ndarray = field(
        default_factory=lambda: np.ones(3, dtype=np.float64)
    )
    orientation_gain: float = 1.0
    workspace_bounds: dict = field(
        default_factory=lambda: {
            "x": [-0.5, 0.5],
            "y": [-0.5, 0.5],
            "z": [0.82, 1.2],
        }
    )
    frame_rotation: np.ndarray = field(
        default_factory=lambda: NOMINAL_WATCH_TO_BASE.copy()
    )
    max_position_delta: float = 0.05
    max_orientation_delta: float = 0.5

    @classmethod
    def from_yaml(cls, path: str | Path) -> EECommandConfig:
        """Load config from teleop.yaml.

        Reads ee_interface section from control.
        """
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        ee_section = raw.get("control", {}).get("ee_interface", {})

        kwargs: dict = {}
        if "position_gains" in ee_section:
            kwargs["position_gains"] = np.array(
                ee_section["position_gains"], dtype=np.float64
            )
        if "orientation_gain" in ee_section:
            kwargs["orientation_gain"] = float(ee_section["orientation_gain"])
        if "max_position_delta" in ee_section:
            kwargs["max_position_delta"] = float(ee_section["max_position_delta"])
        if "max_orientation_delta" in ee_section:
            kwargs["max_orientation_delta"] = float(ee_section["max_orientation_delta"])
        if "workspace_bounds" in ee_section:
            kwargs["workspace_bounds"] = ee_section["workspace_bounds"]

        return cls(**kwargs)


class EECommandInterface:
    """Translates EECommands into robosuite OSC_POSE actions.

    Pipeline per step:
        1. Transform position delta: watch frame -> robot base frame
        2. Apply per-axis position gains
        3. Workspace clamp (clip absolute target, recompute delta)
        4. Normalize position to [-1, 1]
        5. Transform orientation: watch frame -> robot base frame
        6. Convert quaternion delta -> axis-angle rotation vector
        7. Apply orientation gain, normalize to [-1, 1]
        8. Assemble 7D action, step env
    """

    def __init__(
        self,
        env,
        config: EECommandConfig | None = None,
    ) -> None:
        self._env = env
        self._config = config or EECommandConfig()
        self._last_obs: dict | None = None

        # Pre-extract bounds as arrays for fast clamping
        bounds = self._config.workspace_bounds
        self._ws_min = np.array(
            [bounds["x"][0], bounds["y"][0], bounds["z"][0]], dtype=np.float64
        )
        self._ws_max = np.array(
            [bounds["x"][1], bounds["y"][1], bounds["z"][1]], dtype=np.float64
        )

    @property
    def env(self):
        return self._env

    def reset(self) -> dict:
        """Reset the environment and internal state."""
        self._last_obs = self._env.reset()
        return self._last_obs

    def step(self, command: EECommand) -> tuple[dict, float, bool, dict]:
        """Convert an EECommand to a 7D action and step the environment."""
        action = self._transform_command(command)
        obs, reward, done, info = self._env.step(action)
        self._last_obs = obs
        return obs, reward, done, info

    def get_ee_pos(self) -> np.ndarray:
        """Current EE position from the last observation."""
        assert self._last_obs is not None, "Call reset() before get_ee_pos()"
        return self._last_obs["robot0_eef_pos"].copy()

    def get_ee_quat_xyzw(self) -> np.ndarray:
        """Current EE orientation (x,y,z,w) from the last observation."""
        assert self._last_obs is not None, "Call reset() before get_ee_quat_xyzw()"
        return self._last_obs["robot0_eef_quat"].copy()

    def _transform_command(self, command: EECommand) -> np.ndarray:
        """Full pipeline: EECommand -> 7D normalized action."""
        cfg = self._config

        # --- Position ---
        pos_base = transform_position_delta(
            command.position_delta, cfg.frame_rotation
        )
        pos_gained = pos_base * cfg.position_gains
        pos_clamped = self._clamp_workspace(pos_gained)
        pos_normalized = self._normalize_position(pos_clamped)

        # --- Orientation ---
        ori_base_wxyz = transform_orientation_quat(
            command.orientation_quat, cfg.frame_rotation
        )
        ori_rotvec = quat_to_rotvec(ori_base_wxyz)
        ori_gained = ori_rotvec * cfg.orientation_gain
        ori_normalized = self._normalize_orientation(ori_gained)

        # --- Gripper ---
        gripper = np.clip(command.gripper, -1.0, 1.0)

        # --- Assemble 7D action ---
        action = np.zeros(7, dtype=np.float64)
        action[0:3] = pos_normalized
        action[3:6] = ori_normalized
        action[6] = gripper

        return action

    def _clamp_workspace(self, delta: np.ndarray) -> np.ndarray:
        """Clip the target absolute position to workspace bounds.

        Returns the effective delta after clamping.
        """
        current_pos = self.get_ee_pos()
        target = current_pos + delta
        clamped = np.clip(target, self._ws_min, self._ws_max)
        return clamped - current_pos

    def _normalize_position(self, delta: np.ndarray) -> np.ndarray:
        """Scale position delta to [-1, 1] based on max_position_delta."""
        return np.clip(
            delta / self._config.max_position_delta, -1.0, 1.0
        )

    def _normalize_orientation(self, rotvec: np.ndarray) -> np.ndarray:
        """Scale rotation vector to [-1, 1] based on max_orientation_delta."""
        return np.clip(
            rotvec / self._config.max_orientation_delta, -1.0, 1.0
        )
