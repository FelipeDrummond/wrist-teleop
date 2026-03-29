"""robosuite PickPlace environment factory.

Configures PickPlace (single object) with Franka Panda, OSC_POSE controller,
agentview camera at 128x128, and deterministic seed control.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
import robosuite as suite
from robosuite import load_composite_controller_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvConfig:
    """Simulation environment configuration loaded from teleop.yaml."""

    task: str = "PickPlace"
    robot: str = "Panda"
    controller: str = "OSC_POSE"
    single_object_mode: int = 1
    object_type: str = "can"
    camera_names: list[str] = field(default_factory=lambda: ["agentview"])
    camera_height: int = 128
    camera_width: int = 128
    control_freq: int = 20
    horizon: int = 600
    render_during_teleop: bool = False
    seed: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> EnvConfig:
        """Load env config from the 'env' section of a YAML file."""
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)
        env_section = raw.get("env", {})
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in env_section.items() if k in known_fields}
        return cls(**filtered)


def make_env(
    config: EnvConfig | None = None,
    config_path: str | Path | None = None,
    render: bool = False,
    seed: int | None = None,
) -> suite.environments.base.MujocoEnv:
    """Create a configured robosuite PickPlace environment.

    Args:
        config: EnvConfig instance. If None, loads from config_path.
        config_path: Path to teleop.yaml. Used only if config is None.
        render: Enable on-screen rendering (overrides config).
        seed: Random seed (overrides config).

    Returns:
        A robosuite environment ready to step.
    """
    if config is None:
        if config_path is None:
            config_path = Path(__file__).parents[2] / "config" / "teleop.yaml"
        config = EnvConfig.from_yaml(config_path)

    effective_seed = seed if seed is not None else config.seed

    # Seed numpy before env creation for deterministic object placement.
    if effective_seed is not None:
        np.random.seed(effective_seed)

    # Panda default composite config already uses OSC_POSE for the arm.
    controller_config = load_composite_controller_config(robot=config.robot)

    env = suite.make(
        env_name=config.task,
        robots=config.robot,
        controller_configs=controller_config,
        has_renderer=render or config.render_during_teleop,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=config.camera_names,
        camera_heights=config.camera_height,
        camera_widths=config.camera_width,
        control_freq=config.control_freq,
        horizon=config.horizon,
        single_object_mode=config.single_object_mode,
        object_type=config.object_type,
        reward_shaping=False,
    )

    logger.info(
        "Created %s env: robot=%s, controller=%s, camera=%dx%d, horizon=%d, seed=%s",
        config.task,
        config.robot,
        config.controller,
        config.camera_width,
        config.camera_height,
        config.horizon,
        effective_seed,
    )

    return env


def get_action_dim(env: suite.environments.base.MujocoEnv) -> int:
    """Return the action dimension for the environment."""
    return env.action_dim


def sample_action(env: suite.environments.base.MujocoEnv) -> np.ndarray:
    """Sample a random action within the environment's action space."""
    dim = get_action_dim(env)
    return np.random.uniform(-1.0, 1.0, size=dim).astype(np.float32)
