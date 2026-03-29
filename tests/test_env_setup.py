"""Acceptance tests for WAT-36: robosuite PickPlaceSingle env + OSC_POSE setup.

Tests:
  1. Environment instantiates and steps without errors
  2. OSC_POSE controller accepts 7D action vector and moves the arm
  3. Camera observations render at 128x128 RGB
  4. Object positions randomize between resets
  5. Seeded resets produce identical initial conditions
  6. Task success signal correctly fires when object reaches target
"""

from __future__ import annotations

import numpy as np
import pytest

from src.sim.env_setup import EnvConfig, make_env, get_action_dim


@pytest.fixture(scope="module")
def env_config() -> EnvConfig:
    return EnvConfig(render_during_teleop=False, seed=42)


@pytest.fixture(scope="module")
def env(env_config):
    e = make_env(config=env_config)
    yield e
    e.close()


class TestEnvInstantiatesAndSteps:
    """AC-1: PickPlaceSingle environment instantiates and steps without errors."""

    def test_env_is_created(self, env):
        assert env is not None

    def test_reset_returns_obs(self, env):
        obs = env.reset()
        assert isinstance(obs, dict)
        assert len(obs) > 0

    def test_step_returns_four_tuple(self, env):
        env.reset()
        action = np.zeros(get_action_dim(env), dtype=np.float32)
        result = env.step(action)
        assert len(result) == 4
        obs, reward, done, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_multiple_steps(self, env):
        env.reset()
        dim = get_action_dim(env)
        for _ in range(10):
            action = np.random.uniform(-1, 1, size=dim).astype(np.float32)
            obs, reward, done, info = env.step(action)
            if done:
                env.reset()


class TestOSCPoseController:
    """AC-2: OSC_POSE controller accepts 7D action vector and moves the arm."""

    def test_action_dim_is_7(self, env):
        assert get_action_dim(env) == 7

    def test_nonzero_action_moves_arm(self, env):
        obs_before = env.reset()
        ee_pos_before = obs_before["robot0_eef_pos"].copy()

        action = np.zeros(7, dtype=np.float32)
        action[0] = 1.0
        for _ in range(20):
            obs_after, _, done, _ = env.step(action)
            if done:
                break

        ee_pos_after = obs_after["robot0_eef_pos"]
        displacement = np.linalg.norm(ee_pos_after - ee_pos_before)
        assert displacement > 1e-4, (
            f"EE did not move: before={ee_pos_before}, after={ee_pos_after}"
        )

    def test_gripper_action_changes_state(self, env):
        env.reset()
        action_open = np.zeros(7, dtype=np.float32)
        action_open[6] = -1.0
        for _ in range(5):
            obs_open, _, _, _ = env.step(action_open)
        gripper_open = obs_open["robot0_gripper_qpos"].copy()

        action_close = np.zeros(7, dtype=np.float32)
        action_close[6] = 1.0
        for _ in range(10):
            obs_close, _, _, _ = env.step(action_close)
        gripper_close = obs_close["robot0_gripper_qpos"]

        assert not np.allclose(gripper_open, gripper_close, atol=1e-4), (
            "Gripper did not change state"
        )


class TestCameraObservations:
    """AC-3: Camera observations render at 128x128 RGB."""

    def test_agentview_image_shape(self, env):
        obs = env.reset()
        img = obs["agentview_image"]
        assert img.shape == (128, 128, 3), f"Unexpected image shape: {img.shape}"

    def test_agentview_image_dtype(self, env):
        obs = env.reset()
        img = obs["agentview_image"]
        assert img.dtype == np.uint8

    def test_agentview_image_not_blank(self, env):
        obs = env.reset()
        img = obs["agentview_image"]
        assert img.max() > 0, "Image is completely black"
        assert img.min() < 255, "Image is completely white"


class TestObjectRandomization:
    """AC-4: Object positions randomize between resets."""

    def test_unseeded_resets_give_different_object_positions(self):
        env = make_env(config=EnvConfig(render_during_teleop=False))
        try:
            obs1 = env.reset()
            state1 = obs1["object-state"].copy()
            obs2 = env.reset()
            state2 = obs2["object-state"].copy()
            # With no seed, consecutive resets should differ
            # (extremely unlikely to match by chance)
            assert not np.allclose(state1, state2, atol=1e-6), (
                "Object states identical across resets without seeding"
            )
        finally:
            env.close()


class TestSeededDeterminism:
    """AC-5: Seeded resets produce identical initial conditions."""

    def test_same_seed_gives_same_initial_state(self):
        env_a = make_env(config=EnvConfig(render_during_teleop=False), seed=42)
        obs_a = env_a.reset()
        eef_a = obs_a["robot0_eef_pos"].copy()
        env_a.close()

        env_b = make_env(config=EnvConfig(render_during_teleop=False), seed=42)
        obs_b = env_b.reset()
        eef_b = obs_b["robot0_eef_pos"].copy()
        env_b.close()

        np.testing.assert_allclose(
            eef_a, eef_b, atol=1e-6,
            err_msg="EEF positions differ with same seed",
        )


class TestSuccessSignal:
    """AC-6: Task success signal correctly fires when object reaches target."""

    def test_check_success_exists(self, env):
        assert hasattr(env, "_check_success"), (
            "Environment missing _check_success method"
        )

    def test_check_success_returns_bool(self, env):
        env.reset()
        action = np.zeros(get_action_dim(env), dtype=np.float32)
        env.step(action)
        result = env._check_success()
        assert isinstance(result, (bool, np.bool_))

    def test_initial_state_is_not_success(self, env):
        env.reset()
        action = np.zeros(get_action_dim(env), dtype=np.float32)
        env.step(action)
        assert not env._check_success(), (
            "Task should not be successful at initial state"
        )


class TestConfigLoading:
    """Verify EnvConfig loads correctly from YAML."""

    def test_load_from_yaml(self, tmp_path):
        yaml_content = """
env:
  task: PickPlace
  robot: Panda
  controller: OSC_POSE
  single_object_mode: 1
  object_type: can
  camera_names: ["agentview"]
  camera_height: 128
  camera_width: 128
  control_freq: 20
  horizon: 600
  seed: 42
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)
        config = EnvConfig.from_yaml(config_file)
        assert config.task == "PickPlace"
        assert config.robot == "Panda"
        assert config.controller == "OSC_POSE"
        assert config.camera_height == 128
        assert config.horizon == 600
        assert config.seed == 42
        assert config.object_type == "can"

    def test_default_config(self):
        config = EnvConfig()
        assert config.task == "PickPlace"
        assert config.robot == "Panda"
        assert config.camera_names == ["agentview"]
        assert config.single_object_mode == 1
        assert config.object_type == "can"
