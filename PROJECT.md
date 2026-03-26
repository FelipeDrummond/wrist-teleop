# WatchAI Teleop — IMU-Based Imitation Learning for Pick-and-Place

## One-liner

A low-cost teleoperation system using the Apple Watch Ultra 3 as an IMU-based controller to collect manipulation demonstrations in simulation, then train imitation learning policies for autonomous pick-and-place.

## Motivation

Purpose-built teleoperation interfaces (leader-follower arms, VR controllers, spacemice) cost $500–$5,000+ and require custom hardware. The Apple Watch Ultra 3 has a high-quality IMU (accelerometer + gyroscope + magnetometer), a Digital Crown for analog input, buttons for mode switching, and a Taptic Engine for haptic feedback — all on a $400 device that millions of people already own.

This project tests the hypothesis: **can a commodity smartwatch produce demonstrations of sufficient quality to train competitive imitation learning policies?**

## Project Structure

```
watchai-teleop/
├── PROJECT.md                  # This file
├── config/
│   └── teleop.yaml             # All tunable parameters (gains, filters, modes)
├── watch/                      # watchOS app (Swift)
│   └── TeleopStreamer/
│       ├── ContentView.swift
│       ├── MotionManager.swift       # CoreMotion capture + streaming
│       ├── CrownManager.swift        # Digital Crown input
│       └── WebSocketClient.swift     # Network streaming to Python
├── src/
│   ├── receiver/
│   │   └── websocket_server.py       # Async WebSocket receiver
│   ├── estimation/
│   │   ├── zupt_detector.py          # Zero-Velocity Update detection
│   │   ├── eskf.py                   # Error-State Kalman Filter
│   │   └── calibration.py            # Session calibration routine
│   ├── control/
│   │   ├── orientation_mapper.py     # Quaternion → EE orientation
│   │   ├── eskf_position_mode.py     # ESKF-based position control
│   │   ├── rate_control_mode.py      # Tilt-to-velocity fallback
│   │   ├── clutch.py                 # Decouple/re-engage mechanism
│   │   ├── gripper.py                # Crown → gripper mapping
│   │   ├── smoother.py               # Low-pass command filter
│   │   └── bridge.py                 # Main teleop loop (orchestrator)
│   ├── sim/
│   │   ├── env_setup.py              # robosuite PickPlaceSingle config
│   │   ├── ee_interface.py           # EE command → OSC_POSE action
│   │   └── frame_transforms.py       # Watch frame ↔ robot frame
│   ├── data/
│   │   ├── recorder.py               # Episode recording wrapper
│   │   ├── lerobot_converter.py      # HDF5 → LeRobot format
│   │   ├── quality_filter.py         # Automated demo quality checks
│   │   └── cli.py                    # Demo management CLI
│   ├── training/
│   │   ├── train_act.py              # ACT training config
│   │   ├── train_diffusion.py        # Diffusion Policy config
│   │   └── eval_harness.py           # Automated rollout evaluation
│   └── analysis/
│       ├── dataset_stats.py          # Dataset statistics + figures
│       └── paper_figures.py          # Publication-ready plots
├── scripts/
│   ├── run_teleop.py                 # Entry point: calibrate + teleop + record
│   ├── run_eval.py                   # Entry point: evaluate a trained policy
│   └── run_experiment.py             # Entry point: run full experiment suite
├── tests/
│   ├── test_eskf.py
│   ├── test_zupt.py
│   ├── test_frame_transforms.py
│   └── test_recording.py
└── data/
    ├── raw/                          # Raw HDF5 episodes
    ├── datasets/                     # LeRobot-formatted datasets
    └── checkpoints/                  # Trained policy checkpoints
```

## Architecture

### Pipeline Overview

```
┌──────────────────┐    WebSocket     ┌──────────────────────────────────────┐
│  Apple Watch      │ ───(~50-100Hz)──▶│  Python Backend                      │
│  Ultra 3          │    JSON stream   │                                      │
│                   │                  │  ┌────────────┐   ┌───────────────┐  │
│  CoreMotion:      │                  │  │  Receiver   │──▶│  ESKF + ZUPT  │  │
│  - quaternion     │                  │  │  (async WS) │   │  (estimation) │  │
│  - userAccel      │                  │  └────────────┘   └───────┬───────┘  │
│  - rotationRate   │                  │                           │          │
│  - crownPosition  │                  │  ┌────────────────────────▼────────┐ │
│  - buttonEvents   │                  │  │  Teleop Bridge (20Hz loop)      │ │
│                   │                  │  │  - orientation mapper            │ │
└──────────────────┘                  │  │  - translation (ESKF or rate)   │ │
                                       │  │  - clutching                    │ │
                                       │  │  - gripper (crown)             │ │
                                       │  │  - command smoothing            │ │
                                       │  └────────────────┬───────────────┘ │
                                       │                   │ EE commands      │
                                       │  ┌────────────────▼───────────────┐ │
                                       │  │  robosuite (MuJoCo)            │ │
                                       │  │  - PickPlaceSingle             │ │
                                       │  │  - Franka Panda + OSC_POSE     │ │
                                       │  │  - Camera obs (128x128)        │ │
                                       │  └────────────────┬───────────────┘ │
                                       │                   │ (s, a, obs)      │
                                       │  ┌────────────────▼───────────────┐ │
                                       │  │  Recording Wrapper             │ │
                                       │  │  → HDF5 → LeRobot format      │ │
                                       │  └────────────────────────────────┘ │
                                       └──────────────────────────────────────┘
                                                          │
                                       ┌──────────────────▼───────────────────┐
                                       │  Policy Training (LeRobot)           │
                                       │  - ACT (Zhao et al., RSS 2023)       │
                                       │  - Diffusion Policy (Chi, RSS 2023)  │
                                       │  → Autonomous pick-and-place         │
                                       └──────────────────────────────────────┘
```

### Data Flow Per Tick (~20Hz)

1. **Receiver** reads latest `SensorFrame` from WebSocket ring buffer
2. **ESKF** runs `predict(accel, dt)` and checks ZUPT
3. **Orientation mapper** computes `q_delta = q_current * q_neutral_inv`, transforms to robot frame
4. **Translation controller** computes position delta (ESKF mode) or velocity (rate mode)
5. **Smoother** applies low-pass filter to all command channels
6. **EE interface** converts to robosuite OSC_POSE 7D action: `[dx, dy, dz, dax, day, daz, gripper]`
7. **robosuite** steps physics, returns observation
8. **Recorder** logs `(obs, action, reward, done)` to episode buffer

## Core Technical Decisions

### Translation Control: The Central Challenge

The watch IMU provides **orientation** almost perfectly (Apple's sensor fusion handles this) but cannot directly measure **position**. Double-integrating acceleration to get position drifts unboundedly in seconds.

We use a layered approach:

1. **Primary: ZUPT-aided ESKF** — Error-State Kalman Filter that fuses IMU integration with zero-velocity updates. When the hand is stationary (detected via sliding-window variance test on accel/gyro norm), we inject velocity=0 as a measurement, resetting drift. Between ZUPTs (2-5 seconds of motion), drift is bounded to ~2-5cm.
   - Ref: Skog et al., "Zero-Velocity Detection — An Algorithm Evaluation," IEEE TBME, 2010
   - Ref: Solà, "Quaternion kinematics for the error-state Kalman filter," 2017

2. **Fallback: Rate control** — Wrist tilt angle maps to EE velocity. Tilt forward → robot moves forward. Drift-free but indirect (feels like a joystick, not a natural hand motion).

3. **Clutching** — Button press decouples watch from robot. Operator repositions hand, releases to re-engage. Each re-engagement triggers a ZUPT. Limits drift accumulation to short windows.

The translation mode is selectable at runtime. Task selection should favor **orientation-dominant** tasks where translation precision is less critical.

### Simulation Environment

- **robosuite** (Zhu et al., 2020) — Franka Panda, `PickPlaceSingle` task, `OSC_POSE` controller
- Chosen for: IL community standard, direct comparability with published baselines, built-in spacemouse support for baseline comparison
- Camera: agentview at 128x128 RGB (84x84 during live teleop for performance)

### Policy Training

- **LeRobot** (Cadene et al., 2024) — HuggingFace's standardized IL framework
- **ACT** (Zhao et al., RSS 2023) — action chunking with transformers, lightweight
- **Diffusion Policy** (Chi et al., RSS 2023) — DDPM-based, potentially more robust to noisy watch demos due to denoising formulation
- Dataset format: LeRobot standard (mp4-encoded camera obs, zarr/parquet state data)

### Coordinate Frames

```
Watch Frame (CoreMotion):        Robot Base Frame (robosuite):
  +X: towards crown               +X: forward (away from robot base)
  +Y: along band (up arm)         +Y: left
  +Z: out of screen               +Z: up

Transform: defined in src/sim/frame_transforms.py
           calibrated per-session via calibration routine
```

The exact transform depends on how the operator wears the watch and their posture. The calibration routine (hold still → define neutral) establishes the per-session mapping.

## Config Reference

All tunable parameters live in `config/teleop.yaml`:

```yaml
# Receiver
receiver:
  host: "0.0.0.0"
  port: 8765
  ring_buffer_size: 500

# ESKF
eskf:
  process_noise_accel: 0.1        # Q diagonal for acceleration
  process_noise_bias: 0.001       # Q diagonal for bias random walk
  zupt_measurement_noise: 0.01    # R for ZUPT velocity measurement
  workspace_bounds:                # Franka reachable space [min, max] per axis
    x: [-0.4, 0.4]
    y: [-0.4, 0.4]
    z: [0.05, 0.6]

# ZUPT Detector
zupt:
  window_size_ms: 200
  accel_var_threshold: 0.05
  gyro_var_threshold: 0.02
  hysteresis_frames: 5

# Control
control:
  mode: "eskf"                    # "eskf" or "rate"
  loop_frequency_hz: 20

  orientation:
    gain: 1.0                     # Per-axis: [roll, pitch, yaw]

  eskf_position:
    scaling: 1.0

  rate_control:
    dead_zone_deg: 5.0
    max_velocity_cm_s: 10.0
    gain_curve: "quadratic"       # "linear" or "quadratic"

  gripper:
    mode: "continuous"            # "continuous" (crown) or "toggle" (double-tap)

  smoothing:
    enabled: true
    filter_type: "ema"            # "ema" or "butterworth"
    position_cutoff_hz: 5.0
    orientation_cutoff_hz: 8.0

# Recording
recording:
  save_dir: "data/raw"
  camera_resolution: [128, 128]
  save_raw_imu: true              # Include watch telemetry in recordings
  episode_timeout_s: 30

# Environment
env:
  task: "PickPlaceSingle"
  robot: "Panda"
  controller: "OSC_POSE"
  camera_names: ["agentview"]
  render_during_teleop: true
```

## Development Guidelines

### Language & Tooling

- **watchOS app**: Swift, SwiftUI, CoreMotion, WatchKit
- **Python backend**: Python 3.10+, asyncio
- **Core deps**: `robosuite`, `mujoco`, `numpy`, `scipy` (for ESKF), `h5py`, `lerobot`, `torch`
- **Tooling**: `uv` for Python dependency management, `pytest` for tests, `wandb` for experiment tracking

### Code Conventions

- Type hints everywhere — use `dataclasses` for structured data (e.g., `SensorFrame`, `TeleopCommand`)
- Docstrings on all public classes and functions (Google style)
- Config via YAML loaded into Pydantic models — never hardcode tunable parameters
- All numpy operations should use explicit dtypes (`np.float64` for ESKF, `np.float32` for observations)
- Coordinate transforms must have unit tests with known input/output pairs

### Testing Strategy

- **Unit tests** for ESKF (predict/update with known inputs), ZUPT detector (synthetic signals), frame transforms (axis-by-axis), recording (round-trip shape checks)
- **Integration test**: synthetic IMU stream → receiver → ESKF → control → env step → recording. Run headless, verify recorded data shapes.
- No tests needed for subjective tuning (gains, filter cutoffs) — those are iterated experimentally

### Git Workflow

- Branch per Linear ticket: `wat-XX-short-description`
- One PR per ticket — keep PRs focused and reviewable
- Merge to `main` when acceptance criteria met
- Tag milestones: `v0.1-m1-sensor`, `v0.2-m2-sim`, etc.

## Milestones & Dependency Graph

```
M1: Sensor Pipeline ──────┐
  WAT-31: watchOS app      │
  WAT-32: Python receiver  │
  WAT-33: ZUPT detector    ├──▶ M3: End-to-End Teleop
  WAT-34: ESKF             │      WAT-39: Orientation mapping
  WAT-35: Calibration      │      WAT-40: ESKF position mode
                           │      WAT-41: Rate control fallback
M2: Sim Environment ───────┘      WAT-42: Clutching
  WAT-36: robosuite setup          WAT-43: Gripper (crown)
  WAT-37: EE interface             WAT-44: Command smoothing
  WAT-38: Keyboard teleop          WAT-45: Bridge server (main loop)
                                        │
                                        ▼
                                   M4: Demo Collection
                                     WAT-46: Recording wrapper
                                     WAT-47: LeRobot converter
                                     WAT-48: Episode CLI
                                     WAT-49: Quality filters
                                     WAT-50: Dataset stats
                                        │
                                        ▼
                                   M5: Training & Evaluation
                                     - ACT training config
                                     - Diffusion Policy config
                                     - Eval harness
                                     - Watch vs spacemouse experiment
                                     - Learning curves
                                     - Paper figures
```

**M1 and M2 are parallel.** M3 requires both. M4 requires M3. M5 requires M4.

## Key References

| Paper | Relevance |
|---|---|
| Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (RSS 2023) | ACT architecture — our primary policy |
| Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (RSS 2023) | Diffusion Policy — secondary policy, potentially more noise-robust |
| Chi et al., "Universal Manipulation Interface" (RSS 2024) | UMI — closest prior work on low-cost teleop interfaces for IL |
| Cadene et al., "LeRobot" (2024) | Training framework and dataset format |
| Zhu et al., "robosuite" (2020) | Simulation environment |
| Skog et al., "Zero-Velocity Detection — An Algorithm Evaluation" (IEEE TBME, 2010) | ZUPT detection algorithms |
| Solà, "Quaternion kinematics for the error-state Kalman filter" (2017) | ESKF formulation |
| Foxlin, "Pedestrian Tracking with Shoe-Mounted Inertial Sensors" (IEEE CG&A, 2005) | ZUPT-aided INS architecture |
| Herath et al., "RoNIN: Robust Neural Inertial Navigation" (ICRA 2020) | Learned inertial odometry (stretch goal) |

## Publication Target

**Contribution claim**: "We demonstrate that a commodity smartwatch can serve as a viable teleoperation interface for collecting manipulation demonstrations, achieving competitive policy performance to purpose-built interfaces on orientation-dominant tasks at a fraction of the cost."

**Core experiment**: Train ACT and Diffusion Policy on watch vs. spacemouse demos (50+ each), compare success rates on pick-and-place with statistical significance tests.

**Target venues**: CoRL workshop, IROS workshop, HRI, or CBA 2026.
