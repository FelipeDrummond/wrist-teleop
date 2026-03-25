# 🤖 WatchAI Teleop

**Use your Apple Watch as a teleoperation controller to collect robot manipulation demonstrations for imitation learning.**

Apple Watch Ultra 3 IMU → Python → MuJoCo/robosuite (Franka Panda) → LeRobot dataset → ACT / Diffusion Policy → Autonomous pick-and-place

---

## Why

Teleoperation interfaces for imitation learning are expensive ($500–$5,000+) and require custom hardware. The Apple Watch Ultra 3 already has a high-quality 6-axis IMU, a Digital Crown for analog input, buttons for mode switching, and haptic feedback — for $400.

This project explores whether a commodity smartwatch can collect demonstrations good enough to train competitive manipulation policies.

## How It Works

The operator wears the watch and moves their hand to control a simulated Franka Panda arm. Wrist **orientation** maps directly to the robot's end-effector orientation (this is the watch's strong suit). **Translation** is estimated via a ZUPT-aided Error-State Kalman Filter that bounds IMU drift between stationary moments, with a rate-control fallback for robustness. The Digital Crown controls the gripper. A clutch button decouples the watch so you can reposition your hand without moving the robot.

Demonstrations are recorded in [LeRobot](https://github.com/huggingface/lerobot) format and used to train [ACT](https://tonyzhaozh.github.io/aloha/) and [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) models.

## Architecture

```
Apple Watch Ultra 3                    Python Backend
┌─────────────────────┐     WS       ┌─────────────────────────────┐
│ CoreMotion @ 100Hz  │────(JSON)───▶│ Receiver → ESKF + ZUPT      │
│ Digital Crown       │              │       ↓                      │
│ Action Button       │              │ Teleop Bridge (20Hz)         │
│ Taptic Engine       │◀──(haptics)──│  orientation + translation   │
└─────────────────────┘              │  clutch + gripper + smoothing│
                                     │       ↓                      │
                                     │ robosuite / MuJoCo           │
                                     │  Franka Panda · OSC_POSE     │
                                     │  PickPlaceSingle · 128×128   │
                                     │       ↓                      │
                                     │ Recording → HDF5 → LeRobot  │
                                     └──────────────┬──────────────┘
                                                    ↓
                                     Policy Training (ACT / Diffusion Policy)
                                                    ↓
                                          Autonomous Pick-and-Place
```

## Quick Start

### Prerequisites

- Apple Watch Ultra 3 (or any watchOS 10+ device with CoreMotion)
- Mac with Xcode 15+ (for the watchOS app)
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for dependency management

### Setup

```bash
# Clone
git clone https://github.com/<you>/watchai-teleop.git
cd watchai-teleop

# Python environment
uv sync

# Install robosuite + MuJoCo
uv pip install robosuite mujoco

# Build and deploy the watchOS app via Xcode
open watch/TeleopStreamer.xcodeproj
```

### Run

```bash
# 1. Start the receiver (waits for watch connection)
python scripts/run_teleop.py --config config/teleop.yaml

# 2. Open the watch app, tap "Connect"

# 3. Calibrate: hold wrist still for 3 seconds

# 4. Teleoperate! Collect demos with:
#    - Wrist rotation → gripper orientation
#    - Hand movement → arm translation (ESKF mode)
#    - Crown → gripper open/close
#    - Action button → clutch (decouple/re-engage)
#    - 's' key → mark episode as success
#    - 'r' key → discard and reset

# 5. Export to LeRobot format
python src/data/cli.py export --input data/raw --output data/datasets/watch_demos

# 6. Train a policy
python src/training/train_act.py --dataset data/datasets/watch_demos

# 7. Evaluate
python scripts/run_eval.py --checkpoint data/checkpoints/act_latest.pt --n_rollouts 50
```

## Project Structure

```
watchai-teleop/
├── PROJECT.md                    # Detailed architecture & design doc (for Claude Code)
├── config/teleop.yaml            # All tunable parameters
├── watch/TeleopStreamer/          # watchOS app (Swift)
├── src/
│   ├── receiver/                 # WebSocket server for IMU stream
│   ├── estimation/               # ZUPT detector, ESKF, calibration
│   ├── control/                  # Orientation, translation, clutch, gripper, smoothing
│   ├── sim/                      # robosuite env setup, EE interface, frame transforms
│   ├── data/                     # Recording, LeRobot conversion, quality filters, CLI
│   ├── training/                 # ACT & Diffusion Policy configs, eval harness
│   └── analysis/                 # Dataset stats, paper figures
├── scripts/                      # Entry points: run_teleop, run_eval, run_experiment
├── tests/
└── data/                         # Raw episodes, datasets, checkpoints
```

See [`PROJECT.md`](PROJECT.md) for the full design document including coordinate frames, config reference, and implementation details.

## The Translation Problem

A wrist-mounted IMU gives you orientation for free (Apple's sensor fusion is excellent) but **cannot directly measure position**. Double-integrating acceleration drifts unboundedly in seconds.

We mitigate this with a layered approach:

1. **ZUPT-aided ESKF** — Zero-Velocity Updates reset drift whenever the hand pauses. Between pauses (~2-5s), drift is bounded to ~2-5cm.
2. **Rate control fallback** — Wrist tilt maps to velocity (like a joystick). Drift-free but indirect.
3. **Clutching** — Button press freezes the robot, letting you reposition your hand. Each re-engage triggers a ZUPT.

Task selection compensates for the interface: pick-and-place is orientation-dominant (getting the grasp angle right matters more than millimeter-precise positioning).

## Experiment Design

The core question: *how do watch-collected demos compare to spacemouse demos for policy training?*

| Condition | Interface | Policy | Demos | Eval Rollouts |
|-----------|-----------|--------|-------|---------------|
| Watch + ACT | Apple Watch | ACT | 50 | 50 |
| Watch + DP | Apple Watch | Diffusion Policy | 50 | 50 |
| Spacemouse + ACT | 3Dconnexion | ACT | 50 | 50 |
| Spacemouse + DP | 3Dconnexion | Diffusion Policy | 50 | 50 |

Plus learning curves ({10, 25, 50, 75, 100} demos), translation mode ablation (ESKF vs rate control), and failure analysis.

## Key References

- **ACT** — Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (RSS 2023)
- **Diffusion Policy** — Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (RSS 2023)
- **UMI** — Chi et al., "Universal Manipulation Interface" (RSS 2024)
- **LeRobot** — Cadene et al. (2024) — training framework and dataset format
- **robosuite** — Zhu et al. (2020) — simulation environment
- **ZUPT-aided INS** — Skog et al., IEEE TBME (2010)
- **ESKF** — Solà, "Quaternion kinematics for the error-state Kalman filter" (2017)

## License

MIT
