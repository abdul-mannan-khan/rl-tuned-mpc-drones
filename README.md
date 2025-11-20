# RL-Enhanced MPC for Multi-Drone Systems

**Reinforcement Learning-based Hyperparameter Tuning for Model Predictive Control in UAV Trajectory Tracking**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyBullet](https://img.shields.io/badge/Simulator-PyBullet-green.svg)](https://pybullet.org/)
[![CasADi](https://img.shields.io/badge/Optimizer-CasADi-orange.svg)](https://web.casadi.org/)

## Overview

This project implements a novel approach to autonomous drone control by combining **Model Predictive Control (MPC)** with **Reinforcement Learning (RL)** for automatic hyperparameter tuning. The system learns optimal MPC cost function weights across different drone platforms, enabling robust trajectory tracking with minimal manual tuning.

### Key Features

- 🚁 **Nonlinear MPC Controller** with 12-state quadrotor dynamics
- 🤖 **RL-based Hyperparameter Tuning** using PPO (Proximal Policy Optimization)
- 🔄 **Sequential Transfer Learning** across multiple drone platforms
- 📊 **High-Fidelity Simulation** with PyBullet physics engine
- 🎯 **Robust Trajectory Tracking** with real-time optimization
- 📈 **Comprehensive Performance Metrics** and visualization

## Project Status

### Phase 1: Simulator Selection & Validation ✅
- **Status:** COMPLETE
- **Simulator:** PyBullet (gym-pybullet-drones v2.0.0)
- **Achievement:** Successfully validated MPC tracking with 190-step demonstration
- **Documentation:** [Phase 1 Report](PHASE_01_COMPLETE.md)

### Phase 2: MPC Controller Implementation ⏳
- **Status:** IMPLEMENTATION COMPLETE, TESTING IN PROGRESS
- **Achievement:** Full nonlinear MPC with CasADi optimization implemented
- **Components:**
  - ✅ 12-state nonlinear dynamics model
  - ✅ CasADi + IPOPT optimization framework
  - ✅ RK4 integration
  - ✅ Cost function with state and control penalties
  - ✅ Control constraints and warm starting
  - ✅ Test infrastructure with PyBullet
- **Documentation:** [Phase 2 Progress](PHASE_02_PROGRESS.md)

### Phase 3: RL Integration 📋
- **Status:** PLANNED
- **Components:** PPO agent, state/action/reward design, training pipeline

## Installation

### Prerequisites

- Python 3.10 or higher
- Windows 10/11 (tested) or Linux/macOS
- 8GB+ RAM recommended
- GPU optional (for faster RL training)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/rl_tuned_mpc.git
cd rl_tuned_mpc
```

2. **Create virtual environment:**
```bash
python -m venv venv_drones
# Windows:
venv_drones\Scripts\activate
# Linux/Mac:
source venv_drones/bin/activate
```

3. **Install dependencies:**
```bash
# Core dependencies
pip install "numpy<2.0,>=1.24"
pip install casadi==3.7.2
pip install pyyaml==6.0.3
pip install matplotlib

# Install gym-pybullet-drones
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones
pip install -e .
cd ..

# For RL (Phase 3)
pip install stable-baselines3==2.7.0
pip install gymnasium==1.2.2
```

## Quick Start

### 1. Run MPC Hover Test

```bash
python tests/test_mpc_controller.py
```

This will:
- Load the MPC controller configuration
- Create a PyBullet simulation environment
- Run a 10-second hover stability test
- Generate performance plots in `results/phase_02/`

### 2. Test Simulator Capabilities

```bash
python tests/test_simulator_capabilities.py
```

### 3. Explore MPC Configuration

Edit `configs/mpc_crazyflie.yaml` to adjust:
- Prediction horizon (`N`)
- Weight matrices (`Q`, `R`, `Q_terminal`)
- Control constraints
- Drone physical parameters

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────┐
│                  RL Agent (Phase 3)                 │
│         Tunes MPC weights: Q, R, Q_terminal         │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│              MPC Controller (Phase 2)               │
│  • 12-state nonlinear dynamics                      │
│  • CasADi + IPOPT optimization                      │
│  • Real-time trajectory tracking                    │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│         PyBullet Simulator (Phase 1)                │
│  • High-fidelity physics (240 Hz)                   │
│  • Crazyflie 2.X drone model                        │
│  • Real-time visualization                          │
└─────────────────────────────────────────────────────┘
```

### MPC Formulation

**State Vector (12D):**
```
x = [px, py, pz, vx, vy, vz, roll, pitch, yaw, p, q, r]
```

**Control Vector (4D):**
```
u = [thrust, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd]
```

**Optimization Problem:**
```
min  Σ [(x_k - x_ref)' Q (x_k - x_ref) + u_k' R u_k]
     + (x_N - x_ref_N)' Q_terminal (x_N - x_ref_N)

s.t. x_{k+1} = f(x_k, u_k)    [dynamics]
     u_min ≤ u_k ≤ u_max       [constraints]
     x_0 = x_current            [initial condition]
```

## Project Structure

```
rl_tuned_mpc/
├── src/
│   ├── mpc/
│   │   ├── __init__.py
│   │   └── mpc_controller.py      # Nonlinear MPC implementation
│   ├── controllers/
│   │   └── __init__.py
│   └── __init__.py
│
├── configs/
│   └── mpc_crazyflie.yaml         # MPC configuration
│
├── tests/
│   ├── test_mpc_controller.py     # MPC hover & tracking tests
│   └── test_simulator_capabilities.py  # Simulator validation
│
├── docs/
│   └── phase_01_simulator_selection/
│       └── SIMULATOR_REPORT.md
│
├── results/
│   ├── phase_01/
│   │   └── test_results_summary.json
│   └── phase_02/                  # MPC test results
│
├── checkpoints/
│   └── phase_01_checkpoint.yaml
│
├── paper/                         # Research paper (LaTeX)
│   ├── main.tex
│   └── figures/
│
├── PHASE_01_COMPLETE.md           # Phase 1 summary
├── PHASE_02_PROGRESS.md           # Phase 2 status
├── README.md                      # This file
└── .gitignore
```

## Configuration

### MPC Parameters (`configs/mpc_crazyflie.yaml`)

```yaml
mpc:
  prediction_horizon: 20        # Look-ahead steps
  timestep: 0.020833            # 48 Hz control

  Q: [100, 100, 150, ...]       # State cost weights
  R: [0.1, 1.0, 1.0, 1.0]       # Control cost weights

  u_min: [0.0, -3.0, -3.0, -3.0]
  u_max: [0.6, 3.0, 3.0, 3.0]

drone:
  mass: 0.027                    # Crazyflie 2.X
  inertia: {Ixx: 1.4e-5, ...}
```

## Testing

### Run All Tests

```bash
# MPC controller tests
python tests/test_mpc_controller.py

# Simulator capability tests
python tests/test_simulator_capabilities.py
```

### Expected Results

**MPC Hover Test:**
- Position RMSE: < 0.1 m (target)
- Solve time: < 20 ms average (target)
- Success rate: > 95%

## Performance

### Current Benchmarks (Crazyflie 2.X)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Position RMSE | < 0.05 m | 0.966 m | ⚠️ Tuning needed |
| Solve Time (avg) | < 20 ms | 149 ms | ⚠️ Optimization needed |
| Success Rate | > 95% | 99.8% | ✅ |
| Real-time Factor | > 0.5x | 0.14x | ⚠️ |

**Note:** Initial implementation prioritizes correctness over performance. Optimization in progress.

## Development Roadmap

- [x] **Phase 1:** Simulator selection & validation (COMPLETE)
- [x] **Phase 2:** MPC controller implementation (IN PROGRESS)
  - [x] Core MPC implementation
  - [x] Test infrastructure
  - [ ] Control allocation tuning
  - [ ] Weight matrix optimization
  - [ ] Performance optimization
- [ ] **Phase 3:** RL integration
  - [ ] PPO agent implementation
  - [ ] State/action/reward design
  - [ ] Training pipeline
  - [ ] Hyperparameter search
- [ ] **Phase 4:** Multi-platform validation
- [ ] **Phase 5:** Transfer learning
- [ ] **Phase 6:** Real hardware testing

## Dependencies

### Core
- Python 3.10+
- NumPy 1.26.4
- CasADi 3.7.2 (optimization)
- PyYAML 6.0.3

### Simulation
- PyBullet 3.2.7
- gym-pybullet-drones 2.0.0
- Gymnasium 1.2.2

### RL (Phase 3)
- Stable-Baselines3 2.7.0
- PyTorch (via SB3)

### Visualization
- Matplotlib 3.10.1

## Contributing

This is a research project. For questions or collaboration:
- **Author:** Dr. Abdul Manan Khan
- **Email:** [Your Email]
- **Institution:** [Your Institution]

## Citation

If you use this work in your research, please cite:

```bibtex
@article{khan2024rl_mpc,
  title={RL-Enhanced Model Predictive Control for Multi-Drone Systems},
  author={Khan, Abdul Manan},
  journal={[Conference/Journal]},
  year={2024}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- **gym-pybullet-drones:** https://github.com/utiasDSL/gym-pybullet-drones
- **CasADi:** https://web.casadi.org/
- **PyBullet:** https://pybullet.org/
- **Stable-Baselines3:** https://stable-baselines3.readthedocs.io/

## References

1. J. A. E. Andersson et al., "CasADi: A software framework for nonlinear optimization," Mathematical Programming Computation, 2019.
2. Panerati et al., "gym-pybullet-drones: A Gym environment for quadrotor control," arXiv preprint, 2021.
3. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.

---

**Last Updated:** 2025-11-20
**Status:** Active Development - Phase 2 in progress
