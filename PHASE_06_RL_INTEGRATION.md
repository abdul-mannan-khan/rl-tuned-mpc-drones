# Phase 6: RL Integration - PPO for MPC Hyperparameter Tuning

**Status:** Infrastructure Complete - Ready for Training
**Date:** 2025-11-22
**Algorithm:** Proximal Policy Optimization (PPO)
**Platform:** Crazyflie (baseline)

---

## Executive Summary

Phase 6 implements Reinforcement Learning (RL) for automatic MPC hyperparameter tuning using Proximal Policy Optimization (PPO). This phase provides the infrastructure to learn optimal MPC weight matrices (Q, R) and prediction horizon (N) through trial-and-error interaction with the simulation environment.

### Context from Phase 5

**Important Note:** Phase 5 achieved **0.10-0.12m RMSE** using CF2X model with Crazyflie MPC configuration - already **10× better** than Phase 6's target of 1.33m RMSE from the original roadmap.

**Implications:**
- RL tuning may not be necessary for the current CF2X simulation setup
- RL remains valuable for:
  - Real hardware deployment (when dynamics differ from simulation)
  - Trajectory optimization
  - Disturbance rejection
  - Multi-drone coordination
  - Exploration of hyperparameter sensitivity

---

## Implementation Overview

### Phase 6.1: Gymnasium Environment

**File:** `src/rl/mpc_tuning_env.py`

Implements a Gymnasium-compliant environment that treats MPC weight tuning as an RL problem.

**State Space (29D):**
```python
[position_error (3),         # Current tracking error
 velocity_error (3),         # Velocity tracking error
 control_effort (4),         # Recent control commands
 current_Q_weights (12),     # Normalized state costs
 current_R_weights (4),      # Normalized control costs
 current_horizon (1),        # Prediction horizon N
 settling_time (1),          # Time to settle
 overshoot (1)]              # Maximum overshoot
```

**Action Space (17D):**
```python
[Q_weight_adjustments (12),  # Exponential multipliers for Q
 R_weight_adjustments (4),   # Exponential multipliers for R
 horizon_adjustment (1)]     # Linear change to N
```

**Reward Function:**
```python
r = -10*pos_error - 1*vel_error - 0.01*control - 5*overshoot
```

**Key Features:**
- Bryson's Rule initialization for Q and R weights
- Exponential action scaling (allows large weight adjustments)
- Multiple trajectory types (circular, figure-8, hover)
- Integration with CF2X PyBullet model (Phase 5 finding)
- Crash detection and termination

### Phase 6.2: PPO Training

**File:** `src/rl/train_ppo.py`

Complete training pipeline using Stable Baselines3 PPO implementation.

**PPO Hyperparameters:**
```python
learning_rate: 3e-4
n_steps: 2048              # Steps per env before update
batch_size: 64
n_epochs: 10               # Optimization epochs per update
gamma: 0.99                # Discount factor
gae_lambda: 0.95           # GAE parameter
clip_range: 0.2            # PPO clip range
ent_coef: 0.01             # Entropy coefficient
vf_coef: 0.5               # Value function coefficient
max_grad_norm: 0.5         # Gradient clipping
```

**Network Architecture:**
```python
Policy: MLP (Multi-Layer Perceptron)
Actor network: [256, 256] hidden units
Critic network: [256, 256] hidden units
```

**Training Features:**
- Parallel environment training (4 workers default)
- Checkpointing every 1000 steps
- Evaluation callback with best model saving
- Weights & Biases (W&B) integration
- TensorBoard logging
- Progress bar and real-time metrics

---

## Installation

### 1. Install RL Dependencies

```bash
pip install -r requirements_rl.txt
```

This installs:
- `stable-baselines3[extra]>=2.0.0` - PPO implementation
- `gymnasium>=0.28.0` - RL environment standard
- `wandb>=0.15.0` - Experiment tracking
- `tensorboard>=2.13.0` - Logging

### 2. (Optional) Configure Weights & Biases

```bash
wandb login
```

Or train without W&B:
```bash
python src/rl/train_ppo.py --no-wandb
```

---

## Usage

### Test Environment (Recommended First Step)

Verify the environment works before training:

```bash
python tests/test_rl_env.py
```

**Expected Output:**
```
======================================================================
RL Environment Test Suite
======================================================================
Testing environment creation...
  ✓ Environment created successfully
  - Observation space: Box(-inf, inf, (29,), float32)
  - Action space: Box(-1.0, 1.0, (17,), float32)

Testing environment reset...
  ✓ Environment reset successfully
  ...

Test Results: 5/5 passed
======================================================================

✓ All tests passed! Environment is ready for training.
```

### Training

#### Basic Training (Crazyflie, Circular Trajectory)

```bash
python src/rl/train_ppo.py --platform crazyflie --trajectory circular --timesteps 20000
```

#### Advanced Training Options

```bash
# Train with figure-8 trajectory
python src/rl/train_ppo.py --platform crazyflie --trajectory figure8 --timesteps 20000

# Train with more parallel environments (faster but more RAM)
python src/rl/train_ppo.py --n-envs 8 --timesteps 20000

# Train without W&B logging
python src/rl/train_ppo.py --no-wandb --timesteps 20000

# Faster training (fewer timesteps for testing)
python src/rl/train_ppo.py --timesteps 5000 --save-freq 500
```

**Training Output:**
```
======================================================================
PPO Training for MPC Hyperparameter Tuning
======================================================================
Platform:        crazyflie
Trajectory:      circular
Total timesteps: 20,000
Parallel envs:   4
Eval frequency:  1,000
Save frequency:  1,000
======================================================================

Creating vectorized environment...
Creating evaluation environment...
Initializing PPO model...

Starting PPO training...
Training will run for 20,000 timesteps
Expected episodes: ~40

[Progress bar with episode metrics]
```

### Evaluation

#### Evaluate Trained Model

```bash
# Evaluate final model
python src/rl/train_ppo.py --evaluate --n-eval-episodes 10

# Evaluate specific checkpoint
python src/rl/train_ppo.py --evaluate --model-path checkpoints/crazyflie/ppo_mpc_10000_steps

# Evaluate with GUI visualization
python src/rl/train_ppo.py --evaluate --render --n-eval-episodes 3
```

**Evaluation Output:**
```
Loading model from: models/crazyflie/ppo_mpc_final

Evaluating PPO policy on crazyflie (10 episodes)...
Trajectory: circular
======================================================================

  Episode  1: RMSE = 0.1245 m, Reward =  -145.23, Length = 500 steps
  Episode  2: RMSE = 0.1103 m, Reward =  -132.45, Length = 500 steps
  ...
  Episode 10: RMSE = 0.1176 m, Reward =  -138.91, Length = 500 steps

======================================================================
Evaluation Results:
======================================================================
  Mean RMSE:   0.1182 ± 0.0045 m
  Mean Reward: -139.23 ± 6.12
  Mean Length: 500.0 steps
======================================================================
```

---

## File Structure

```
rl_tuned_mpc/
├── src/
│   └── rl/
│       ├── __init__.py                 # RL module init
│       ├── mpc_tuning_env.py           # Gymnasium environment (850 lines)
│       └── train_ppo.py                # PPO training script (450 lines)
├── tests/
│   └── test_rl_env.py                  # Environment test suite
├── requirements_rl.txt                 # RL dependencies
├── models/                             # Trained models
│   └── crazyflie/
│       ├── ppo_mpc_final.zip           # Final trained model
│       ├── best/                       # Best model from eval
│       └── wandb/                      # W&B checkpoints
├── checkpoints/                        # Training checkpoints
│   ├── crazyflie/
│   │   ├── ppo_mpc_1000_steps.zip
│   │   ├── ppo_mpc_2000_steps.zip
│   │   └── ...
│   └── phase_06_checkpoint.yaml        # Phase 6 completion
└── logs/                               # Training logs
    └── crazyflie/
        ├── PPO_1/                      # TensorBoard logs
        └── eval/                       # Evaluation results
```

---

## Key Implementation Details

### Bryson's Rule Initialization

The environment initializes MPC weights using Bryson's Rule:

```python
Q_i = 1 / (max_acceptable_deviation_i)²
R_i = 1 / (max_acceptable_control_i)²
```

**Crazyflie Parameters:**
```python
platform_params = {
    'pos': 0.10,        # 10cm position error
    'vel': 0.25,        # 0.25 m/s velocity error
    'att': 0.20,        # 0.2 rad (~11.5°) attitude error
    'ang_vel': 0.25     # 0.25 rad/s angular rate error
}

Q_bryson = [100, 100, 100,  # Position (px, py, pz)
            16, 16, 16,      # Velocity (vx, vy, vz)
            25, 25, 25,      # Attitude (roll, pitch, yaw)
            16, 16, 16]      # Angular rates (p, q, r)
```

### Action Scaling

Actions are in range [-1, 1] and mapped exponentially to weight multipliers:

```python
Q_multiplier = exp(action * 0.5)  # Range: ~[0.6, 1.6]
Q_new = Q_bryson * Q_multiplier
```

This allows RL to make significant adjustments while maintaining stability.

### Trajectory Generation

Three trajectory types supported:

1. **Circular:** Constant radius, useful for steady-state tuning
2. **Figure-8:** Lemniscate with varying curvature, tests transient response
3. **Hover:** Stationary setpoint, tests disturbance rejection

---

## Expected Performance

### Phase 6 Targets (from Roadmap)

- **Target RMSE:** ≤ 1.35m
- **Training time:** ~3 hours (20,000 timesteps × 4 envs)
- **Convergence:** Reward plateau after ~15,000 timesteps
- **Improvement:** 3-6× better than Bryson's Rule baseline

### Reality Check (Based on Phase 5)

**Current Baseline (Bryson's Rule + CF2X):** 0.10-0.12m RMSE
**Phase 6 Target (from Roadmap):** 1.33m RMSE

**Analysis:**
- Current baseline already **10× better** than Phase 6 target
- RL unlikely to improve significantly on CF2X simulation
- RL remains valuable for:
  - Real hardware (different dynamics than simulation)
  - Investigating hyperparameter sensitivity
  - Learning robust policies across trajectory variations

---

## Troubleshooting

### Environment Errors

**Problem:** `ImportError: No module named 'stable_baselines3'`
**Solution:** Install RL dependencies:
```bash
pip install -r requirements_rl.txt
```

**Problem:** PyBullet GUI crashes or freezes
**Solution:** Disable GUI during training:
```bash
python src/rl/train_ppo.py --no-wandb  # GUI disabled by default in training
```

### Training Issues

**Problem:** Training very slow
**Solution:** Reduce parallel environments or episode length:
```bash
python src/rl/train_ppo.py --n-envs 2 --timesteps 10000
```

**Problem:** MPC solver failures during training
**Solution:** Check solver logs, may need to adjust weight bounds in environment

### Evaluation Issues

**Problem:** Evaluation RMSE worse than baseline
**Analysis:** This is expected! Phase 5 baseline (0.10m) is already near-optimal for CF2X simulation. RL needs different dynamics to show improvement.

---

## Phase 6 Deliverables

### Completed

- [x] Gymnasium environment implementation (`mpc_tuning_env.py`)
- [x] Bryson's Rule initialization
- [x] PPO training infrastructure (`train_ppo.py`)
- [x] Environment test suite (`test_rl_env.py`)
- [x] W&B and TensorBoard integration
- [x] Comprehensive documentation
- [x] Requirements file for dependencies

### Ready for Training

- [ ] Run 20,000 timestep training
- [ ] Evaluate trained model
- [ ] Create Phase 6 checkpoint
- [ ] Compare RL vs. Bryson's Rule performance

---

## Next Steps

### Option 1: Complete Phase 6 as Designed

1. Install RL dependencies: `pip install -r requirements_rl.txt`
2. Test environment: `python tests/test_rl_env.py`
3. Train PPO: `python src/rl/train_ppo.py --timesteps 20000`
4. Evaluate: `python src/rl/train_ppo.py --evaluate`
5. Create checkpoint documenting results

### Option 2: Adapt Phase 6 for Real Hardware

Since simulation already achieves excellent performance (0.10m RMSE), consider:
1. Skip RL training on CF2X simulation
2. Use Phase 6 infrastructure for real hardware validation
3. Apply RL when actual drone dynamics differ from simulation
4. Focus on sim-to-real transfer learning

### Option 3: Extend Phase 6 Scope

Use RL infrastructure for:
- **Trajectory optimization:** Learn optimal reference trajectories
- **Multi-drone coordination:** Extend to cooperative control
- **Disturbance rejection:** Train robust policies under wind/noise
- **Model uncertainty:** Learn policies robust to parameter variations

---

## References

### Development Roadmap
- Original target: 1.33m RMSE (from Phase 6 specification)
- Phase 5 achievement: 0.10-0.12m RMSE (10× better)

### Papers and Resources
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- PPO: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Bryson's Rule: Bryson & Ho, "Applied Optimal Control" (1975)

---

## Conclusion

Phase 6 infrastructure is complete and ready for RL training. However, given Phase 5's exceptional results (0.10m RMSE), the value proposition for RL tuning on the current CF2X simulation is limited. **RL integration is most valuable when deploying to real hardware or exploring scenarios beyond the current simulation capabilities.**

**Recommendation:** Proceed with lightweight training (5,000-10,000 timesteps) to validate infrastructure, then focus on real hardware deployment or extended RL applications (trajectory optimization, multi-drone control, etc.).

---

**Phase 6 Status:** ✅ Infrastructure Complete - Ready for Training
**Created:** 2025-11-22
**Author:** Phase 6 Development Team

---
