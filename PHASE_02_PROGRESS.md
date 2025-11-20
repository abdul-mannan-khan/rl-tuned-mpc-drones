# Phase 2: MPC Controller Implementation - IN PROGRESS

**Date:** 2025-11-20
**Status:** IMPLEMENTATION COMPLETE, TESTING IN PROGRESS
**Next Phase:** Tuning and validation

---

## Summary

Phase 2 focuses on implementing a nonlinear Model Predictive Control (MPC) controller for quadrotor trajectory tracking. The controller uses CasADi for symbolic computation and IPOPT for nonlinear optimization.

---

## Completed Tasks

### 1. MPC Module Structure ✅
**Location:** `src/mpc/`

Created organized module structure:
```
src/
├── __init__.py
├── mpc/
│   ├── __init__.py
│   └── mpc_controller.py  (530 lines)
└── controllers/
    └── __init__.py
```

### 2. MPC Controller Implementation ✅
**File:** `src/mpc/mpc_controller.py`

**Key Features:**
- **State Space:** 12D [px, py, pz, vx, vy, vz, roll, pitch, yaw, p, q, r]
- **Control Space:** 4D [thrust, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd]
- **Prediction Horizon:** N = 20 steps
- **Control Frequency:** 48 Hz (dt = 0.020833s)

**Implementation Details:**

#### Nonlinear Dynamics Model
```python
def _dynamics(self, x, u):
    # 12-state quadrotor dynamics
    # - Position kinematics
    # - Velocity dynamics (thrust forces)
    # - Attitude kinematics (Euler angles)
    # - Angular acceleration (first-order actuator model)
```

- Full 6-DOF rigid body dynamics
- Thrust force in world frame using rotation matrices
- Euler angle kinematics
- First-order actuator model (τ = 10.0)

#### Optimization Framework
- **Solver:** CasADi Opti interface with IPOPT
- **Integration:** RK4 (4th order Runge-Kutta)
- **Variables:** State trajectory X and control trajectory U
- **Parameters:** Initial state X0 and reference trajectory X_ref

#### Cost Function
**Stage Cost (k = 0...N-1):**
```
L_k = (x_k - x_ref_k)' * Q * (x_k - x_ref_k) + u_k' * R * u_k
```

**Terminal Cost:**
```
L_N = (x_N - x_ref_N)' * Q_terminal * (x_N - x_ref_N)
```

**Total Objective:**
```
J = Σ L_k + L_N
```

#### Constraints
1. **Initial Condition:** x_0 = x_current
2. **Dynamics:** x_{k+1} = f(x_k, u_k) ∀k
3. **Control Limits:** u_min ≤ u_k ≤ u_max ∀k

#### Advanced Features
- **Warm Starting:** Uses shifted previous solution as initial guess
- **Weight Update:** `update_weights()` method for RL integration
- **Statistics Tracking:** Solve time, success rate, failure count
- **Robust Error Handling:** Returns safe hover control on failure

### 3. Configuration File ✅
**File:** `configs/mpc_crazyflie.yaml`

**MPC Parameters:**
- Prediction horizon: N = 20
- Timestep: dt = 0.020833s (48 Hz)
- Physics frequency: 240 Hz (5 physics steps per control step)

**Weight Matrices (Baseline):**
```yaml
Q: [100, 100, 150, 10, 10, 20, 50, 50, 30, 1, 1, 1]
R: [0.1, 1.0, 1.0, 1.0]
Q_terminal: [500, 500, 750, 50, 50, 100, 250, 250, 150, 10, 10, 10]
```

**Control Constraints:**
```yaml
u_min: [0.0, -3.0, -3.0, -3.0]  # [thrust_min, ang_vel_min...]
u_max: [0.6, 3.0, 3.0, 3.0]     # Crazyflie physical limits
```

**Drone Physical Parameters (Crazyflie 2.X):**
- Mass: 0.027 kg
- Inertia: Ixx = Iyy = 1.4e-5, Izz = 2.2e-5 kg·m²
- Max thrust: 0.81 N
- Max angular velocity: 3.0 rad/s

### 4. Test Suite Implementation ✅
**File:** `tests/test_mpc_controller.py` (402 lines)

**Test Infrastructure:**
- PyBullet simulation integration (CtrlAviary)
- State extraction from observations
- Quaternion to Euler angle conversion
- Reference trajectory generation
- Control allocation (MPC → motor RPMs)

**Test Cases:**
1. **Hover Stability Test**
   - Duration: 10 seconds
   - Target: [0, 0, 1.0] meters
   - Metrics: Position RMSE, max error, final error
   - Pass criteria:
     - RMSE < 0.1 m
     - Final error < 0.05 m
     - Avg solve time < 20 ms

**Results Visualization:**
- Position tracking plot (X, Y, Z)
- Velocity plot (Vx, Vy, Vz)
- Control inputs plot (thrust, angular rates)
- Saved to: `results/phase_02/mpc_hover_test.png`

**Performance Metrics:**
- MPC solve time (average, max)
- Success rate
- Position tracking error (RMSE, max, final)
- Real-time factor

---

## Technical Architecture

### Control Loop
```
1. Observe current state x from PyBullet
2. Generate reference trajectory x_ref
3. Solve MPC optimization: min J subject to dynamics & constraints
4. Extract first control u_0 from optimal trajectory
5. Convert u_0 to motor RPMs
6. Apply RPMs to simulator
7. Repeat at 48 Hz
```

### MPC Optimization Problem
```
Variables:
  X = [x_0, x_1, ..., x_N]  ∈ R^{12×(N+1)}
  U = [u_0, u_1, ..., u_{N-1}]  ∈ R^{4×N}

Minimize:
  Σ_{k=0}^{N-1} [(x_k - x_ref_k)' Q (x_k - x_ref_k) + u_k' R u_k]
  + (x_N - x_ref_N)' Q_terminal (x_N - x_ref_N)

Subject to:
  x_0 = x_current
  x_{k+1} = RK4(x_k, u_k, dt)  for k = 0...N-1
  u_min ≤ u_k ≤ u_max  for k = 0...N-1
```

### Integration with Simulator
```python
obs = env.reset()  # Get initial observation
state = get_state_from_obs(obs)  # Extract 12D state
ref_traj = generate_reference_trajectory(state, target)  # Create reference
control = mpc.compute_control(state, ref_traj)  # Solve MPC
rpms = control_to_rpm(control)  # Convert to motor commands
obs = env.step(rpms)  # Apply and advance simulation
```

---

## Dependencies

**Installed and Verified:**
- CasADi 3.7.2 ✅
- PyYAML 6.0.3 ✅
- NumPy 1.26.4 ✅
- Matplotlib 3.10.1 ✅
- gym-pybullet-drones 2.0.0 ✅
- PyBullet 3.2.7 ✅

---

## Current Status

### Testing Status
- MPC controller test is currently executing
- Test parameters:
  - Duration: 10 seconds
  - Target altitude: 1.0 m
  - Headless mode (no GUI)
  - Control frequency: 48 Hz
  - Physics frequency: 240 Hz

### Known Issues
- First MPC solve may take longer due to cold start
- IPOPT may require several iterations to converge initially
- Output buffering may delay console output

---

## Next Steps

### Immediate (Current Session)
1. ⏳ Complete hover stability test
2. ⏳ Analyze test results
3. ⏳ Generate performance plots
4. 📋 Tune MPC weights if needed

### Phase 2 Remaining Tasks
1. **Waypoint Tracking Test**
   - Multiple waypoints
   - Smooth transitions
   - Performance validation

2. **Trajectory Following Test**
   - Circular trajectory
   - Lemniscate (figure-8)
   - Velocity tracking

3. **Hyperparameter Tuning**
   - Adjust Q, R matrices
   - Optimize prediction horizon N
   - Fine-tune constraints

4. **Performance Optimization**
   - Reduce solve time
   - Improve convergence
   - Enhance robustness

5. **Documentation**
   - MPC formulation document
   - Performance benchmarks
   - Tuning guidelines

---

## Success Criteria for Phase 2 Completion

- [ ] Hover test passes (RMSE < 0.1m, solve time < 20ms)
- [ ] Waypoint tracking successful (smooth transitions)
- [ ] Trajectory following accurate
- [ ] MPC weights tuned for baseline performance
- [ ] Complete documentation
- [ ] Checkpoint file created
- [ ] Ready for Phase 3 (RL Integration)

---

## File Structure

```
rl_tuned_mpc/
├── src/
│   └── mpc/
│       ├── __init__.py
│       └── mpc_controller.py          # 530 lines, complete
├── configs/
│   └── mpc_crazyflie.yaml            # Complete configuration
├── tests/
│   └── test_mpc_controller.py        # 402 lines, comprehensive
├── results/
│   └── phase_02/                      # (to be created)
│       ├── mpc_hover_test.png
│       ├── test_results.json
│       └── performance_metrics.json
└── PHASE_02_PROGRESS.md              # This file
```

---

## Code Quality

- **Type Hints:** Extensive use of Python type annotations
- **Documentation:** Comprehensive docstrings for all classes and methods
- **Error Handling:** Robust exception handling with fallback control
- **Modularity:** Clean separation of concerns
- **Testability:** Easy to test individual components
- **Extensibility:** Ready for RL integration in Phase 3

---

## Performance Expectations

**Target Metrics:**
- Position RMSE: < 0.05 m
- Velocity RMSE: < 0.1 m/s
- Angle RMSE: < 5 degrees (0.087 rad)
- MPC solve time: < 20 ms (avg)
- Success rate: > 95%
- Real-time factor: > 0.5x

---

## Integration with RL (Phase 3 Preview)

The MPC controller is designed for RL integration:

```python
# RL agent will tune these weights
mpc.update_weights(Q_new, R_new, Q_terminal_new)

# RL reward based on tracking performance
reward = -position_error - velocity_error - control_effort

# RL state includes MPC statistics
rl_state = [tracking_error, solve_time, convergence_rate, ...]
```

---

## References

- **Phase 1 Report:** `docs/phase_01_simulator_selection/SIMULATOR_REPORT.md`
- **Development Roadmap:** `Project docx/DEVELOPMENT_ROADMAP_DETAILED.md`
- **Configuration:** `configs/mpc_crazyflie.yaml`
- **Code:** `src/mpc/mpc_controller.py`

---

*Last Updated: 2025-11-20*
*Status: Implementation complete, testing in progress*
