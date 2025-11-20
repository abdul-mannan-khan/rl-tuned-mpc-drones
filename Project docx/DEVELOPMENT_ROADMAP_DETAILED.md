# Step-by-Step Development Roadmap
## RL-Enhanced MPC for Multi-Drone Systems

**Author:** Dr. Abdul Manan Khan  
**Project:** Incremental development from basic simulation to full RL-MPC framework  
**Date:** November 2025  
**Version:** 1.0

---

## Development Philosophy

### Core Principles
1. **Incremental Progress:** Build and validate one component at a time
2. **Test Early, Test Often:** Verify each phase before moving forward
3. **Document Everything:** Each phase has clear entry/exit criteria and documentation
4. **Resumable Progress:** Can pause and resume at any checkpoint
5. **No Moving Back:** Only proceed when current phase is 100% validated

### Project Structure
```
rl_mpc_drones/
├── README.md                          # Project overview
├── PROGRESS_LOG.md                    # Development journal
├── docs/
│   ├── phase_01_simulator_selection/
│   ├── phase_02_pid_controller/
│   ├── phase_03_obstacle_avoidance/
│   ├── phase_04_mpc_implementation/
│   ├── phase_05_multi_drone_validation/
│   ├── phase_06_rl_integration/
│   └── phase_07_transfer_learning/
├── configs/
│   ├── drone_crazyflie.yaml
│   ├── drone_racing.yaml
│   ├── drone_generic.yaml
│   └── drone_heavy.yaml
├── src/
│   ├── simulators/
│   ├── controllers/
│   ├── mpc/
│   ├── rl_agents/
│   └── utils/
├── tests/
├── results/
│   ├── phase_01/
│   ├── phase_02/
│   └── ...
└── checkpoints/
    └── phase_XX_checkpoint.yaml
```

---

## Phase 1: Simulator Selection & Validation
**Duration:** 2-3 days  
**Goal:** Select and validate the best simulator for our requirements

### 1.1 Simulators to Evaluate

#### Option A: PyBullet (gym-pybullet-drones)
**Pros:**
- Fast, lightweight
- Good Python integration
- Free and open-source
- Active community support
- Realistic physics

**Cons:**
- Basic visualization
- Limited sensor models
- Less realistic rendering

**Test Setup:**
```bash
pip install gym-pybullet-drones
pip install pybullet
```

#### Option B: Webots
**Pros:**
- Excellent visualization
- Professional-grade physics
- Rich sensor suite
- Good documentation
- Used in your published work

**Cons:**
- Larger installation
- Slower than PyBullet
- More complex setup

**Test Setup:**
- Download from https://cyberbotics.com
- Install Python API

#### Option C: Gazebo + PX4
**Pros:**
- Industry standard
- High fidelity
- ROS integration

**Cons:**
- Very complex setup
- Slow simulation
- Overkill for our needs

### 1.2 Evaluation Criteria

Create test script: `tests/test_simulator_capabilities.py`

```python
"""
Simulator Capability Test Suite
Tests each simulator against project requirements
"""

class SimulatorTest:
    def __init__(self, simulator_name):
        self.simulator = simulator_name
        self.results = {}
    
    def test_basic_flight(self):
        """Test basic hover and waypoint navigation"""
        # Launch simulator
        # Command hover at 2m
        # Measure stability
        pass
    
    def test_physics_accuracy(self):
        """Verify realistic dynamics"""
        # Drop test (free fall)
        # Thrust response
        # Inertia effects
        pass
    
    def test_computational_speed(self):
        """Measure real-time factor"""
        # Run 60 seconds simulation
        # Measure wall-clock time
        # Calculate real-time factor (target: >1.5x)
        pass
    
    def test_state_access(self):
        """Verify we can get all needed states"""
        # Position (x, y, z)
        # Velocity (vx, vy, vz)
        # Orientation (roll, pitch, yaw)
        # Angular velocity (p, q, r)
        pass
    
    def test_control_interface(self):
        """Verify we can send control commands"""
        # Thrust commands
        # Angular rate commands
        # Response time (<20ms)
        pass
    
    def test_visualization(self):
        """Check visual feedback quality"""
        # 3D rendering
        # Trajectory plotting
        # Real-time display
        pass
    
    def generate_report(self):
        """Create comparison report"""
        # Save results to results/phase_01/
        pass
```

### 1.3 Decision Matrix

| Requirement | Weight | PyBullet | Webots | Gazebo |
|-------------|--------|----------|--------|--------|
| Physics accuracy | 10 | 8 | 9 | 10 |
| Computation speed | 9 | 10 | 7 | 4 |
| Python integration | 9 | 10 | 8 | 6 |
| Visualization | 7 | 6 | 9 | 8 |
| Setup complexity | 8 | 9 | 7 | 4 |
| Documentation | 7 | 8 | 9 | 9 |
| **Total Score** | - | **8.6** | **8.1** | **6.7** |

### 1.4 Phase 1 Deliverables

**Documentation: `docs/phase_01_simulator_selection/SIMULATOR_REPORT.md`**

```markdown
# Simulator Selection Report

## Executive Summary
Selected simulator: [PyBullet/Webots/Gazebo]
Rationale: [2-3 sentences]

## Test Results
### Basic Flight Test
- Hover stability: [PASS/FAIL]
- Waypoint tracking: [PASS/FAIL]
- Video: results/phase_01/basic_flight.mp4

### Physics Accuracy
- Free fall test: [results]
- Thrust response: [results]
- Comparison with analytical model: [±X% error]

### Computational Performance
- Real-time factor: [X.XX]
- CPU usage: [X%]
- Memory usage: [X MB]

### State Access
- All 12 states accessible: [YES/NO]
- Update frequency: [X Hz]
- Latency: [X ms]

## Final Decision
Simulator: [NAME]
Installation verified: [YES]
Ready for Phase 2: [YES]
```

**Checkpoint File: `checkpoints/phase_01_checkpoint.yaml`**

```yaml
phase: 1
name: "Simulator Selection & Validation"
status: COMPLETED
completion_date: "2025-11-XX"
selected_simulator: "pybullet"  # or webots
tested_capabilities:
  - basic_flight: PASS
  - physics_accuracy: PASS
  - computational_speed: PASS
  - state_access: PASS
  - control_interface: PASS
  - visualization: PASS
next_phase: 2
notes: |
  PyBullet selected for speed and Python integration.
  Real-time factor: 2.3x on Intel i7-10700K
  All 12 states accessible at 240Hz
```

### 1.5 Exit Criteria (Must Pass All)
- [ ] Simulator installed and running
- [ ] Basic flight test passes (hover + waypoint)
- [ ] All 12 drone states accessible
- [ ] Control commands responsive (<20ms)
- [ ] Real-time factor >1.5x
- [ ] Documentation complete
- [ ] Checkpoint file created

---

## Phase 2: PID Controller Implementation
**Duration:** 2-3 days  
**Goal:** Implement and validate basic PID controller for single drone

### 2.1 Entry Requirements
- Phase 1 checkpoint complete
- Simulator operational
- No blocking issues from Phase 1

### 2.2 Implementation Tasks

#### Task 2.1: PID Controller Class

**File: `src/controllers/pid_controller.py`**

```python
"""
PID Controller for Quadrotor UAV
Cascaded control: Position → Velocity → Attitude → Angular Rate
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class PIDGains:
    """PID gain configuration"""
    kp: float
    ki: float
    kd: float
    
class QuadrotorPIDController:
    """
    Cascaded PID controller for quadrotor
    
    Control Architecture:
    1. Position PID → Desired Velocity
    2. Velocity PID → Desired Attitude
    3. Attitude PID → Angular Rate Commands
    """
    
    def __init__(self, config_path):
        """Load PID gains from config file"""
        self.load_config(config_path)
        self.reset()
    
    def load_config(self, config_path):
        """Load gains from YAML"""
        # Position control
        self.pos_gains = {
            'x': PIDGains(kp=2.0, ki=0.0, kd=1.0),
            'y': PIDGains(kp=2.0, ki=0.0, kd=1.0),
            'z': PIDGains(kp=4.0, ki=2.0, kd=2.0)
        }
        
        # Attitude control  
        self.att_gains = {
            'roll': PIDGains(kp=6.0, ki=3.0, kd=0.5),
            'pitch': PIDGains(kp=6.0, ki=3.0, kd=0.5),
            'yaw': PIDGains(kp=4.0, ki=0.0, kd=0.0)
        }
        
    def reset(self):
        """Reset integral terms"""
        self.integral_pos = np.zeros(3)
        self.integral_att = np.zeros(3)
        self.last_error_pos = np.zeros(3)
        self.last_error_att = np.zeros(3)
        
    def compute_control(self, current_state, desired_state, dt):
        """
        Compute control commands
        
        Args:
            current_state: dict with keys [pos, vel, att, ang_vel]
            desired_state: dict with keys [pos, vel, yaw]
            dt: timestep
            
        Returns:
            control: [thrust, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd]
        """
        # Step 1: Position control
        pos_error = desired_state['pos'] - current_state['pos']
        vel_desired = self._position_control(pos_error, dt)
        
        # Step 2: Velocity control  
        vel_error = vel_desired - current_state['vel']
        att_desired = self._velocity_control(vel_error, current_state['att'][2])
        
        # Step 3: Attitude control
        att_error = att_desired - current_state['att']
        ang_vel_cmd = self._attitude_control(att_error, dt)
        
        # Step 4: Thrust from altitude control
        thrust = self._altitude_control(pos_error[2], vel_error[2], dt)
        
        return np.array([thrust, ang_vel_cmd[0], ang_vel_cmd[1], ang_vel_cmd[2]])
    
    def _position_control(self, error, dt):
        """Position PID → Desired velocity"""
        self.integral_pos += error * dt
        derivative = (error - self.last_error_pos) / dt
        self.last_error_pos = error
        
        vel_cmd = np.zeros(3)
        for i, axis in enumerate(['x', 'y', 'z']):
            gains = self.pos_gains[axis]
            vel_cmd[i] = (gains.kp * error[i] + 
                         gains.ki * self.integral_pos[i] +
                         gains.kd * derivative[i])
        
        # Saturate
        vel_cmd = np.clip(vel_cmd, -2.0, 2.0)
        return vel_cmd
    
    def _velocity_control(self, vel_error, current_yaw):
        """Velocity PID → Desired attitude"""
        # Simplified: velocity error → desired roll/pitch
        roll_desired = vel_error[1]  # y velocity → roll
        pitch_desired = -vel_error[0]  # x velocity → pitch (negative)
        
        # Saturate
        roll_desired = np.clip(roll_desired, -0.3, 0.3)  # ±17 deg
        pitch_desired = np.clip(pitch_desired, -0.3, 0.3)
        
        return np.array([roll_desired, pitch_desired, current_yaw])
    
    def _attitude_control(self, att_error, dt):
        """Attitude PID → Angular rate commands"""
        self.integral_att += att_error * dt
        derivative = (att_error - self.last_error_att) / dt
        self.last_error_att = att_error
        
        ang_vel_cmd = np.zeros(3)
        for i, axis in enumerate(['roll', 'pitch', 'yaw']):
            gains = self.att_gains[axis]
            ang_vel_cmd[i] = (gains.kp * att_error[i] +
                             gains.ki * self.integral_att[i] +
                             gains.kd * derivative[i])
        
        # Saturate
        ang_vel_cmd = np.clip(ang_vel_cmd, -3.0, 3.0)  # rad/s
        return ang_vel_cmd
    
    def _altitude_control(self, pos_error_z, vel_error_z, dt):
        """Altitude PID → Thrust command"""
        gains = self.pos_gains['z']
        
        thrust = (gains.kp * pos_error_z +
                 gains.ki * self.integral_pos[2] +
                 gains.kd * vel_error_z)
        
        # Add hover thrust (varies by drone mass)
        hover_thrust = 9.81  # Will be platform-specific
        thrust += hover_thrust
        
        # Saturate
        thrust = np.clip(thrust, 0.0, 20.0)
        return thrust
```

#### Task 2.2: Test Script

**File: `tests/test_pid_controller.py`**

```python
"""
PID Controller Test Suite
Tests: hover, step response, waypoint tracking
"""

import numpy as np
import matplotlib.pyplot as plt
from src.controllers.pid_controller import QuadrotorPIDController
from src.simulators.drone_env import DroneEnv

class PIDControllerTest:
    def __init__(self, simulator="pybullet"):
        self.env = DroneEnv(simulator)
        self.controller = QuadrotorPIDController("configs/pid_default.yaml")
        
    def test_hover(self, duration=10.0, target_height=2.0):
        """
        Test 1: Hover Stability
        Target: ±0.05m position error
        """
        print("\n=== Test 1: Hover Stability ===")
        
        results = {
            'time': [],
            'position': [],
            'velocity': [],
            'control': []
        }
        
        state = self.env.reset()
        desired = {'pos': np.array([0, 0, target_height]), 
                   'vel': np.zeros(3),
                   'yaw': 0.0}
        
        dt = 0.01
        t = 0
        while t < duration:
            # Compute control
            control = self.controller.compute_control(state, desired, dt)
            
            # Apply to simulator
            state = self.env.step(control)
            
            # Log
            results['time'].append(t)
            results['position'].append(state['pos'].copy())
            results['velocity'].append(state['vel'].copy())
            results['control'].append(control.copy())
            
            t += dt
        
        # Analyze
        positions = np.array(results['position'])
        pos_error = np.abs(positions - desired['pos'])
        
        max_error = np.max(pos_error[:, 2])  # Z-axis
        rmse = np.sqrt(np.mean(pos_error[:, 2]**2))
        
        print(f"Max altitude error: {max_error:.3f} m")
        print(f"RMSE altitude: {rmse:.3f} m")
        print(f"Status: {'PASS' if max_error < 0.1 else 'FAIL'}")
        
        # Plot
        self._plot_hover_results(results, desired)
        
        return max_error < 0.1
    
    def test_step_response(self):
        """
        Test 2: Step Response
        Command 2m altitude step, measure overshoot and settling time
        """
        print("\n=== Test 2: Step Response ===")
        
        # TODO: Implement
        pass
    
    def test_waypoint_tracking(self, waypoints):
        """
        Test 3: Waypoint Navigation
        Follow sequence of waypoints
        """
        print("\n=== Test 3: Waypoint Tracking ===")
        
        # TODO: Implement
        pass
    
    def _plot_hover_results(self, results, desired):
        """Generate hover test plots"""
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        time = np.array(results['time'])
        pos = np.array(results['position'])
        vel = np.array(results['velocity'])
        
        # Position
        axes[0].plot(time, pos[:, 0], label='X')
        axes[0].plot(time, pos[:, 1], label='Y')
        axes[0].plot(time, pos[:, 2], label='Z')
        axes[0].axhline(desired['pos'][2], color='r', linestyle='--', label='Target Z')
        axes[0].set_ylabel('Position (m)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Velocity
        axes[1].plot(time, vel[:, 0], label='Vx')
        axes[1].plot(time, vel[:, 1], label='Vy')
        axes[1].plot(time, vel[:, 2], label='Vz')
        axes[1].set_ylabel('Velocity (m/s)')
        axes[1].legend()
        axes[1].grid(True)
        
        # Control
        control = np.array(results['control'])
        axes[2].plot(time, control[:, 0], label='Thrust')
        axes[2].set_ylabel('Thrust (N)')
        axes[2].set_xlabel('Time (s)')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/phase_02/hover_test.png', dpi=150)
        plt.show()

if __name__ == "__main__":
    tester = PIDControllerTest()
    
    # Run tests
    hover_pass = tester.test_hover(duration=10.0, target_height=2.0)
    
    # Save results
    with open('results/phase_02/test_results.txt', 'w') as f:
        f.write(f"Hover Test: {'PASS' if hover_pass else 'FAIL'}\n")
```

### 2.3 Phase 2 Deliverables

**Documentation: `docs/phase_02_pid_controller/PID_IMPLEMENTATION.md`**

```markdown
# PID Controller Implementation Report

## Overview
Cascaded PID controller for quadrotor control

## Architecture
1. Position Control Loop (10 Hz)
2. Velocity Control Loop (50 Hz)
3. Attitude Control Loop (100 Hz)

## Tuning Process
### Initial Gains (Ziegler-Nichols)
[Document how gains were selected]

### Test Results
#### Hover Test
- Max position error: X.XXX m
- RMSE: X.XXX m
- Settling time: X.X s
- Overshoot: X.X%
- **Status: PASS/FAIL**

#### Step Response
[Results and plots]

#### Waypoint Tracking
[Results and plots]

## Tuning Recommendations
[Lessons learned, what worked, what didn't]

## Next Steps
Ready for Phase 3: Obstacle Avoidance
```

**Checkpoint: `checkpoints/phase_02_checkpoint.yaml`**

```yaml
phase: 2
name: "PID Controller Implementation"
status: COMPLETED
completion_date: "2025-11-XX"
achievements:
  - pid_controller_implemented: true
  - hover_test: PASS
  - step_response_test: PASS
  - waypoint_tracking_test: PASS
metrics:
  hover_max_error: 0.045  # meters
  hover_rmse: 0.023
  settling_time: 2.3  # seconds
next_phase: 3
code_files:
  - src/controllers/pid_controller.py
  - tests/test_pid_controller.py
  - configs/pid_default.yaml
```

### 2.4 Exit Criteria
- [ ] PID controller class implemented
- [ ] Hover test passes (error <0.1m)
- [ ] Step response acceptable (overshoot <20%)
- [ ] Waypoint tracking functional
- [ ] All tests documented with plots
- [ ] Code committed to repository
- [ ] Checkpoint file created

---

## Phase 3: Obstacle Avoidance
**Duration:** 3-4 days  
**Goal:** Add obstacles and implement collision-free navigation

### 3.1 Entry Requirements
- Phase 2 checkpoint complete
- PID controller working reliably
- Drone can track waypoints

### 3.2 Implementation Tasks

#### Task 3.1: Obstacle Environment

**File: `src/environments/obstacle_course.py`**

```python
"""
Obstacle course environment
Creates static obstacles for navigation testing
"""

class ObstacleCourse:
    def __init__(self, course_type="simple"):
        self.course_type = course_type
        self.obstacles = []
        self.create_course()
    
    def create_course(self):
        """Create obstacle configuration"""
        if self.course_type == "simple":
            # 3 cylindrical obstacles
            self.obstacles = [
                {'type': 'cylinder', 'pos': [2, 0, 1.5], 'radius': 0.3, 'height': 3},
                {'type': 'cylinder', 'pos': [4, 2, 1.5], 'radius': 0.3, 'height': 3},
                {'type': 'cylinder', 'pos': [6, 0, 1.5], 'radius': 0.3, 'height': 3}
            ]
        elif self.course_type == "complex":
            # Multiple obstacles at different heights
            pass
    
    def check_collision(self, drone_pos, safety_margin=0.5):
        """Check if drone position collides with obstacles"""
        for obs in self.obstacles:
            if obs['type'] == 'cylinder':
                # 2D distance check (x, y)
                dist_xy = np.linalg.norm(drone_pos[:2] - obs['pos'][:2])
                if dist_xy < (obs['radius'] + safety_margin):
                    # Check height
                    if 0 < drone_pos[2] < obs['height']:
                        return True
        return False
    
    def get_safe_waypoints(self, start, goal):
        """
        Generate collision-free waypoint sequence
        Uses simple potential field method
        """
        # TODO: Implement path planning
        pass
```

#### Task 3.2: Path Planning Integration

**File: `src/planning/simple_planner.py`**

```python
"""
Simple path planner for obstacle avoidance
Uses RRT or potential fields
"""

class SimplePlanner:
    def __init__(self, obstacle_course):
        self.obstacles = obstacle_course
    
    def plan_path(self, start, goal, altitude=2.0):
        """
        Plan collision-free path
        
        Method: Simple waypoint insertion to avoid obstacles
        """
        waypoints = [start]
        
        # Check direct path
        if not self._path_collides(start, goal):
            waypoints.append(goal)
            return waypoints
        
        # Insert intermediate waypoints
        for obs in self.obstacles.obstacles:
            # Go around obstacle
            detour = self._compute_detour(start, goal, obs, altitude)
            waypoints.extend(detour)
        
        waypoints.append(goal)
        return waypoints
    
    def _path_collides(self, p1, p2):
        """Check if straight line path collides"""
        # Sample points along path
        samples = 50
        for i in range(samples):
            alpha = i / samples
            point = p1 + alpha * (p2 - p1)
            if self.obstacles.check_collision(point):
                return True
        return False
    
    def _compute_detour(self, start, goal, obstacle, altitude):
        """Compute waypoint to go around obstacle"""
        # Simple: offset perpendicular to line
        direction = goal - start
        perpendicular = np.array([-direction[1], direction[0], 0])
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        
        offset = obstacle['radius'] + 1.0  # 1m clearance
        detour_point = obstacle['pos'] + perpendicular * offset
        detour_point[2] = altitude
        
        return [detour_point]
```

### 3.3 Test Cases

**File: `tests/test_obstacle_avoidance.py`**

```python
"""
Obstacle avoidance test suite
"""

class ObstacleAvoidanceTest:
    def test_simple_course(self):
        """
        Test: Navigate through 3 obstacles
        Success: Reach goal without collision
        """
        pass
    
    def test_complex_course(self):
        """
        Test: Navigate through dense obstacles
        """
        pass
    
    def test_collision_detection(self):
        """
        Test: Verify collision detection works
        """
        pass
```

### 3.4 Phase 3 Deliverables

**Documentation: `docs/phase_03_obstacle_avoidance/OBSTACLE_REPORT.md`**

```markdown
# Obstacle Avoidance Implementation Report

## Environment Setup
- Obstacle type: Cylinders
- Number of obstacles: 3 (simple), 8 (complex)
- Safety margin: 0.5m

## Path Planning Method
Algorithm: [RRT/Potential Field/Waypoint Insertion]

## Test Results
### Simple Course
- Success rate: X/10 runs
- Average completion time: X.X s
- Closest approach: X.XX m
- **Status: PASS/FAIL**

### Complex Course  
[Results]

## Issues Encountered
[Document any problems and solutions]

## Videos
- simple_course.mp4
- complex_course.mp4
```

**Checkpoint: `checkpoints/phase_03_checkpoint.yaml`**

```yaml
phase: 3
name: "Obstacle Avoidance"
status: COMPLETED
completion_date: "2025-11-XX"
achievements:
  - obstacle_environment: true
  - path_planning: true
  - collision_detection: true
  - simple_course_test: PASS
  - complex_course_test: PASS
metrics:
  success_rate_simple: 10/10
  success_rate_complex: 8/10
  min_clearance: 0.52  # meters
next_phase: 4
```

### 3.5 Exit Criteria
- [ ] Obstacles rendered in simulation
- [ ] Collision detection functional
- [ ] Path planner implemented
- [ ] Simple course: 100% success rate
- [ ] Complex course: >80% success rate
- [ ] Video recordings saved
- [ ] Documentation complete

---

## Phase 4: MPC Implementation
**Duration:** 5-7 days  
**Goal:** Implement and validate nonlinear MPC controller

### 4.1 Entry Requirements
- Phase 3 checkpoint complete
- Drone can navigate obstacles with PID
- Simulation stable and reliable

### 4.2 Implementation Tasks

#### Task 4.1: MPC Controller Core

**File: `src/mpc/mpc_controller.py`**

```python
"""
Nonlinear MPC Controller for Quadrotor
Uses CasADi for symbolic computation and IPOPT for optimization
"""

import casadi as ca
import numpy as np

class MPCController:
    """
    Nonlinear MPC for quadrotor trajectory tracking
    
    State: [px, py, pz, vx, vy, vz, roll, pitch, yaw, p, q, r]
    Control: [thrust, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd]
    """
    
    def __init__(self, config_path):
        """Initialize MPC with configuration"""
        self.load_config(config_path)
        self.setup_optimization()
    
    def load_config(self, config_path):
        """Load MPC parameters from YAML"""
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Prediction horizon
        self.N = config['mpc']['prediction_horizon']  # 10
        self.dt = config['mpc']['timestep']  # 0.02
        
        # Weight matrices (will be tuned by RL)
        Q_diag = config['mpc']['Q']  # 12-dimensional
        R_diag = config['mpc']['R']  # 4-dimensional
        
        self.Q = np.diag(Q_diag)
        self.R = np.diag(R_diag)
        
        # Constraints
        self.u_min = np.array(config['mpc']['u_min'])
        self.u_max = np.array(config['mpc']['u_max'])
        
        # Drone parameters (platform-specific)
        self.mass = config['drone']['mass']
        self.Ixx = config['drone']['inertia']['Ixx']
        self.Iyy = config['drone']['inertia']['Iyy']
        self.Izz = config['drone']['inertia']['Izz']
    
    def setup_optimization(self):
        """
        Setup CasADi optimization problem
        Creates symbolic variables and constraints
        """
        # State and control dimensions
        nx = 12
        nu = 4
        
        # Symbolic variables
        x = ca.SX.sym('x', nx)  # state
        u = ca.SX.sym('u', nu)  # control
        
        # Drone dynamics (symbolic)
        x_dot = self._dynamics(x, u)
        
        # Objective function (stage cost)
        x_ref = ca.SX.sym('x_ref', nx)
        L = ca.mtimes([(x - x_ref).T, self.Q, (x - x_ref)]) + \
            ca.mtimes([u.T, self.R, u])
        
        # Create CasADi functions
        self.f_dynamics = ca.Function('f', [x, u], [x_dot])
        self.f_cost = ca.Function('L', [x, u, x_ref], [L])
        
        # Setup NLP
        self._setup_nlp()
    
    def _dynamics(self, x, u):
        """
        Quadrotor dynamics (continuous time)
        
        State: [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
        Control: [T, p_cmd, q_cmd, r_cmd]
        """
        # Extract states
        pos = x[0:3]
        vel = x[3:6]
        att = x[6:9]  # Euler angles
        omega = x[9:12]  # Angular velocity
        
        # Extract controls
        thrust = u[0]
        omega_cmd = u[1:4]
        
        # Position derivatives (velocity)
        pos_dot = vel
        
        # Velocity derivatives (Newton's 2nd law)
        phi, theta, psi = att[0], att[1], att[2]
        
        # Rotation matrix (body to world)
        # Simplified for small angles
        vel_dot = ca.vertcat(
            (thrust / self.mass) * (ca.sin(psi)*ca.sin(phi) + ca.cos(psi)*ca.sin(theta)*ca.cos(phi)),
            (thrust / self.mass) * (-ca.cos(psi)*ca.sin(phi) + ca.sin(psi)*ca.sin(theta)*ca.cos(phi)),
            (thrust / self.mass) * ca.cos(theta)*ca.cos(phi) - 9.81
        )
        
        # Attitude dynamics (simplified)
        att_dot = omega  # Approximation for small angles
        
        # Angular acceleration (assume fast inner loop)
        omega_dot = 10.0 * (omega_cmd - omega)  # First-order model
        
        return ca.vertcat(pos_dot, vel_dot, att_dot, omega_dot)
    
    def _setup_nlp(self):
        """Setup the NLP for MPC optimization"""
        nx = 12
        nu = 4
        
        # Decision variables: [u_0, ..., u_{N-1}, x_1, ..., x_N]
        n_vars = nu * self.N + nx * (self.N + 1)
        
        # Create Opti object
        self.opti = ca.Opti()
        
        # Decision variables
        self.U = self.opti.variable(nu, self.N)
        self.X = self.opti.variable(nx, self.N + 1)
        
        # Parameters
        self.X0 = self.opti.parameter(nx)  # Initial state
        self.X_ref = self.opti.parameter(nx, self.N + 1)  # Reference trajectory
        
        # Objective
        obj = 0
        for k in range(self.N):
            # Stage cost
            x_err = self.X[:, k] - self.X_ref[:, k]
            obj += ca.mtimes([x_err.T, self.Q, x_err])
            obj += ca.mtimes([self.U[:, k].T, self.R, self.U[:, k]])
        
        # Terminal cost
        x_err_final = self.X[:, self.N] - self.X_ref[:, self.N]
        obj += ca.mtimes([x_err_final.T, self.Q, x_err_final])
        
        self.opti.minimize(obj)
        
        # Dynamics constraints (RK4 integration)
        for k in range(self.N):
            x_next = self._rk4_step(self.X[:, k], self.U[:, k], self.dt)
            self.opti.subject_to(self.X[:, k+1] == x_next)
        
        # Initial condition
        self.opti.subject_to(self.X[:, 0] == self.X0)
        
        # Control constraints
        for k in range(self.N):
            self.opti.subject_to(self.U[:, k] >= self.u_min)
            self.opti.subject_to(self.U[:, k] <= self.u_max)
        
        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-6
        }
        self.opti.solver('ipopt', opts)
    
    def _rk4_step(self, x, u, dt):
        """RK4 integration step"""
        k1 = self.f_dynamics(x, u)
        k2 = self.f_dynamics(x + dt/2 * k1, u)
        k3 = self.f_dynamics(x + dt/2 * k2, u)
        k4 = self.f_dynamics(x + dt * k3, u)
        
        x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return x_next
    
    def compute_control(self, current_state, reference_trajectory):
        """
        Solve MPC optimization problem
        
        Args:
            current_state: Current 12D state
            reference_trajectory: Reference states for N+1 steps
            
        Returns:
            optimal_control: First control input from solution
            solve_time: Time taken to solve (ms)
        """
        import time
        
        # Set parameters
        self.opti.set_value(self.X0, current_state)
        self.opti.set_value(self.X_ref, reference_trajectory)
        
        # Solve
        t_start = time.time()
        try:
            sol = self.opti.solve()
            solve_time = (time.time() - t_start) * 1000  # ms
            
            # Extract first control
            u_opt = sol.value(self.U[:, 0])
            
            return u_opt, solve_time, True
        
        except Exception as e:
            print(f"MPC solve failed: {e}")
            # Return safe fallback control
            u_safe = np.array([self.mass * 9.81, 0, 0, 0])
            return u_safe, 0, False
    
    def update_weights(self, Q_new, R_new):
        """
        Update MPC weight matrices
        Called by RL agent during training
        """
        self.Q = np.diag(Q_new)
        self.R = np.diag(R_new)
        
        # Re-setup optimization with new weights
        self.setup_optimization()
```

#### Task 4.2: MPC Configuration

**File: `configs/mpc_crazyflie.yaml`**

```yaml
# MPC Configuration for Crazyflie 2.X

mpc:
  prediction_horizon: 10
  control_horizon: 10
  timestep: 0.02  # 50 Hz
  
  # Weight matrices (initial - will be tuned by RL)
  # Q: [px, py, pz, vx, vy, vz, roll, pitch, yaw, p, q, r]
  Q: [100, 100, 100,    # Position weights
      10, 10, 10,       # Velocity weights
      50, 50, 20,       # Orientation weights
      1, 1, 1]          # Angular velocity weights
  
  # R: [thrust, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd]
  R: [1, 1, 1, 1]
  
  # Control constraints
  u_min: [0.0, -3.0, -3.0, -3.0]  # [thrust_min, ang_vel_min...]
  u_max: [0.6, 3.0, 3.0, 3.0]     # Crazyflie limits

drone:
  name: "Crazyflie 2.X"
  mass: 0.027  # kg
  inertia:
    Ixx: 1.4e-5
    Iyy: 1.4e-5
    Izz: 2.2e-5
```

#### Task 4.3: MPC Testing

**File: `tests/test_mpc_controller.py`**

```python
"""
MPC Controller Test Suite
Comprehensive testing of MPC performance
"""

class MPCTest:
    def __init__(self, platform="crazyflie"):
        self.platform = platform
        self.env = DroneEnv()
        self.mpc = MPCController(f"configs/mpc_{platform}.yaml")
        
    def test_hover(self, duration=10.0):
        """
        Test: MPC hover stability
        Metrics: Position RMSE, max error, solve time
        """
        print(f"\n=== MPC Hover Test ({self.platform}) ===")
        
        results = {
            'time': [],
            'position': [],
            'position_error': [],
            'velocity': [],
            'attitude': [],
            'control': [],
            'solve_time': [],
            'solve_success': []
        }
        
        # Reset
        state = self.env.reset()
        target = np.array([0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        # Reference trajectory (constant hover)
        ref_traj = np.tile(target, (self.mpc.N + 1, 1)).T
        
        t = 0
        dt = 0.02
        while t < duration:
            # Get current state (12D)
            x_current = self._get_state_vector(state)
            
            # Compute MPC control
            u_opt, solve_time, success = self.mpc.compute_control(
                x_current, ref_traj
            )
            
            # Apply control
            state = self.env.step(u_opt)
            
            # Log results
            results['time'].append(t)
            results['position'].append(state['pos'].copy())
            results['position_error'].append(
                np.linalg.norm(state['pos'] - target[:3])
            )
            results['velocity'].append(state['vel'].copy())
            results['attitude'].append(state['att'].copy())
            results['control'].append(u_opt.copy())
            results['solve_time'].append(solve_time)
            results['solve_success'].append(success)
            
            t += dt
        
        # Analyze results
        self._analyze_hover(results, target)
        self._plot_hover_results(results, target)
        
        return results
    
    def test_circular_trajectory(self, radius=2.0, period=20.0):
        """
        Test: Circular trajectory tracking
        Primary metric for RL training
        """
        print(f"\n=== MPC Circular Tracking ({self.platform}) ===")
        
        results = {'time': [], 'position': [], 'reference': [], 
                   'position_error': [], 'solve_time': []}
        
        state = self.env.reset()
        
        duration = period
        t = 0
        dt = 0.02
        
        while t < duration:
            # Generate reference trajectory
            ref_traj = self._generate_circular_reference(
                t, radius, period, self.mpc.N
            )
            
            # Get current state
            x_current = self._get_state_vector(state)
            
            # MPC control
            u_opt, solve_time, success = self.mpc.compute_control(
                x_current, ref_traj
            )
            
            # Apply
            state = self.env.step(u_opt)
            
            # Log
            results['time'].append(t)
            results['position'].append(state['pos'].copy())
            results['reference'].append(ref_traj[:3, 0].copy())
            results['position_error'].append(
                np.linalg.norm(state['pos'] - ref_traj[:3, 0])
            )
            results['solve_time'].append(solve_time)
            
            t += dt
        
        # Analyze
        rmse = self._compute_rmse(results)
        print(f"Tracking RMSE: {rmse:.3f} m")
        print(f"Mean solve time: {np.mean(results['solve_time']):.1f} ms")
        
        # Plot
        self._plot_circular_tracking(results)
        
        return rmse
    
    def test_all_states_tracking(self):
        """
        Test: Track all 12 states
        Generate report with RMSE for each state
        """
        print(f"\n=== Full State Tracking Test ===")
        
        # Define test trajectory (aggressive)
        # TODO: Implement
        
        pass
    
    def _analyze_hover(self, results, target):
        """Analyze hover performance"""
        pos_errors = np.array(results['position_error'])
        solve_times = np.array(results['solve_time'])
        
        rmse = np.sqrt(np.mean(pos_errors**2))
        max_error = np.max(pos_errors)
        mean_solve_time = np.mean(solve_times)
        max_solve_time = np.max(solve_times)
        
        print(f"\nHover Performance:")
        print(f"  Position RMSE: {rmse:.4f} m")
        print(f"  Max error: {max_error:.4f} m")
        print(f"  Mean solve time: {mean_solve_time:.1f} ms")
        print(f"  Max solve time: {max_solve_time:.1f} ms")
        print(f"  Status: {'PASS' if rmse < 0.1 else 'FAIL'}")
        
        # Save to file
        with open(f'results/phase_04/hover_{self.platform}.txt', 'w') as f:
            f.write(f"Platform: {self.platform}\n")
            f.write(f"RMSE: {rmse:.4f} m\n")
            f.write(f"Max Error: {max_error:.4f} m\n")
            f.write(f"Mean Solve Time: {mean_solve_time:.1f} ms\n")
    
    def _compute_rmse(self, results):
        """Compute trajectory tracking RMSE"""
        errors = np.array(results['position_error'])
        return np.sqrt(np.mean(errors**2))
    
    def _generate_circular_reference(self, t, radius, period, N):
        """Generate circular reference trajectory"""
        ref = np.zeros((12, N + 1))
        
        dt = 0.02
        for i in range(N + 1):
            t_future = t + i * dt
            omega = 2 * np.pi / period
            
            # Position
            ref[0, i] = radius * np.cos(omega * t_future)
            ref[1, i] = radius * np.sin(omega * t_future)
            ref[2, i] = 2.0  # Fixed altitude
            
            # Velocity
            ref[3, i] = -radius * omega * np.sin(omega * t_future)
            ref[4, i] = radius * omega * np.cos(omega * t_future)
            ref[5, i] = 0.0
            
            # Orientation (yaw tangent to circle)
            ref[8, i] = np.arctan2(ref[4, i], ref[3, i])
        
        return ref
    
    def _plot_hover_results(self, results, target):
        """Generate plots for hover test"""
        # Similar to PID test plots
        pass
    
    def _plot_circular_tracking(self, results):
        """Generate 3D trajectory plot"""
        fig = plt.figure(figsize=(12, 5))
        
        # 3D trajectory
        ax1 = fig.add_subplot(121, projection='3d')
        pos = np.array(results['position'])
        ref = np.array(results['reference'])
        
        ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', label='Actual')
        ax1.plot(ref[:, 0], ref[:, 1], ref[:, 2], 'r--', label='Reference')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.legend()
        ax1.set_title('3D Trajectory')
        
        # Tracking error over time
        ax2 = fig.add_subplot(122)
        ax2.plot(results['time'], results['position_error'])
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position Error (m)')
        ax2.set_title('Tracking Error')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results/phase_04/circular_{self.platform}.png', dpi=150)
        plt.show()
    
    def _get_state_vector(self, state_dict):
        """Convert state dict to 12D vector"""
        return np.concatenate([
            state_dict['pos'],
            state_dict['vel'],
            state_dict['att'],
            state_dict['ang_vel']
        ])

if __name__ == "__main__":
    # Test MPC on Crazyflie
    tester = MPCTest("crazyflie")
    
    # Run tests
    tester.test_hover(duration=10.0)
    rmse = tester.test_circular_trajectory(radius=2.0, period=20.0)
    
    print(f"\nPhase 4 Complete: MPC RMSE = {rmse:.3f} m")
```

### 4.4 Phase 4 Deliverables

**Documentation: `docs/phase_04_mpc_implementation/MPC_REPORT.md`**

```markdown
# MPC Implementation Report

## Controller Specification
- Prediction horizon: 10 steps (0.2s)
- Control frequency: 50 Hz
- Solver: IPOPT
- Integration: RK4

## Initial Weight Tuning
### Manual Tuning Process
[Document how initial Q, R were selected]

### Bryson's Rule Application
For Crazyflie (mass = 0.027 kg):
- Position acceptable error: 0.1 m → Q_pos = 100
- Velocity acceptable error: 0.5 m/s → Q_vel = 4
- [etc.]

## Test Results

### Platform: Crazyflie 2.X
#### Hover Test
- Position RMSE: X.XXX m
- Max error: X.XXX m
- Mean solve time: XX.X ms
- Status: PASS/FAIL

#### Circular Tracking
- RMSE: X.XX m
- Mean solve time: XX.X ms
- Status: PASS/FAIL

### State-by-State Tracking
| State | RMSE | Max Error | Status |
|-------|------|-----------|--------|
| px | X.XX m | X.XX m | PASS |
| py | X.XX m | X.XX m | PASS |
| pz | X.XX m | X.XX m | PASS |
| vx | X.XX m/s | X.XX m/s | PASS |
| ... | ... | ... | ... |

## Performance Analysis
### Computational Performance
- Mean solve time: XX ms
- Max solve time: XX ms
- Solve success rate: XX%
- Real-time capable: YES/NO

### Tracking Performance
- Overall good/acceptable/poor
- Best performance: [which states]
- Needs improvement: [which states]

## Issues & Solutions
[Document any problems encountered]

## Next Steps
- Ready for multi-platform validation (Phase 5)
- MPC parameters to be optimized by RL (Phase 6)
```

**Checkpoint: `checkpoints/phase_04_checkpoint.yaml`**

```yaml
phase: 4
name: "MPC Implementation"
status: COMPLETED
completion_date: "2025-11-XX"
platform: "crazyflie"
achievements:
  - mpc_controller_implemented: true
  - casadi_integration: true
  - ipopt_solver: true
  - hover_test: PASS
  - circular_tracking: PASS
  - state_tracking_validated: true
metrics:
  hover_rmse: 0.XXX  # meters
  circular_rmse: X.XX  # meters (baseline for RL)
  mean_solve_time: XX.X  # ms
  solve_success_rate: 0.XXX
next_phase: 5
code_files:
  - src/mpc/mpc_controller.py
  - configs/mpc_crazyflie.yaml
  - tests/test_mpc_controller.py
notes: |
  MPC working well on Crazyflie.
  Baseline RMSE: X.XX m (to be improved by RL)
  Solver time acceptable: XX ms avg
```

### 4.5 Exit Criteria
- [ ] MPC controller implemented with CasADi
- [ ] IPOPT solver working reliably
- [ ] Hover test passes (RMSE < 0.1m)
- [ ] Circular tracking functional (RMSE < 2.0m)
- [ ] All 12 states tracked
- [ ] Solve time < 50ms (real-time capable)
- [ ] Full documentation with plots
- [ ] Checkpoint created

---

## Phase 5: Multi-Platform MPC Validation
**Duration:** 3-4 days  
**Goal:** Test MPC on all 4 drone platforms

[CONTINUED IN NEXT RESPONSE DUE TO LENGTH...]
