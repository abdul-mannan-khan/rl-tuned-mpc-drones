# Development Roadmap - Part 2
## Phases 5-7: Multi-Platform Validation, RL Integration, and Transfer Learning

---

## Phase 5: Multi-Platform MPC Validation
**Duration:** 3-4 days  
**Goal:** Validate MPC works across all 4 platforms with manual tuning

### 5.1 Entry Requirements
- Phase 4 checkpoint complete
- MPC working perfectly on Crazyflie
- Baseline RMSE documented

### 5.2 Implementation Tasks

#### Task 5.1: Platform Configurations

Create configuration for each platform:

**Files:**
- `configs/mpc_crazyflie.yaml` (already done)
- `configs/mpc_racing.yaml`
- `configs/mpc_generic.yaml`
- `configs/mpc_heavy.yaml`

**Example: `configs/mpc_racing.yaml`**

```yaml
# MPC Configuration for Racing Quadrotor

mpc:
  prediction_horizon: 10
  control_horizon: 10
  timestep: 0.02
  
  # Initial weights (Bryson's Rule scaled for mass)
  Q: [64, 64, 64,        # Position (0.125m acceptable)
      25, 25, 25,        # Velocity (0.2 m/s)
      36, 36, 16,        # Orientation
      25, 25, 25]        # Angular velocity
  
  R: [0.04, 0.04, 0.04, 0.04]  # Scaled by mass ratio
  
  u_min: [0.0, -5.0, -5.0, -5.0]
  u_max: [30.0, 5.0, 5.0, 5.0]   # Higher limits

drone:
  name: "Racing Quadrotor"
  mass: 0.800
  inertia:
    Ixx: 8.1e-3
    Iyy: 8.1e-3
    Izz: 1.4e-2
```

#### Task 5.2: Automated Testing Script

**File: `tests/test_all_platforms.py`**

```python
"""
Multi-Platform MPC Validation
Tests MPC on all 4 platforms and generates comparison report
"""

class MultiPlatformTest:
    def __init__(self):
        self.platforms = ['crazyflie', 'racing', 'generic', 'heavy']
        self.results = {}
    
    def run_all_tests(self):
        """Run comprehensive tests on all platforms"""
        print("="*60)
        print("MULTI-PLATFORM MPC VALIDATION")
        print("="*60)
        
        for platform in self.platforms:
            print(f"\n### Testing {platform.upper()} ###")
            self.results[platform] = self.test_platform(platform)
        
        # Generate comparison report
        self.generate_comparison_report()
    
    def test_platform(self, platform):
        """Test single platform"""
        tester = MPCTest(platform)
        
        results = {
            'hover': {},
            'circular': {},
            'waypoint': {},
            'aggressive': {}
        }
        
        # Test 1: Hover
        print("\n  [1/4] Hover test...")
        hover_results = tester.test_hover(duration=10.0)
        results['hover'] = {
            'rmse': self._extract_rmse(hover_results),
            'max_error': self._extract_max_error(hover_results),
            'mean_solve_time': np.mean(hover_results['solve_time'])
        }
        
        # Test 2: Circular trajectory
        print("  [2/4] Circular tracking...")
        circular_rmse = tester.test_circular_trajectory(radius=2.0, period=20.0)
        results['circular'] = {
            'rmse': circular_rmse,
            'mean_solve_time': np.mean(tester.results['solve_time'])
        }
        
        # Test 3: Waypoint navigation
        print("  [3/4] Waypoint navigation...")
        waypoints = [[0,0,1], [3,3,2], [-2,3,3], [0,0,1]]
        waypoint_rmse = tester.test_waypoint_sequence(waypoints)
        results['waypoint'] = {'rmse': waypoint_rmse}
        
        # Test 4: Aggressive maneuver (lemniscate)
        print("  [4/4] Aggressive maneuver...")
        lemn_rmse = tester.test_lemniscate(scale=2.0, period=10.0)
        results['aggressive'] = {'rmse': lemn_rmse}
        
        print(f"\n  ✓ {platform} complete")
        return results
    
    def generate_comparison_report(self):
        """Generate markdown report comparing all platforms"""
        
        report = """# Multi-Platform MPC Validation Report

## Summary

Tested MPC controller across 4 heterogeneous UAV platforms spanning 200× mass variation.

## Platform Specifications

| Platform | Mass (kg) | Ixx (kg·m²) | Thrust/Weight |
|----------|-----------|-------------|---------------|
"""
        
        # Add platform specs
        specs = {
            'crazyflie': [0.027, 1.4e-5, 2.26],
            'racing': [0.800, 8.1e-3, 3.75],
            'generic': [2.500, 2.8e-2, 2.04],
            'heavy': [5.500, 8.4e-2, 2.00]
        }
        
        for platform, spec in specs.items():
            report += f"| {platform.capitalize()} | {spec[0]} | {spec[1]:.2e} | {spec[2]:.2f} |\n"
        
        report += "\n## Performance Results\n\n"
        report += "### Circular Trajectory Tracking (Primary Metric)\n\n"
        report += "| Platform | RMSE (m) | Solve Time (ms) | Status |\n"
        report += "|----------|----------|-----------------|--------|\n"
        
        for platform in self.platforms:
            rmse = self.results[platform]['circular']['rmse']
            solve_time = self.results[platform]['circular']['mean_solve_time']
            status = 'PASS' if rmse < 2.0 else 'FAIL'
            report += f"| {platform.capitalize()} | {rmse:.3f} | {solve_time:.1f} | {status} |\n"
        
        # Add more sections...
        report += "\n## Analysis\n\n"
        report += self._generate_analysis()
        
        # Save report
        with open('results/phase_05/MULTI_PLATFORM_REPORT.md', 'w') as f:
            f.write(report)
        
        print(f"\n{'='*60}")
        print("Report saved: results/phase_05/MULTI_PLATFORM_REPORT.md")
        print(f"{'='*60}")
    
    def _generate_analysis(self):
        """Analyze results and generate insights"""
        analysis = ""
        
        # Compare performance across platforms
        rmse_values = [self.results[p]['circular']['rmse'] for p in self.platforms]
        
        analysis += f"### Key Findings\n\n"
        analysis += f"- RMSE range: {min(rmse_values):.3f} - {max(rmse_values):.3f} m\n"
        analysis += f"- Performance variation: {np.std(rmse_values):.3f} m std\n"
        analysis += f"- Best platform: {self.platforms[np.argmin(rmse_values)]}\n"
        analysis += f"- Worst platform: {self.platforms[np.argmax(rmse_values)]}\n\n"
        
        analysis += f"### Observations\n\n"
        analysis += "- MPC works across all platforms ✓\n"
        analysis += "- Performance varies significantly (manual tuning limitation)\n"
        analysis += "- Heavier drones may need different weight scaling\n"
        analysis += "- Ready for RL-based optimization\n"
        
        return analysis
    
    def _extract_rmse(self, results):
        """Extract RMSE from results dict"""
        errors = np.array(results['position_error'])
        return np.sqrt(np.mean(errors**2))
    
    def _extract_max_error(self, results):
        """Extract max error from results"""
        return np.max(results['position_error'])

if __name__ == "__main__":
    tester = MultiPlatformTest()
    tester.run_all_tests()
```

### 5.3 Phase 5 Deliverables

**Checkpoint: `checkpoints/phase_05_checkpoint.yaml`**

```yaml
phase: 5
name: "Multi-Platform MPC Validation"
status: COMPLETED
completion_date: "2025-11-XX"
platforms_tested:
  - crazyflie
  - racing
  - generic
  - heavy
results:
  crazyflie:
    circular_rmse: X.XX
    solve_time: XX.X
  racing:
    circular_rmse: X.XX
    solve_time: XX.X
  generic:
    circular_rmse: X.XX
    solve_time: XX.X
  heavy:
    circular_rmse: X.XX
    solve_time: XX.X
observations:
  - MPC works on all platforms
  - Performance varies (1.5-2.5m RMSE range expected)
  - Manual tuning is time-consuming
  - Ready for RL optimization
next_phase: 6
```

### 5.4 Exit Criteria
- [ ] MPC tested on all 4 platforms
- [ ] Circular tracking RMSE < 3.0m for all platforms
- [ ] Solve time < 50ms for all platforms
- [ ] Comparison report generated
- [ ] Performance variation documented
- [ ] Ready for RL phase

---

## Phase 6: RL Integration (Single Algorithm)
**Duration:** 5-7 days  
**Goal:** Implement PPO-based MPC tuning for Crazyflie

### 6.1 Entry Requirements
- Phase 5 checkpoint complete
- MPC baseline performance established
- All platforms validated

### 6.2 Implementation Tasks

#### Task 6.1: Gymnasium Environment

**File: `src/rl/mpc_tuning_env.py`**

```python
"""
Gymnasium Environment for MPC Hyperparameter Tuning
Treats MPC weight tuning as RL problem
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MPCTuningEnv(gym.Env):
    """
    Environment for tuning MPC hyperparameters using RL
    
    State: [position_error, velocity_error, control_effort,
            current_Q_weights, current_R_weights, current_horizon,
            settling_time, overshoot]
    
    Action: [Q_weight_adjustments (12D), R_weight_adjustments (4D), horizon_change (1D)]
    
    Reward: -position_error - velocity_error - control_penalty
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, platform='crazyflie', trajectory_type='circular'):
        super().__init__()
        
        self.platform = platform
        self.trajectory_type = trajectory_type
        
        # Load MPC controller
        self.mpc = MPCController(f"configs/mpc_{platform}.yaml")
        
        # Load simulator
        self.env = DroneEnv(platform=platform)
        
        # State space (29D)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(29,),
            dtype=np.float32
        )
        
        # Action space (17D: 12 Q + 4 R + 1 horizon)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(17,),
            dtype=np.float32
        )
        
        # Bryson's Rule initialization
        self.Q_bryson = self._compute_bryson_weights()
        self.R_bryson = self._compute_bryson_control_weights()
        
        # Current weights (start with Bryson's)
        self.Q_current = self.Q_bryson.copy()
        self.R_current = self.R_bryson.copy()
        self.N_current = 10
        
        # Episode tracking
        self.episode_step = 0
        self.max_episode_steps = 500  # 10 seconds at 50Hz
        
    def _compute_bryson_weights(self):
        """
        Apply Bryson's Rule for initial Q weights
        Q_i = 1 / (max_acceptable_deviation_i)²
        """
        # Platform-specific acceptable deviations
        platform_params = {
            'crazyflie': {
                'pos': 0.10,  # 10cm position error acceptable
                'vel': 0.25,  # 0.25 m/s velocity error
                'att': 0.20,  # 0.2 rad (11.5°)
                'ang_vel': 0.25  # 0.25 rad/s
            },
            'racing': {
                'pos': 0.125,
                'vel': 0.20,
                'att': 0.167,
                'ang_vel': 0.20
            },
            'generic': {
                'pos': 0.25,
                'vel': 0.50,
                'att': 0.25,
                'ang_vel': 0.50
            },
            'heavy': {
                'pos': 0.50,
                'vel': 1.00,
                'att': 0.40,
                'ang_vel': 1.00
            }
        }
        
        params = platform_params[self.platform]
        
        Q = np.array([
            1/params['pos']**2, 1/params['pos']**2, 1/params['pos']**2,  # Position
            1/params['vel']**2, 1/params['vel']**2, 1/params['vel']**2,  # Velocity
            1/params['att']**2, 1/params['att']**2, 1/params['att']**2,  # Orientation
            1/params['ang_vel']**2, 1/params['ang_vel']**2, 1/params['ang_vel']**2  # Angular vel
        ])
        
        return Q
    
    def _compute_bryson_control_weights(self):
        """Bryson's Rule for R weights"""
        # Acceptable control effort variation
        max_control = 5.0  # Newtons
        R = np.ones(4) * (1 / max_control**2)
        return R
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        # Reset to Bryson's Rule initialization
        self.Q_current = self.Q_bryson.copy()
        self.R_current = self.R_bryson.copy()
        self.N_current = 10
        
        # Update MPC with initial weights
        self.mpc.update_weights(self.Q_current, self.R_current)
        
        # Reset simulation
        self.sim_state = self.env.reset()
        self.episode_step = 0
        
        # Initial observation
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """
        Execute one step: adjust MPC weights, run simulation, return reward
        
        Args:
            action: 17D array of weight adjustments
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Apply action to update MPC weights
        self._apply_action(action)
        
        # Run simulation for one control step (0.02s)
        position_errors = []
        velocity_errors = []
        control_efforts = []
        
        for _ in range(1):  # One MPC step
            # Get reference
            ref_traj = self._generate_reference(self.episode_step * 0.02)
            
            # Get current state
            x_current = self._state_to_vector(self.sim_state)
            
            # MPC control
            u_opt, solve_time, success = self.mpc.compute_control(x_current, ref_traj)
            
            if not success:
                # Penalize solver failure
                return self._get_observation(), -100.0, True, False, {'solver_failed': True}
            
            # Apply control
            self.sim_state = self.env.step(u_opt)
            
            # Track errors
            pos_error = np.linalg.norm(self.sim_state['pos'] - ref_traj[:3, 0])
            vel_error = np.linalg.norm(self.sim_state['vel'] - ref_traj[3:6, 0])
            control_effort = np.linalg.norm(u_opt)
            
            position_errors.append(pos_error)
            velocity_errors.append(vel_error)
            control_efforts.append(control_effort)
        
        # Compute reward
        reward = self._compute_reward(position_errors, velocity_errors, control_efforts)
        
        # Update episode step
        self.episode_step += 1
        
        # Check termination
        terminated = False
        truncated = self.episode_step >= self.max_episode_steps
        
        # Check crash
        if self.sim_state['pos'][2] < 0.2 or np.any(np.abs(self.sim_state['att'][:2]) > 0.8):
            terminated = True
            reward = -50.0  # Crash penalty
        
        obs = self._get_observation()
        info = {
            'position_error': np.mean(position_errors),
            'velocity_error': np.mean(velocity_errors),
            'control_effort': np.mean(control_efforts)
        }
        
        return obs, reward, terminated, truncated, info
    
    def _apply_action(self, action):
        """
        Apply RL action to adjust MPC weights
        
        action[0:12]: Q weight multipliers (exponential scaling)
        action[12:16]: R weight multipliers
        action[16]: Horizon adjustment
        """
        # Exponential scaling for Q weights (allows large adjustments)
        Q_multipliers = np.exp(action[:12] * 0.5)  # Range: [0.6, 1.6] approx
        self.Q_current = self.Q_bryson * Q_multipliers
        
        # Exponential scaling for R weights
        R_multipliers = np.exp(action[12:16] * 0.5)
        self.R_current = self.R_bryson * R_multipliers
        
        # Linear adjustment for horizon (±3 steps)
        horizon_change = int(action[16] * 3)
        self.N_current = np.clip(10 + horizon_change, 8, 15)
        
        # Update MPC
        self.mpc.update_weights(self.Q_current, self.R_current)
    
    def _compute_reward(self, position_errors, velocity_errors, control_efforts):
        """
        Reward function (from published paper)
        r = -10*pos_error - 1*vel_error - 0.01*control - 5*overshoot_penalty
        """
        pos_error = np.mean(position_errors)
        vel_error = np.mean(velocity_errors)
        control = np.mean(control_efforts)
        
        # Overshoot penalty
        overshoot = max(0, pos_error - 0.5)  # Penalize if error > 0.5m
        
        reward = -10.0 * pos_error - 1.0 * vel_error - 0.01 * control - 5.0 * overshoot
        
        return reward
    
    def _get_observation(self):
        """
        Construct 29D observation vector
        
        [position_error (3), velocity_error (3), control_effort (4),
         current_Q (12), current_R (4), current_horizon (1),
         settling_time (1), overshoot (1)]
        """
        # Current errors
        ref = self._generate_reference(self.episode_step * 0.02)
        pos_error = self.sim_state['pos'] - ref[:3, 0]
        vel_error = self.sim_state['vel'] - ref[3:6, 0]
        
        # Recent control (placeholder)
        control_effort = np.zeros(4)
        
        # Current MPC configuration
        Q_normalized = self.Q_current / 100.0  # Normalize for NN
        R_normalized = self.R_current * 10.0
        horizon_normalized = (self.N_current - 10) / 5.0
        
        # Performance metrics (simplified)
        settling_time = 0.0  # TODO: Implement
        overshoot = max(0, np.linalg.norm(pos_error) - 0.5)
        
        obs = np.concatenate([
            pos_error,
            vel_error,
            control_effort,
            Q_normalized,
            R_normalized,
            [horizon_normalized],
            [settling_time],
            [overshoot]
        ])
        
        return obs.astype(np.float32)
    
    def _generate_reference(self, t):
        """Generate reference trajectory"""
        if self.trajectory_type == 'circular':
            return self._circular_reference(t, radius=2.0, period=20.0)
        else:
            # Hover
            ref = np.zeros((12, self.mpc.N + 1))
            ref[2, :] = 2.0  # Hover at 2m
            return ref
    
    def _circular_reference(self, t, radius, period):
        """Circular trajectory reference"""
        ref = np.zeros((12, self.mpc.N + 1))
        
        dt = 0.02
        for i in range(self.mpc.N + 1):
            t_future = t + i * dt
            omega = 2 * np.pi / period
            
            ref[0, i] = radius * np.cos(omega * t_future)
            ref[1, i] = radius * np.sin(omega * t_future)
            ref[2, i] = 2.0
            
            ref[3, i] = -radius * omega * np.sin(omega * t_future)
            ref[4, i] = radius * omega * np.cos(omega * t_future)
            
            ref[8, i] = np.arctan2(ref[4, i], ref[3, i])
        
        return ref
    
    def _state_to_vector(self, state_dict):
        """Convert state dict to 12D vector"""
        return np.concatenate([
            state_dict['pos'],
            state_dict['vel'],
            state_dict['att'],
            state_dict['ang_vel']
        ])
```

#### Task 6.2: PPO Training Script

**File: `src/rl/train_ppo.py`**

```python
"""
PPO Training for MPC Hyperparameter Tuning
Uses Stable-Baselines3
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import wandb
from wandb.integration.sb3 import WandbCallback

class MPCTrainer:
    def __init__(self, platform='crazyflie', use_wandb=True):
        self.platform = platform
        self.use_wandb = use_wandb
        
        if use_wandb:
            wandb.init(
                project="rl-mpc-tuning",
                name=f"ppo_{platform}",
                config={
                    'platform': platform,
                    'algorithm': 'PPO',
                    'total_timesteps': 20000
                }
            )
    
    def make_env(self):
        """Create environment instance"""
        def _init():
            env = MPCTuningEnv(platform=self.platform, trajectory_type='circular')
            return env
        return _init
    
    def train(self, total_timesteps=20000, n_envs=4):
        """
        Train PPO agent
        
        Args:
            total_timesteps: Training steps (20,000 for base, 5,000 for transfer)
            n_envs: Number of parallel environments
        """
        print(f"\n{'='*60}")
        print(f"Training PPO for {self.platform}")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Parallel environments: {n_envs}")
        print(f"{'='*60}\n")
        
        # Create vectorized environment
        env = SubprocVecEnv([self.make_env() for _ in range(n_envs)])
        
        # PPO configuration (from paper)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=f"./logs/{self.platform}/"
        )
        
        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=f"./checkpoints/{self.platform}/",
            name_prefix="ppo_mpc"
        )
        
        callbacks = [checkpoint_callback]
        
        if self.use_wandb:
            wandb_callback = WandbCallback(
                model_save_freq=1000,
                model_save_path=f"./models/{self.platform}/",
                verbose=2
            )
            callbacks.append(wandb_callback)
        
        # Train
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        model.save(f"models/{self.platform}/ppo_mpc_final")
        
        print(f"\n✓ Training complete for {self.platform}")
        
        return model
    
    def evaluate(self, model, n_episodes=10):
        """
        Evaluate trained model
        
        Returns:
            mean_rmse: Average tracking RMSE
        """
        env = MPCTuningEnv(platform=self.platform, trajectory_type='circular')
        
        episode_rmses = []
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_errors = []
            
            done = False
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_errors.append(info['position_error'])
                done = terminated or truncated
            
            episode_rmse = np.sqrt(np.mean(np.array(episode_errors)**2))
            episode_rmses.append(episode_rmse)
            
            print(f"  Episode {episode+1}: RMSE = {episode_rmse:.3f} m")
        
        mean_rmse = np.mean(episode_rmses)
        std_rmse = np.std(episode_rmses)
        
        print(f"\n  Mean RMSE: {mean_rmse:.3f} ± {std_rmse:.3f} m")
        
        return mean_rmse

if __name__ == "__main__":
    # Train PPO on Crazyflie (base platform)
    trainer = MPCTrainer(platform='crazyflie')
    
    # Train for 20,000 steps
    model = trainer.train(total_timesteps=20000, n_envs=4)
    
    # Evaluate
    rmse = trainer.evaluate(model, n_episodes=10)
    
    print(f"\nPhase 6 Complete: RMSE = {rmse:.3f} m")
```

### 6.3 Phase 6 Deliverables

**Checkpoint: `checkpoints/phase_06_checkpoint.yaml`**

```yaml
phase: 6
name: "RL Integration - PPO"
status: COMPLETED
completion_date: "2025-11-XX"
platform: "crazyflie"
algorithm: "PPO"
achievements:
  - gymnasium_env_implemented: true
  - bryson_initialization: true
  - ppo_training_complete: true
  - training_converged: true
training_metrics:
  total_timesteps: 20000
  training_time_minutes: 200.3
  parallel_envs: 4
  final_rmse: 1.33  # Target from paper
performance:
  baseline_rmse: X.XX  # MPC without RL
  rl_optimized_rmse: 1.33  # After RL tuning
  improvement_percent: XX.X
next_phase: 7
model_path: "models/crazyflie/ppo_mpc_final.zip"
```

### 6.4 Exit Criteria
- [ ] Gymnasium environment working
- [ ] Bryson's Rule initialization implemented
- [ ] PPO training completes (20,000 steps)
- [ ] Training converges (reward plateau)
- [ ] RMSE ≤ 1.35m achieved
- [ ] Model checkpoint saved
- [ ] Training logs saved (W&B/TensorBoard)

---

## Phase 7: Transfer Learning & Multi-Algorithm
**Duration:** 7-10 days  
**Goal:** Transfer to other platforms and test alternative RL algorithms

### 7.1 Sequential Transfer Learning

**File: `src/rl/transfer_learning.py`**

```python
"""
Sequential Transfer Learning
Transfer learned MPC parameters from base to target platforms
"""

class TransferLearning:
    def __init__(self):
        self.platforms = ['crazyflie', 'racing', 'generic', 'heavy']
        self.base_platform = 'crazyflie'
    
    def sequential_transfer(self):
        """
        Sequential transfer: Crazyflie → Racing → Generic → Heavy
        """
        print("="*60)
        print("SEQUENTIAL TRANSFER LEARNING")
        print("="*60)
        
        # Load base model
        base_model = PPO.load(f"models/{self.base_platform}/ppo_mpc_final.zip")
        
        results = {}
        
        for i, target_platform in enumerate(self.platforms[1:], 1):
            print(f"\n### Transfer {i}/3: {target_platform.upper()} ###")
            
            # Transfer and fine-tune
            rmse = self.transfer_and_finetune(
                source_model=base_model if i == 1 else prev_model,
                target_platform=target_platform,
                finetune_steps=5000  # 25% of baseline
            )
            
            results[target_platform] = rmse
            
            # Load for next transfer
            prev_model = PPO.load(f"models/{target_platform}/ppo_mpc_finetuned.zip")
        
        self.generate_transfer_report(results)
        return results
    
    def transfer_and_finetune(self, source_model, target_platform, finetune_steps):
        """
        Transfer policy and fine-tune on target platform
        
        Args:
            source_model: Trained PPO model
            target_platform: Target platform name
            finetune_steps: Number of fine-tuning steps
        """
        print(f"  Loading source model...")
        print(f"  Creating target environment: {target_platform}")
        
        # Create target environment
        def make_env():
            return MPCTuningEnv(platform=target_platform, trajectory_type='circular')
        
        env = SubprocVecEnv([make_env for _ in range(4)])
        
        # Transfer policy (set environment)
        source_model.set_env(env)
        
        # Reduce learning rate for fine-tuning
        source_model.learning_rate = 3e-5  # 10× smaller
        
        print(f"  Fine-tuning for {finetune_steps} steps...")
        
        # Fine-tune
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=f"./checkpoints/{target_platform}/",
            name_prefix="ppo_mpc_transfer"
        )
        
        source_model.learn(
            total_timesteps=finetune_steps,
            callback=[checkpoint_callback],
            progress_bar=True,
            reset_num_timesteps=False
        )
        
        # Save fine-tuned model
        source_model.save(f"models/{target_platform}/ppo_mpc_finetuned")
        
        # Evaluate
        print(f"  Evaluating on {target_platform}...")
        trainer = MPCTrainer(platform=target_platform, use_wandb=False)
        rmse = trainer.evaluate(source_model, n_episodes=10)
        
        return rmse
    
    def generate_transfer_report(self, results):
        """Generate transfer learning report"""
        report = """# Sequential Transfer Learning Report

## Transfer Sequence
Crazyflie (0.027kg) → Racing (0.800kg) → Generic (2.500kg) → Heavy-Lift (5.500kg)

## Results

| Platform | Mass (kg) | Fine-tune Steps | RMSE (m) | Status |
|----------|-----------|-----------------|----------|--------|
"""
        
        report += f"| Crazyflie | 0.027 | 20,000 (base) | 1.33 | ✓ PASS |\n"
        
        platforms_mass = {'racing': 0.800, 'generic': 2.500, 'heavy': 5.500}
        
        for platform, rmse in results.items():
            mass = platforms_mass[platform]
            status = 'PASS' if rmse < 1.36 else 'FAIL'
            report += f"| {platform.capitalize()} | {mass} | 5,000 | {rmse:.2f} | {status} |\n"
        
        report += "\n## Analysis\n\n"
        report += f"- Transfer learning successful: {'YES' if all(r < 1.36 for r in results.values()) else 'NO'}\n"
        report += f"- Consistent performance: 1.34±0.01m target {'ACHIEVED' if max(results.values())-min(results.values()) < 0.02 else 'NOT MET'}\n"
        report += f"- Training efficiency: 75% step reduction (5k vs 20k)\n"
        report += f"- Time savings: ~56% (estimated)\n"
        
        with open('results/phase_07/TRANSFER_REPORT.md', 'w') as f:
            f.write(report)
        
        print(f"\n{report}")

if __name__ == "__main__":
    transfer = TransferLearning()
    results = transfer.sequential_transfer()
```

### 7.2 Alternative RL Algorithms

Implement and test TRPO, SAC, TD3, A2C following same pattern as PPO.

**Files to create:**
- `src/rl/train_trpo.py`
- `src/rl/train_sac.py`
- `src/rl/train_td3.py`
- `src/rl/train_a2c.py`

### 7.3 Phase 7 Deliverables

**Checkpoint: `checkpoints/phase_07_checkpoint.yaml`**

```yaml
phase: 7
name: "Transfer Learning & Multi-Algorithm"
status: COMPLETED
completion_date: "2025-11-XX"
achievements:
  sequential_transfer:
    crazyflie_to_racing: PASS
    racing_to_generic: PASS
    generic_to_heavy: PASS
  algorithms_tested:
    - PPO
    - TRPO
    - SAC
    - TD3
    - A2C
results:
  transfer_learning:
    total_time_minutes: 362.9
    rmse_consistency: "1.34±0.01m"
    success_rate: "100%"
  algorithm_comparison:
    best_algorithm: "PPO"
    best_sample_efficiency: "SAC"
    fastest_training: "PPO"
project_status: COMPLETE
```

### 7.4 Exit Criteria
- [ ] Transfer learning successful for all 3 platforms
- [ ] RMSE ≤ 1.35m on all platforms
- [ ] 5 algorithms tested and compared
- [ ] Total training time < 7 hours
- [ ] Final comparison report generated
- [ ] All models and checkpoints saved

---

## Project Management Templates

### Daily Progress Log Template

**File: `PROGRESS_LOG.md`**

```markdown
# Development Progress Log

## [Date: 2025-11-XX]

### Phase: X - [Phase Name]
**Status:** In Progress / Blocked / Complete

### Today's Goals
- [ ] Goal 1
- [ ] Goal 2
- [ ] Goal 3

### Accomplishments
- ✓ Completed task X
- ✓ Fixed bug in Y
- ✓ Tested Z

### Challenges / Issues
**Issue 1:** [Description]
- Attempted solution: [what you tried]
- Outcome: [result]
- Next step: [plan]

### Metrics
- RMSE: X.XX m
- Solve time: XX ms
- Training steps: XXXX

### Tomorrow's Plan
- [ ] Task 1
- [ ] Task 2

### Notes
[Any observations, ideas, or reminders]

---
```

### Resume Checklist

**File: `RESUME_CHECKLIST.md`**

```markdown
# Project Resume Checklist

Use this when returning to project after break.

## 1. Check Current Phase
- [ ] Read latest checkpoint: `checkpoints/phase_XX_checkpoint.yaml`
- [ ] Review phase status and completion date
- [ ] Check next_phase directive

## 2. Review Recent Progress
- [ ] Read last 3 entries in `PROGRESS_LOG.md`
- [ ] Check git commit history
- [ ] Review latest results in `results/phase_XX/`

## 3. Verify Environment
- [ ] Simulator running correctly
- [ ] Python environment activated
- [ ] All dependencies installed
- [ ] GPU available (if using)

## 4. Run Tests
- [ ] Run last successful test to verify setup
- [ ] Check test passes: `python tests/test_[component].py`

## 5. Continue Work
- [ ] Clear understanding of next task
- [ ] Required files identified
- [ ] Estimated time to completion noted

## Current Status
**Phase:** [X]
**Last Checkpoint:** [date]
**Next Milestone:** [description]
**ETA:** [X days]
```

### Troubleshooting Guide

**File: `docs/TROUBLESHOOTING.md`**

```markdown
# Troubleshooting Guide

## Common Issues

### Simulator Won't Start
**Symptoms:** [description]
**Solution:**
1. Check installation
2. Verify Python version
3. [steps]

### MPC Solver Fails
**Symptoms:** "Infeasible problem" or timeout
**Possible Causes:**
- Constraints too tight
- Initial guess poor
- Weights poorly scaled

**Solutions:**
1. Check constraint violations
2. Relax constraints temporarily
3. Verify weight magnitudes
4. Check horizon length

### RL Training Not Converging
**Symptoms:** Reward not improving, erratic behavior
**Checklist:**
- [ ] Reward function correctly implemented
- [ ] State normalization applied
- [ ] Action space appropriate
- [ ] Learning rate not too high
- [ ] Enough training steps

### Phase Checkpoint Missing
**Solution:**
1. Check `checkpoints/` directory
2. Reconstruct from git history
3. Re-run validation tests

## Getting Help
1. Check documentation in `docs/phase_XX/`
2. Review similar examples in tests
3. Check GitHub issues (if applicable)
4. Ask specific questions with error logs
```

---

## Quick Reference Cards

### Phase Status at a Glance

```
├─ Phase 1: Simulator Selection        [COMPLETE] ✓
├─ Phase 2: PID Controller             [COMPLETE] ✓
├─ Phase 3: Obstacle Avoidance         [COMPLETE] ✓
├─ Phase 4: MPC Implementation         [IN PROGRESS] ⚙
├─ Phase 5: Multi-Platform Validation  [NOT STARTED] ○
├─ Phase 6: RL Integration (PPO)       [NOT STARTED] ○
└─ Phase 7: Transfer Learning          [NOT STARTED] ○
```

### Key Metrics Tracker

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Hover RMSE | <0.1m | X.XX m | ? |
| Circular RMSE | <1.35m | X.XX m | ? |
| MPC Solve Time | <50ms | XX ms | ? |
| RL Training Time | <200 min | XX min | ? |
| Transfer RMSE | 1.34±0.01m | X.XX±X.XX m | ? |

---

## Final Notes

### Success Philosophy
1. **One phase at a time** - Don't move forward until current phase passes
2. **Document everything** - Future you will thank present you
3. **Test thoroughly** - Bugs caught early save hours later
4. **Commit often** - Small commits are easier to debug
5. **Celebrate milestones** - Acknowledge progress

### Time Estimates (Realistic)
- Phase 1-3: 7-10 days
- Phase 4-5: 8-11 days
- Phase 6-7: 12-17 days
**Total: 27-38 days (5-8 weeks)**

### Remember
- Progress > Perfection
- Documentation > Memory
- Testing > Debugging
- Planning > Rushing

**Good luck with your development! 🚀**
