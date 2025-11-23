"""
Gymnasium Environment for MPC Hyperparameter Tuning
Treats MPC weight tuning as an RL problem

Author: Dr. Abdul Manan Khan
Project: RL-Enhanced MPC for Multi-Drone Systems
Phase: 6 - RL Integration
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import yaml
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpc.mpc_controller import MPCController
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class MPCTuningEnv(gym.Env):
    """
    Gymnasium environment for tuning MPC hyperparameters using RL

    State Space (29D):
        - Position error (3D): [ex, ey, ez]
        - Velocity error (3D): [evx, evy, evz]
        - Control effort (4D): [thrust, roll_rate, pitch_rate, yaw_rate]
        - Current Q weights (12D): normalized state cost weights
        - Current R weights (4D): normalized control cost weights
        - Current horizon (1D): normalized prediction horizon
        - Settling time (1D): time to reach steady state
        - Overshoot (1D): maximum overshoot magnitude

    Action Space (17D):
        - Q weight adjustments (12D): exponential multipliers for state costs
        - R weight adjustments (4D): exponential multipliers for control costs
        - Horizon adjustment (1D): linear change to prediction horizon

    Reward Function:
        r = -10*pos_error - 1*vel_error - 0.01*control - 5*overshoot
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, platform='crazyflie', trajectory_type='circular',
                 max_episode_steps=500, gui=False):
        """
        Initialize MPC tuning environment

        Args:
            platform: Drone platform ('crazyflie', 'racing', 'generic', 'heavy-lift')
            trajectory_type: Reference trajectory ('circular', 'figure8', 'hover')
            max_episode_steps: Maximum steps per episode (default: 500 = 10s @ 50Hz)
            gui: Enable PyBullet GUI visualization
        """
        super().__init__()

        self.platform = platform
        self.trajectory_type = trajectory_type
        self.max_episode_steps = max_episode_steps
        self.gui = gui

        # Load MPC configuration
        config_path = f"configs/mpc_crazyflie.yaml"  # Using CF2X config (Phase 5 finding)
        with open(config_path, 'r') as f:
            self.mpc_config = yaml.safe_load(f)

        # Initialize MPC controller
        self.mpc = MPCController(config_path)

        # Drone model (using CF2X as per Phase 5 findings)
        self.drone_model = DroneModel.CF2X

        # Simulation parameters (MUST be set before _init_simulation)
        self.dt = self.mpc_config['mpc']['timestep']
        self.control_freq_hz = self.mpc_config['simulation']['control_freq_hz']
        self.physics_freq_hz = self.mpc_config['simulation']['physics_freq_hz']

        # Trajectory parameters
        self.traj_radius = 0.5  # meters
        self.traj_period = 10.0  # seconds
        self.hover_altitude = 1.0  # meters

        # Initialize PyBullet environment (after parameters are set)
        self.sim_env = None
        self._init_simulation()

        # State space (29D)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(29,),
            dtype=np.float32
        )

        # Action space (17D: 12 Q + 4 R + 1 horizon)
        # Actions in range [-1, 1] will be mapped to weight multipliers
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(17,),
            dtype=np.float32
        )

        # Bryson's Rule initialization
        self.Q_bryson = self._compute_bryson_weights()
        self.R_bryson = self._compute_bryson_control_weights()

        # Current weights (start with Bryson's Rule)
        self.Q_current = self.Q_bryson.copy()
        self.R_current = self.R_bryson.copy()
        self.N_current = self.mpc_config['mpc']['prediction_horizon']

        # Episode tracking
        self.episode_step = 0
        self.episode_errors = []
        self.episode_controls = []

        # Track current observation (CtrlAviary doesn't have .last() method)
        self.current_obs = None

    def _init_simulation(self):
        """Initialize PyBullet simulation environment"""
        initial_xyzs = np.array([[0.0, 0.0, 1.0]])  # Start at 1m altitude
        initial_rpys = np.array([[0.0, 0.0, 0.0]])

        self.sim_env = CtrlAviary(
            drone_model=self.drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=Physics.PYB,
            pyb_freq=self.physics_freq_hz,
            ctrl_freq=self.control_freq_hz,
            gui=self.gui,
            record=False
        )

    def _compute_bryson_weights(self):
        """
        Apply Bryson's Rule for initial Q weights
        Q_i = 1 / (max_acceptable_deviation_i)²

        Platform-specific acceptable deviations based on Phase 5 findings
        """
        platform_params = {
            'crazyflie': {
                'pos': 0.10,  # 10cm position error acceptable
                'vel': 0.25,  # 0.25 m/s velocity error
                'att': 0.20,  # 0.2 rad (11.5°)
                'ang_vel': 0.25  # 0.25 rad/s
            },
            'racing': {
                'pos': 0.125,
                'vel': 0.30,
                'att': 0.167,
                'ang_vel': 0.29
            },
            'generic': {
                'pos': 0.25,
                'vel': 0.50,
                'att': 0.25,
                'ang_vel': 0.50
            },
            'heavy-lift': {
                'pos': 0.50,
                'vel': 1.00,
                'att': 0.40,
                'ang_vel': 1.00
            }
        }

        # Use crazyflie params for all (Phase 5 finding)
        params = platform_params.get(self.platform, platform_params['crazyflie'])

        Q = np.array([
            1/params['pos']**2, 1/params['pos']**2, 1/params['pos']**2,  # px, py, pz
            1/params['vel']**2, 1/params['vel']**2, 1/params['vel']**2,  # vx, vy, vz
            1/params['att']**2, 1/params['att']**2, 1/params['att']**2,  # roll, pitch, yaw
            1/params['ang_vel']**2, 1/params['ang_vel']**2, 1/params['ang_vel']**2  # p, q, r
        ])

        return Q

    def _compute_bryson_control_weights(self):
        """
        Bryson's Rule for R weights
        R_i = 1 / (max_acceptable_control_i)²
        """
        # Acceptable control effort variation
        max_thrust = 5.0  # Newtons
        max_ang_rate = 3.0  # rad/s

        R = np.array([
            1 / max_thrust**2,
            1 / max_ang_rate**2,
            1 / max_ang_rate**2,
            1 / max_ang_rate**2
        ])

        return R

    def reset(self, seed=None, options=None):
        """
        Reset environment for new episode

        Returns:
            observation: Initial 29D state vector
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Reset to Bryson's Rule initialization
        self.Q_current = self.Q_bryson.copy()
        self.R_current = self.R_bryson.copy()
        self.N_current = self.mpc_config['mpc']['prediction_horizon']

        # Update MPC with initial weights
        self.mpc.update_weights(self.Q_current, self.R_current)

        # Reset simulation (reuse existing environment, don't create new one)
        if self.sim_env is None:
            self._init_simulation()
        obs, _ = self.sim_env.reset()
        self.current_obs = obs[0]  # Extract first drone's observation (shape: 1,20 -> 20)

        # Reset episode tracking
        self.episode_step = 0
        self.episode_errors = []
        self.episode_controls = []

        # Get initial observation
        obs = self._get_observation()
        info = {
            'Q_weights': self.Q_current.copy(),
            'R_weights': self.R_current.copy(),
            'horizon': self.N_current
        }

        return obs, info

    def step(self, action):
        """
        Execute one RL step: adjust MPC weights and run simulation

        Args:
            action: 17D array of weight adjustments

        Returns:
            observation: 29D state vector
            reward: Scalar reward value
            terminated: Episode ended due to failure
            truncated: Episode ended due to time limit
            info: Additional information
        """
        # Apply action to update MPC weights
        self._apply_action(action)

        # Run simulation for one control step
        position_errors = []
        velocity_errors = []
        control_efforts = []

        # Get current state from PyBullet (use internally tracked observation)
        obs_sim = self.current_obs
        state_current = self._obs_to_state(obs_sim)

        # Generate reference trajectory
        t = self.episode_step * self.dt
        ref_traj = self._generate_reference(t)

        # Compute MPC control
        try:
            u_opt, solve_time, success = self.mpc.compute_control(state_current, ref_traj)

            if not success:
                # Penalize solver failure
                return self._get_observation(), -100.0, True, False, {
                    'solver_failed': True,
                    'position_error': np.nan,
                    'velocity_error': np.nan,
                    'solve_time_ms': solve_time
                }

            # Apply control to simulation
            action_sim = self._mpc_to_sim_action(u_opt)
            obs_sim, _, _, _, _ = self.sim_env.step(action_sim)

            # Update current observation (extract first drone)
            self.current_obs = obs_sim[0]  # Shape: 1,20 -> 20

            # Compute tracking errors
            state_new = self._obs_to_state(self.current_obs)
            pos_error = np.linalg.norm(state_new[:3] - ref_traj[:3, 0])
            vel_error = np.linalg.norm(state_new[3:6] - ref_traj[3:6, 0])
            control_effort = np.linalg.norm(u_opt)

            position_errors.append(pos_error)
            velocity_errors.append(vel_error)
            control_efforts.append(control_effort)

            # Track for episode statistics
            self.episode_errors.append(pos_error)
            self.episode_controls.append(control_effort)

        except Exception as e:
            print(f"MPC step failed: {e}")
            return self._get_observation(), -100.0, True, False, {'error': str(e)}

        # Compute reward
        reward = self._compute_reward(position_errors, velocity_errors, control_efforts)

        # Update episode step
        self.episode_step += 1

        # Check termination conditions
        terminated = False
        truncated = self.episode_step >= self.max_episode_steps

        # Check crash (altitude too low or excessive tilt)
        altitude = state_new[2]
        roll, pitch = state_new[6:8]

        if altitude < 0.2 or np.abs(roll) > 0.8 or np.abs(pitch) > 0.8:
            terminated = True
            reward = -50.0  # Crash penalty

        # Get observation
        obs = self._get_observation()

        info = {
            'position_error': np.mean(position_errors) if position_errors else 0.0,
            'velocity_error': np.mean(velocity_errors) if velocity_errors else 0.0,
            'control_effort': np.mean(control_efforts) if control_efforts else 0.0,
            'solve_time_ms': solve_time if 'solve_time' in locals() else 0.0,
            'episode_step': self.episode_step
        }

        return obs, reward, terminated, truncated, info

    def _apply_action(self, action):
        """
        Apply RL action to adjust MPC weights

        action[0:12]: Q weight multipliers (exponential scaling)
        action[12:16]: R weight multipliers (exponential scaling)
        action[16]: Horizon adjustment (linear)
        """
        # Exponential scaling for Q weights (allows range [0.6, 1.6] approx)
        Q_multipliers = np.exp(action[:12] * 0.5)
        self.Q_current = self.Q_bryson * Q_multipliers

        # Exponential scaling for R weights
        R_multipliers = np.exp(action[12:16] * 0.5)
        self.R_current = self.R_bryson * R_multipliers

        # Linear adjustment for horizon (±3 steps)
        horizon_change = int(action[16] * 3)
        base_horizon = self.mpc_config['mpc']['prediction_horizon']
        self.N_current = np.clip(base_horizon + horizon_change, 8, 25)

        # Update MPC controller
        self.mpc.update_weights(self.Q_current, self.R_current)

    def _compute_reward(self, position_errors, velocity_errors, control_efforts):
        """
        Reward function (from development roadmap)
        r = -10*pos_error - 1*vel_error - 0.01*control - 5*overshoot

        Args:
            position_errors: List of position tracking errors
            velocity_errors: List of velocity tracking errors
            control_efforts: List of control effort magnitudes

        Returns:
            reward: Scalar reward value
        """
        pos_error = np.mean(position_errors) if position_errors else 0.0
        vel_error = np.mean(velocity_errors) if velocity_errors else 0.0
        control = np.mean(control_efforts) if control_efforts else 0.0

        # Overshoot penalty (penalize if error > 0.5m)
        overshoot = max(0, pos_error - 0.5)

        # Compute reward
        reward = -10.0 * pos_error - 1.0 * vel_error - 0.01 * control - 5.0 * overshoot

        return reward

    def _get_observation(self):
        """
        Construct 29D observation vector

        [position_error (3), velocity_error (3), control_effort (4),
         current_Q (12), current_R (4), current_horizon (1),
         settling_time (1), overshoot (1)]
        """
        # Get current simulation state
        if self.current_obs is None:
            # If simulation not initialized, return zeros
            return np.zeros(29, dtype=np.float32)

        obs_sim = self.current_obs
        state_current = self._obs_to_state(obs_sim)

        # Get reference at current time
        t = self.episode_step * self.dt
        ref_traj = self._generate_reference(t)

        # Compute errors
        pos_error = state_current[:3] - ref_traj[:3, 0]
        vel_error = state_current[3:6] - ref_traj[3:6, 0]

        # Recent control effort (placeholder - use last recorded or zero)
        control_effort = np.zeros(4)
        if self.episode_controls:
            control_effort = np.array([self.episode_controls[-1], 0, 0, 0])

        # Normalize MPC configuration for neural network
        Q_normalized = self.Q_current / 100.0
        R_normalized = self.R_current * 10.0
        base_horizon = self.mpc_config['mpc']['prediction_horizon']
        horizon_normalized = (self.N_current - base_horizon) / 5.0

        # Performance metrics
        settling_time = 0.0  # Simplified - would need more tracking
        if len(self.episode_errors) > 10:
            # Simple estimate: time when error < 0.1m
            errors = np.array(self.episode_errors)
            settled_idx = np.where(errors < 0.1)[0]
            if len(settled_idx) > 0:
                settling_time = settled_idx[0] * self.dt

        overshoot = max(self.episode_errors) if self.episode_errors else 0.0

        # Construct observation
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
        """
        Generate reference trajectory for MPC

        Args:
            t: Current time (seconds)

        Returns:
            ref_traj: 12 x (N+1) reference trajectory matrix
        """
        # Use fixed MPC horizon (CasADi MPC can't change N dynamically)
        N = self.mpc_config['mpc']['prediction_horizon']

        if self.trajectory_type == 'circular':
            return self._circular_reference(t, self.traj_radius, self.traj_period, N)
        elif self.trajectory_type == 'figure8':
            return self._figure8_reference(t, self.traj_radius, self.traj_period, N)
        else:  # hover
            ref = np.zeros((12, N + 1))
            ref[2, :] = self.hover_altitude  # Hover at specified altitude
            return ref

    def _circular_reference(self, t, radius, period, N):
        """Generate circular trajectory reference"""
        ref = np.zeros((12, N + 1))
        omega = 2 * np.pi / period

        for i in range(N + 1):
            t_future = t + i * self.dt

            # Position
            ref[0, i] = radius * np.cos(omega * t_future)
            ref[1, i] = radius * np.sin(omega * t_future)
            ref[2, i] = self.hover_altitude

            # Velocity
            ref[3, i] = -radius * omega * np.sin(omega * t_future)
            ref[4, i] = radius * omega * np.cos(omega * t_future)
            ref[5, i] = 0.0

            # Orientation (yaw follows velocity)
            ref[8, i] = np.arctan2(ref[4, i], ref[3, i])

        return ref

    def _figure8_reference(self, t, amplitude, period, N):
        """Generate figure-8 (lemniscate) trajectory reference"""
        ref = np.zeros((12, N + 1))
        omega = 2 * np.pi / period

        for i in range(N + 1):
            t_future = t + i * self.dt

            # Lemniscate of Bernoulli
            ref[0, i] = amplitude * np.sin(omega * t_future)
            ref[1, i] = amplitude * np.sin(omega * t_future) * np.cos(omega * t_future)
            ref[2, i] = self.hover_altitude

            # Velocity (numerical derivative)
            ref[3, i] = amplitude * omega * np.cos(omega * t_future)
            ref[4, i] = amplitude * omega * (np.cos(omega * t_future)**2 - np.sin(omega * t_future)**2)
            ref[5, i] = 0.0

            # Orientation
            ref[8, i] = np.arctan2(ref[4, i], ref[3, i])

        return ref

    def _obs_to_state(self, obs):
        """
        Convert PyBullet observation to 12D state vector

        PyBullet obs format: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, ...]

        Returns:
            state: [px, py, pz, vx, vy, vz, roll, pitch, yaw, p, q, r]
        """
        # Position
        pos = obs[0:3]

        # Velocity
        vel = obs[7:10]

        # Quaternion to Euler angles
        qw, qx, qy, qz = obs[3:7]
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arcsin(2*(qw*qy - qz*qx))
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))

        # Angular velocity
        ang_vel = obs[10:13]

        state = np.concatenate([pos, vel, [roll, pitch, yaw], ang_vel])
        return state

    def _mpc_to_sim_action(self, u_mpc):
        """
        Convert MPC control to PyBullet simulation action

        MPC control: [thrust, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd]
        PyBullet action: [rpm0, rpm1, rpm2, rpm3]

        For now, use simplified conversion (would need proper control allocation)
        """
        # Simplified: Use PyBullet's internal controller
        # This would need proper implementation for real use
        thrust = u_mpc[0]

        # Convert to RPM (simplified)
        # For CF2X: hover thrust ~ 0.265N, max thrust ~ 0.6N
        hover_rpm = 20000
        max_rpm = 35000

        thrust_normalized = np.clip(thrust / 0.6, 0, 1)
        rpm_base = hover_rpm + (max_rpm - hover_rpm) * thrust_normalized

        # Equal distribution (simplified - would need mixer matrix)
        action = np.array([rpm_base, rpm_base, rpm_base, rpm_base])

        return action

    def close(self):
        """Clean up environment"""
        if self.sim_env is not None:
            self.sim_env.close()
