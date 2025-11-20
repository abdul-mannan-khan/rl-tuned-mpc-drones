"""
MPC Controller Test Suite
Comprehensive testing of MPC performance with PyBullet simulation

Tests:
1. Hover stability
2. Waypoint tracking
3. Trajectory following
4. Performance metrics

Author: Dr. Abdul Manan Khan
Project: RL-Enhanced MPC for Multi-Drone Systems
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import MPC controller
from src.mpc.mpc_controller import MPCController

# Import gym-pybullet-drones
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class MPCTest:
    """MPC Controller Test Suite"""

    def __init__(self, config_path="configs/mpc_crazyflie.yaml", gui=False, show_plots=True, drone_model=None):
        """
        Initialize test suite

        Args:
            config_path: Path to MPC configuration file
            gui: Whether to show PyBullet GUI
            show_plots: Whether to display matplotlib plot windows
            drone_model: DroneModel enum (CF2X, CF2P, RACE, or None for default)
        """
        self.config_path = config_path
        self.gui = gui
        self.show_plots = show_plots
        self.drone_model = drone_model if drone_model is not None else DroneModel.CF2X
        self.current_iteration = 2  # Default iteration number

        # Create MPC controller
        print(f"Loading MPC controller from {config_path}...")
        self.mpc = MPCController(config_path)
        print(f"MPC initialized: N={self.mpc.N}, dt={self.mpc.dt}")

        # Create PyBullet environment
        print(f"Creating PyBullet environment with {self.drone_model} drone...")
        self.env = CtrlAviary(
            drone_model=self.drone_model,
            num_drones=1,
            initial_xyzs=np.array([[0.0, 0.0, 0.1]]),  # Start slightly above ground (not at target!)
            initial_rpys=np.array([[0.0, 0.0, 0.0]]),
            physics=Physics.PYB,
            pyb_freq=240,  # Physics frequency
            ctrl_freq=48,  # Control frequency (matches MPC dt)
            gui=gui,
            record=False
        )

        self.dt_control = 1.0 / 48  # Control timestep (48 Hz)

        # Results storage
        self.results = {
            'time': [],
            'position': [],
            'velocity': [],
            'orientation': [],
            'angular_velocity': [],
            'control': [],
            'reference': [],
            'solve_time': [],
            'solve_success': []
        }

    def get_state_from_obs(self, obs):
        """
        Extract 12D state vector from observation

        Args:
            obs: Environment observation

        Returns:
            state: 12D numpy array [px, py, pz, vx, vy, vz, roll, pitch, yaw, p, q, r]
        """
        # obs is dict with keys like 'position', 'orientation', etc.
        # Extract state components
        pos = obs[0:3]
        quat = obs[3:7]
        vel = obs[10:13]
        ang_vel = obs[13:16]

        # Convert quaternion to Euler angles
        # quat = [x, y, z, w]
        roll, pitch, yaw = self._quat_to_euler(quat)

        state = np.array([
            pos[0], pos[1], pos[2],        # Position
            vel[0], vel[1], vel[2],        # Velocity
            roll, pitch, yaw,               # Orientation (Euler)
            ang_vel[0], ang_vel[1], ang_vel[2]  # Angular velocity
        ])

        return state

    def _quat_to_euler(self, quat):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw)

        Args:
            quat: Quaternion [x, y, z, w]

        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def generate_reference_trajectory(self, current_state, target_position, N):
        """
        Generate reference trajectory for MPC

        For now, simple straight-line trajectory to target

        Args:
            current_state: Current 12D state
            target_position: Desired position [x, y, z]
            N: Prediction horizon length

        Returns:
            ref_traj: Reference trajectory (12 x N+1)
        """
        ref_traj = np.zeros((12, N + 1))

        # Linear interpolation from current to target
        current_pos = current_state[0:3]

        for k in range(N + 1):
            alpha = min(1.0, k / N)  # Interpolation factor

            # Position reference
            ref_traj[0:3, k] = current_pos + alpha * (target_position - current_pos)

            # Velocity reference (zero at target)
            if k < N:
                ref_traj[3:6, k] = (target_position - current_pos) / (N * self.mpc.dt)
            else:
                ref_traj[3:6, k] = 0.0

            # Zero orientation and angular velocity
            ref_traj[6:12, k] = 0.0

        return ref_traj

    def control_to_rpm(self, control):
        """
        Convert MPC control to motor RPMs for PyBullet using proper control allocation

        Uses the same method as DSLPIDControl from gym-pybullet-drones.

        Args:
            control: [thrust, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd]
                thrust: Total thrust in Newtons
                angular_rate_cmds: Desired angular rates in rad/s

        Returns:
            rpms: 4D array of motor RPMs
        """
        # Extract control components
        thrust_N = control[0]  # Thrust in Newtons
        ang_rate_cmd = control[1:4]  # [p, q, r] commands in rad/s

        # Crazyflie 2.X parameters (from DSLPIDControl)
        KF = 3.16e-10  # Thrust coefficient
        PWM2RPM_SCALE = 0.2685
        PWM2RPM_CONST = 4070.3
        MIN_PWM = 20000
        MAX_PWM = 65535

        # Mixer matrix for CF2X (X-configuration)
        MIXER_MATRIX = np.array([
            [-0.5, -0.5, -1],
            [-0.5,  0.5,  1],
            [ 0.5,  0.5, -1],
            [ 0.5, -0.5,  1]
        ])

        # Convert thrust (Newtons) to PWM units
        # Formula: thrust_pwm = (sqrt(thrust_N / (4*KF)) - PWM2RPM_CONST) / PWM2RPM_SCALE
        thrust_pwm = (np.sqrt(max(0, thrust_N) / (4 * KF)) - PWM2RPM_CONST) / PWM2RPM_SCALE

        # Convert angular rate commands to torques (simplified model)
        # Using approximate gains from DSLPIDControl
        # These are the P gains for attitude control
        P_COEFF_TOR = np.array([70000., 70000., 60000.])

        # For now, treat angular rate commands directly as normalized torque inputs
        # Scale to appropriate range for mixer
        target_torques = ang_rate_cmd * 1000.0  # Scale factor
        target_torques = np.clip(target_torques, -3200, 3200)

        # Mix thrust and torques to get individual motor PWMs
        pwm = thrust_pwm + np.dot(MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, MIN_PWM, MAX_PWM)

        # Convert PWM to RPM
        rpms = PWM2RPM_SCALE * pwm + PWM2RPM_CONST

        # Clip to Crazyflie max RPM
        rpms = np.clip(rpms, 0, 21702)

        return rpms

    def test_hover(self, duration=10.0, target_altitude=1.0):
        """
        Test 1: MPC Hover Stability

        Args:
            duration: Test duration in seconds
            target_altitude: Hover altitude in meters

        Returns:
            success: Whether test passed
        """
        print(f"\n{'='*60}")
        print(f"Test 1: Hover Stability")
        print(f"Duration: {duration}s, Target altitude: {target_altitude}m")
        print(f"{'='*60}\n")

        # Reset environment
        obs, info = self.env.reset()
        obs = obs[0]  # Extract first drone observation

        # Reset results
        self.results = {k: [] for k in self.results.keys()}

        # Target position
        target_pos = np.array([0.0, 0.0, target_altitude])

        t = 0
        step = 0

        start_time = time.time()

        while t < duration:
            # Get current state
            state = self.get_state_from_obs(obs)

            # Generate reference trajectory
            ref_traj = self.generate_reference_trajectory(state, target_pos, self.mpc.N)

            # Compute MPC control
            control, solve_time, success = self.mpc.compute_control(state, ref_traj)

            # Convert to RPMs
            rpms = self.control_to_rpm(control)

            # Apply control
            obs, reward, terminated, truncated, info = self.env.step(
                np.array([rpms])
            )
            obs = obs[0]

            # Log results
            self.results['time'].append(t)
            self.results['position'].append(state[0:3].copy())
            self.results['velocity'].append(state[3:6].copy())
            self.results['orientation'].append(state[6:9].copy())
            self.results['angular_velocity'].append(state[9:12].copy())
            self.results['control'].append(control.copy())
            self.results['reference'].append(target_pos.copy())
            self.results['solve_time'].append(solve_time)
            self.results['solve_success'].append(success)

            t += self.dt_control
            step += 1

            # Print progress
            if step % 50 == 0:
                pos_error = np.linalg.norm(state[0:3] - target_pos)
                print(f"Step {step:4d} | t={t:5.2f}s | Pos error: {pos_error:.4f}m | "
                      f"Solve time: {solve_time:.2f}ms")

        elapsed_time = time.time() - start_time

        # Analyze results
        print(f"\n{'='*60}")
        print(f"Test Results:")
        print(f"{'='*60}")

        positions = np.array(self.results['position'])
        pos_errors = np.linalg.norm(positions - target_pos, axis=1)

        max_error = np.max(pos_errors)
        rmse = np.sqrt(np.mean(pos_errors**2))
        final_error = pos_errors[-1]

        avg_solve_time = np.mean(self.results['solve_time'])
        max_solve_time = np.max(self.results['solve_time'])
        success_rate = np.mean(self.results['solve_success']) * 100

        print(f"Position Tracking:")
        print(f"  Max error:   {max_error:.4f} m")
        print(f"  RMSE:        {rmse:.4f} m")
        print(f"  Final error: {final_error:.4f} m")
        print(f"\nMPC Performance:")
        print(f"  Avg solve time: {avg_solve_time:.2f} ms")
        print(f"  Max solve time: {max_solve_time:.2f} ms")
        print(f"  Success rate:   {success_rate:.1f}%")
        print(f"\nSimulation:")
        print(f"  Real time:      {elapsed_time:.2f} s")
        print(f"  Simulated time: {duration:.2f} s")
        print(f"  Real-time factor: {duration/elapsed_time:.2f}x")

        # Pass criteria
        pass_rmse = bool(rmse < 0.1)
        pass_final = bool(final_error < 0.05)
        pass_solve = bool(avg_solve_time < 20.0)

        passed = bool(pass_rmse and pass_final and pass_solve)

        print(f"\n{'='*60}")
        print(f"Test Result: {'PASS' if passed else 'FAIL'}")
        print(f"{'='*60}\n")

        # Save results
        metrics = {
            "position_tracking": {
                "max_error_m": float(max_error),
                "rmse_m": float(rmse),
                "final_error_m": float(final_error),
                "status": "PASS" if passed else "FAIL"
            },
            "mpc_performance": {
                "avg_solve_time_ms": float(avg_solve_time),
                "max_solve_time_ms": float(max_solve_time),
                "success_rate_percent": float(success_rate),
                "status": "ACCEPTABLE" if avg_solve_time < 20 else "NEEDS_OPTIMIZATION"
            },
            "simulation": {
                "real_time_seconds": float(elapsed_time),
                "simulated_time_seconds": float(duration),
                "real_time_factor": float(duration/elapsed_time),
                "status": "ACCEPTABLE" if duration/elapsed_time > 0.5 else "SLOW"
            }
        }

        self.save_test_results(
            iteration=self.current_iteration,
            passed=passed,
            metrics=metrics,
            notes="MPC hover test with proper PWM/RPM control allocation"
        )

        # Plot results
        self.plot_hover_results(show_plot=self.show_plots)

        return passed

    def plot_hover_results(self, test_name="Hover Test", show_plot=True):
        """
        Plot comprehensive test results showing all state variables

        Args:
            test_name: Name of the test for plot title and filename
            show_plot: Whether to display the plot window (default: True)
        """
        # Create figure with 18 subplots (6 rows x 3 columns) - states + controls + error
        fig, axes = plt.subplots(6, 3, figsize=(18, 18))

        time = np.array(self.results['time'])
        positions = np.array(self.results['position'])
        velocities = np.array(self.results['velocity'])
        orientations = np.array(self.results['orientation'])
        angular_velocities = np.array(self.results['angular_velocity'])
        controls = np.array(self.results['control'])
        reference = np.array(self.results['reference'])

        # Convert orientations from radians to degrees for better readability
        orientations_deg = np.rad2deg(orientations)
        angular_velocities_deg = np.rad2deg(angular_velocities)
        controls_angular_deg = np.rad2deg(controls[:, 1:4])

        # Use custom formatter for better readability
        from matplotlib.ticker import FuncFormatter
        def format_decimal(x, pos):
            """Format numbers with limited decimal places"""
            if abs(x) < 1e-10:
                return '0'
            elif abs(x) < 1:
                return f'{x:.6f}'
            else:
                return f'{x:.4f}'
        formatter = FuncFormatter(format_decimal)

        # ROW 1: POSITION (X, Y, Z) - Each in separate subplot
        # Plot X Position
        axes[0, 0].plot(time, positions[:, 0], 'b-', linewidth=2.5, label='X')
        axes[0, 0].plot(time, reference[:, 0], 'r--', linewidth=2, alpha=0.7, label='X ref')
        axes[0, 0].set_ylabel('X Position (m)', fontsize=10, fontweight='bold')
        axes[0, 0].set_title('X Position', fontsize=11, fontweight='bold')
        axes[0, 0].legend(loc='best', fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].margins(y=0.2)
        axes[0, 0].yaxis.set_major_formatter(formatter)

        # Plot Y Position
        axes[0, 1].plot(time, positions[:, 1], 'r-', linewidth=2.5, label='Y')
        axes[0, 1].plot(time, reference[:, 1], 'b--', linewidth=2, alpha=0.7, label='Y ref')
        axes[0, 1].set_ylabel('Y Position (m)', fontsize=10, fontweight='bold')
        axes[0, 1].set_title('Y Position', fontsize=11, fontweight='bold')
        axes[0, 1].legend(loc='best', fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].margins(y=0.2)
        axes[0, 1].yaxis.set_major_formatter(formatter)

        # Plot Z Position
        axes[0, 2].plot(time, positions[:, 2], 'g-', linewidth=2.5, label='Z')
        axes[0, 2].plot(time, reference[:, 2], 'r--', linewidth=2, alpha=0.7, label='Z ref')
        axes[0, 2].set_ylabel('Z Position (m)', fontsize=10, fontweight='bold')
        axes[0, 2].set_title('Z Position', fontsize=11, fontweight='bold')
        axes[0, 2].legend(loc='best', fontsize=8)
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].margins(y=0.1)
        axes[0, 2].yaxis.set_major_formatter(formatter)

        # ROW 2: VELOCITY (Vx, Vy, Vz) - Each in separate subplot
        # Plot Vx
        axes[1, 0].plot(time, velocities[:, 0], 'b-', linewidth=2.5, label='Vx')
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        axes[1, 0].set_ylabel('Vx (m/s)', fontsize=10, fontweight='bold')
        axes[1, 0].set_title('X Velocity', fontsize=11, fontweight='bold')
        axes[1, 0].legend(loc='best', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].margins(y=0.2)
        axes[1, 0].yaxis.set_major_formatter(formatter)

        # Plot Vy
        axes[1, 1].plot(time, velocities[:, 1], 'r-', linewidth=2.5, label='Vy')
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        axes[1, 1].set_ylabel('Vy (m/s)', fontsize=10, fontweight='bold')
        axes[1, 1].set_title('Y Velocity', fontsize=11, fontweight='bold')
        axes[1, 1].legend(loc='best', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].margins(y=0.2)
        axes[1, 1].yaxis.set_major_formatter(formatter)

        # Plot Vz
        axes[1, 2].plot(time, velocities[:, 2], 'g-', linewidth=2.5, label='Vz')
        axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        axes[1, 2].set_ylabel('Vz (m/s)', fontsize=10, fontweight='bold')
        axes[1, 2].set_title('Z Velocity', fontsize=11, fontweight='bold')
        axes[1, 2].legend(loc='best', fontsize=8)
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].margins(y=0.2)
        axes[1, 2].yaxis.set_major_formatter(formatter)

        # ROW 3: ORIENTATION (Roll, Pitch, Yaw) - Each in separate subplot
        # Plot Roll
        axes[2, 0].plot(time, orientations_deg[:, 0], 'b-', linewidth=2.5, label='Roll')
        axes[2, 0].axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Roll ref (0°)')
        axes[2, 0].set_ylabel('Roll (deg)', fontsize=10, fontweight='bold')
        axes[2, 0].set_title('Roll Angle', fontsize=11, fontweight='bold')
        axes[2, 0].legend(loc='best', fontsize=8)
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].margins(y=0.2)
        axes[2, 0].yaxis.set_major_formatter(formatter)

        # Plot Pitch
        axes[2, 1].plot(time, orientations_deg[:, 1], 'r-', linewidth=2.5, label='Pitch')
        axes[2, 1].axhline(y=0, color='b', linestyle='--', linewidth=2, alpha=0.7, label='Pitch ref (0°)')
        axes[2, 1].set_ylabel('Pitch (deg)', fontsize=10, fontweight='bold')
        axes[2, 1].set_title('Pitch Angle', fontsize=11, fontweight='bold')
        axes[2, 1].legend(loc='best', fontsize=8)
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].margins(y=0.2)
        axes[2, 1].yaxis.set_major_formatter(formatter)

        # Plot Yaw
        axes[2, 2].plot(time, orientations_deg[:, 2], 'g-', linewidth=2.5, label='Yaw')
        axes[2, 2].axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.7, label='Yaw ref (0°)')
        axes[2, 2].set_ylabel('Yaw (deg)', fontsize=10, fontweight='bold')
        axes[2, 2].set_title('Yaw Angle', fontsize=11, fontweight='bold')
        axes[2, 2].legend(loc='best', fontsize=8)
        axes[2, 2].grid(True, alpha=0.3)
        axes[2, 2].margins(y=0.2)
        axes[2, 2].yaxis.set_major_formatter(formatter)

        # ROW 4: ANGULAR VELOCITY (P, Q, R) - Each in separate subplot
        # Plot P (Roll Rate)
        axes[3, 0].plot(time, angular_velocities_deg[:, 0], 'b-', linewidth=2.5, label='P (Roll rate)')
        axes[3, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        axes[3, 0].set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        axes[3, 0].set_ylabel('P (deg/s)', fontsize=10, fontweight='bold')
        axes[3, 0].set_title('Roll Rate (P)', fontsize=11, fontweight='bold')
        axes[3, 0].legend(loc='best', fontsize=8)
        axes[3, 0].grid(True, alpha=0.3)
        axes[3, 0].margins(y=0.2)
        axes[3, 0].yaxis.set_major_formatter(formatter)

        # Plot Q (Pitch Rate)
        axes[3, 1].plot(time, angular_velocities_deg[:, 1], 'r-', linewidth=2.5, label='Q (Pitch rate)')
        axes[3, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        axes[3, 1].set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        axes[3, 1].set_ylabel('Q (deg/s)', fontsize=10, fontweight='bold')
        axes[3, 1].set_title('Pitch Rate (Q)', fontsize=11, fontweight='bold')
        axes[3, 1].legend(loc='best', fontsize=8)
        axes[3, 1].grid(True, alpha=0.3)
        axes[3, 1].margins(y=0.2)
        axes[3, 1].yaxis.set_major_formatter(formatter)

        # Plot R (Yaw Rate)
        axes[3, 2].plot(time, angular_velocities_deg[:, 2], 'g-', linewidth=2.5, label='R (Yaw rate)')
        axes[3, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        axes[3, 2].set_ylabel('R (deg/s)', fontsize=10, fontweight='bold')
        axes[3, 2].set_title('Yaw Rate (R)', fontsize=11, fontweight='bold')
        axes[3, 2].legend(loc='best', fontsize=8)
        axes[3, 2].grid(True, alpha=0.3)
        axes[3, 2].margins(y=0.2)
        axes[3, 2].yaxis.set_major_formatter(formatter)

        # ROW 5: CONTROL EFFORTS (Thrust, Roll Rate Cmd, Pitch Rate Cmd)
        # Plot Thrust
        axes[4, 0].plot(time, controls[:, 0], 'b-', linewidth=2.5, label='Thrust')
        axes[4, 0].axhline(y=0.027 * 9.81, color='r', linestyle='--', linewidth=2, alpha=0.7, label=f'Hover thrust ({0.027*9.81:.3f}N)')
        axes[4, 0].set_ylabel('Thrust (N)', fontsize=10, fontweight='bold')
        axes[4, 0].set_title('Thrust Command', fontsize=11, fontweight='bold')
        axes[4, 0].legend(loc='best', fontsize=8)
        axes[4, 0].grid(True, alpha=0.3)
        axes[4, 0].margins(y=0.1)
        axes[4, 0].yaxis.set_major_formatter(formatter)

        # Plot Roll Rate Command
        axes[4, 1].plot(time, controls_angular_deg[:, 0], 'b-', linewidth=2.5, label='Roll rate cmd')
        axes[4, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        axes[4, 1].set_ylabel('Roll Rate Cmd (deg/s)', fontsize=10, fontweight='bold')
        axes[4, 1].set_title('Roll Rate Command', fontsize=11, fontweight='bold')
        axes[4, 1].legend(loc='best', fontsize=8)
        axes[4, 1].grid(True, alpha=0.3)
        axes[4, 1].margins(y=0.2)
        axes[4, 1].yaxis.set_major_formatter(formatter)

        # Plot Pitch Rate Command
        axes[4, 2].plot(time, controls_angular_deg[:, 1], 'r-', linewidth=2.5, label='Pitch rate cmd')
        axes[4, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        axes[4, 2].set_ylabel('Pitch Rate Cmd (deg/s)', fontsize=10, fontweight='bold')
        axes[4, 2].set_title('Pitch Rate Command', fontsize=11, fontweight='bold')
        axes[4, 2].legend(loc='best', fontsize=8)
        axes[4, 2].grid(True, alpha=0.3)
        axes[4, 2].margins(y=0.2)
        axes[4, 2].yaxis.set_major_formatter(formatter)

        # ROW 6: Yaw Rate Cmd, Position Error, MPC Solve Time
        # Plot Yaw Rate Command
        axes[5, 0].plot(time, controls_angular_deg[:, 2], 'g-', linewidth=2.5, label='Yaw rate cmd')
        axes[5, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        axes[5, 0].set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        axes[5, 0].set_ylabel('Yaw Rate Cmd (deg/s)', fontsize=10, fontweight='bold')
        axes[5, 0].set_title('Yaw Rate Command', fontsize=11, fontweight='bold')
        axes[5, 0].legend(loc='best', fontsize=8)
        axes[5, 0].grid(True, alpha=0.3)
        axes[5, 0].margins(y=0.2)
        axes[5, 0].yaxis.set_major_formatter(formatter)

        # Plot Position Error (magnitude)
        pos_errors = np.linalg.norm(positions - reference, axis=1)
        axes[5, 1].plot(time, pos_errors, 'r-', linewidth=2.5, label='Position error')
        axes[5, 1].axhline(y=0.1, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (0.1m)')
        axes[5, 1].set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        axes[5, 1].set_ylabel('Error (m)', fontsize=10, fontweight='bold')
        axes[5, 1].set_title('Position Tracking Error', fontsize=11, fontweight='bold')
        axes[5, 1].legend(loc='best', fontsize=8)
        axes[5, 1].grid(True, alpha=0.3)
        axes[5, 1].margins(y=0.1)
        axes[5, 1].yaxis.set_major_formatter(formatter)

        # Plot MPC Solve Time
        solve_times = np.array(self.results['solve_time'])
        axes[5, 2].plot(time, solve_times, 'm-', linewidth=2.5, label='Solve time')
        axes[5, 2].axhline(y=20, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (20ms)')
        axes[5, 2].set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        axes[5, 2].set_ylabel('Solve Time (ms)', fontsize=10, fontweight='bold')
        axes[5, 2].set_title('MPC Computation Time', fontsize=11, fontweight='bold')
        axes[5, 2].legend(loc='best', fontsize=8)
        axes[5, 2].grid(True, alpha=0.3)
        axes[5, 2].margins(y=0.1)
        axes[5, 2].yaxis.set_major_formatter(formatter)

        # Add main title
        fig.suptitle(f'MPC {test_name} - Complete State Visualization',
                     fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # Save figure with test-specific filename
        results_dir = Path(__file__).parent.parent / "results" / "phase_02"
        results_dir.mkdir(parents=True, exist_ok=True)
        filename = test_name.lower().replace(' ', '_').replace('-', '_')
        fig_path = results_dir / f"mpc_{filename}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {fig_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)  # Close figure to free memory

    def save_results_csv(self, filename):
        """
        Export detailed test results to CSV file

        Args:
            filename: Path to save CSV file
        """
        import csv

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'Time_s',
                'Pos_X_m', 'Pos_Y_m', 'Pos_Z_m',
                'Vel_X_ms', 'Vel_Y_ms', 'Vel_Z_ms',
                'Roll_rad', 'Pitch_rad', 'Yaw_rad',
                'AngVel_P_rads', 'AngVel_Q_rads', 'AngVel_R_rads',
                'Thrust_N', 'Roll_Rate_Cmd_rads', 'Pitch_Rate_Cmd_rads', 'Yaw_Rate_Cmd_rads',
                'Ref_X_m', 'Ref_Y_m', 'Ref_Z_m',
                'Position_Error_m',
                'Solve_Time_ms',
                'Solve_Success'
            ])

            # Write data rows
            for i in range(len(self.results['time'])):
                pos = self.results['position'][i]
                vel = self.results['velocity'][i]
                att = self.results['orientation'][i]
                ang_vel = self.results['angular_velocity'][i]
                ctrl = self.results['control'][i]
                ref = self.results['reference'][i]

                # Calculate position error
                pos_error = np.linalg.norm(pos - ref)

                row = [
                    f"{self.results['time'][i]:.4f}",
                    f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}",
                    f"{vel[0]:.6f}", f"{vel[1]:.6f}", f"{vel[2]:.6f}",
                    f"{att[0]:.6f}", f"{att[1]:.6f}", f"{att[2]:.6f}",
                    f"{ang_vel[0]:.6f}", f"{ang_vel[1]:.6f}", f"{ang_vel[2]:.6f}",
                    f"{ctrl[0]:.6f}", f"{ctrl[1]:.6f}", f"{ctrl[2]:.6f}", f"{ctrl[3]:.6f}",
                    f"{ref[0]:.6f}", f"{ref[1]:.6f}", f"{ref[2]:.6f}",
                    f"{pos_error:.6f}",
                    f"{self.results['solve_time'][i]:.4f}",
                    f"{int(self.results['solve_success'][i])}"
                ]
                writer.writerow(row)

        print(f"CSV results saved to: {filename}")

    def save_test_results(self, iteration, passed, metrics, notes=""):
        """
        Save detailed test results to JSON file

        Args:
            iteration: Test iteration number
            passed: Whether test passed
            metrics: Dictionary of performance metrics
            notes: Additional notes about the test
        """
        import json
        from pathlib import Path

        results_dir = Path(__file__).parent.parent / "results" / "phase_02"
        results_dir.mkdir(parents=True, exist_ok=True)

        result_data = {
            "iteration": iteration,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_name": "MPC Hover Test",
            "configuration": {
                "mpc_horizon": self.mpc.N,
                "timestep": self.mpc.dt,
                "Q_weights": self.mpc.Q.diagonal().tolist(),
                "R_weights": self.mpc.R.diagonal().tolist(),
                "control_frequency_hz": 48,
                "physics_frequency_hz": 240
            },
            "target": {
                "position": [0.0, 0.0, 1.0],
                "duration_seconds": 10.0
            },
            "results": metrics,
            "pass_criteria": {
                "position_rmse_target_m": 0.1,
                "position_rmse_achieved_m": metrics.get("position_tracking", {}).get("rmse_m", 0),
                "passed": passed
            },
            "notes": notes
        }

        filename = results_dir / f"test_iteration_{iteration:02d}.json"
        with open(filename, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"\nResults saved to: {filename}")

        # Also save CSV version
        csv_filename = results_dir / f"test_iteration_{iteration:02d}.csv"
        self.save_results_csv(csv_filename)

    def generate_figure8_trajectory(self, t, z_altitude=1.0, radius=0.5, period=10.0):
        """
        Generate figure-8 (lemniscate) trajectory at constant altitude

        Args:
            t: Current time
            z_altitude: Altitude to fly at (m)
            radius: Radius of the figure-8 (m)
            period: Period of one complete figure-8 (seconds)

        Returns:
            ref_pos: Reference position [x, y, z]
            ref_vel: Reference velocity [vx, vy, vz]
        """
        # Lemniscate of Bernoulli parametric equations
        omega = 2 * np.pi / period  # Angular frequency
        theta = omega * t

        # Position (figure-8 in XY plane)
        scale = radius
        x = scale * np.sin(theta)
        y = scale * np.sin(theta) * np.cos(theta)  # Figure-8 shape
        z = z_altitude

        # Velocity (derivatives)
        vx = scale * omega * np.cos(theta)
        vy = scale * omega * (np.cos(theta)**2 - np.sin(theta)**2)
        vz = 0.0

        ref_pos = np.array([x, y, z])
        ref_vel = np.array([vx, vy, vz])

        return ref_pos, ref_vel

    def test_figure8(self, duration=20.0, z_altitude=1.0):
        """
        Test 2: Figure-8 Trajectory Tracking

        Args:
            duration: Test duration in seconds
            z_altitude: Altitude for figure-8 (meters)

        Returns:
            success: Whether test passed
        """
        print(f"\n{'='*60}")
        print(f"Test 2: Figure-8 Trajectory Tracking")
        print(f"Duration: {duration}s, Altitude: {z_altitude}m")
        print(f"{'='*60}\n")

        # Reset environment
        obs, info = self.env.reset()
        obs = obs[0]

        # Reset results
        self.results = {k: [] for k in self.results.keys()}

        t = 0
        step = 0
        start_time = time.time()

        while t < duration:
            # Get current state
            state = self.get_state_from_obs(obs)

            # Generate figure-8 reference trajectory
            ref_pos, ref_vel = self.generate_figure8_trajectory(t, z_altitude)

            # Build reference trajectory for MPC horizon
            ref_traj = np.zeros((12, self.mpc.N + 1))
            for k in range(self.mpc.N + 1):
                t_future = t + k * self.mpc.dt
                pos_k, vel_k = self.generate_figure8_trajectory(t_future, z_altitude)
                ref_traj[0:3, k] = pos_k
                ref_traj[3:6, k] = vel_k
                # Zero orientation and angular velocity
                ref_traj[6:12, k] = 0.0

            # Compute MPC control
            control, solve_time, success = self.mpc.compute_control(state, ref_traj)

            # Convert to RPMs
            rpms = self.control_to_rpm(control)

            # Apply control
            obs, reward, terminated, truncated, info = self.env.step(np.array([rpms]))
            obs = obs[0]

            # Log results
            self.results['time'].append(t)
            self.results['position'].append(state[0:3].copy())
            self.results['velocity'].append(state[3:6].copy())
            self.results['orientation'].append(state[6:9].copy())
            self.results['angular_velocity'].append(state[9:12].copy())
            self.results['control'].append(control.copy())
            self.results['reference'].append(ref_pos.copy())
            self.results['solve_time'].append(solve_time)
            self.results['solve_success'].append(success)

            t += self.dt_control
            step += 1

            # Print progress
            if step % 50 == 0:
                pos_error = np.linalg.norm(state[0:3] - ref_pos)
                print(f"Step {step:4d} | t={t:5.2f}s | Pos error: {pos_error:.4f}m | "
                      f"Solve time: {solve_time:.2f}ms")

        elapsed_time = time.time() - start_time

        # Analyze results
        print(f"\n{'='*60}")
        print(f"Test Results:")
        print(f"{'='*60}")

        positions = np.array(self.results['position'])
        references = np.array(self.results['reference'])
        pos_errors = np.linalg.norm(positions - references, axis=1)

        max_error = np.max(pos_errors)
        rmse = np.sqrt(np.mean(pos_errors**2))
        final_error = pos_errors[-1]

        avg_solve_time = np.mean(self.results['solve_time'])
        max_solve_time = np.max(self.results['solve_time'])
        success_rate = np.mean(self.results['solve_success']) * 100

        print(f"Position Tracking:")
        print(f"  Max error:   {max_error:.4f} m")
        print(f"  RMSE:        {rmse:.4f} m")
        print(f"  Final error: {final_error:.4f} m")
        print(f"\nMPC Performance:")
        print(f"  Avg solve time: {avg_solve_time:.2f} ms")
        print(f"  Max solve time: {max_solve_time:.2f} ms")
        print(f"  Success rate:   {success_rate:.1f}%")
        print(f"\nSimulation:")
        print(f"  Real time:      {elapsed_time:.2f} s")
        print(f"  Simulated time: {duration:.2f} s")
        print(f"  Real-time factor: {duration/elapsed_time:.2f}x")

        # Pass criteria (more lenient for complex trajectory)
        pass_rmse = bool(rmse < 0.2)  # 20cm average error acceptable for figure-8
        pass_solve = bool(avg_solve_time < 30.0)  # 30ms acceptable for complex trajectory
        passed = bool(pass_rmse and pass_solve)

        print(f"\n{'='*60}")
        print(f"Test Result: {'PASS' if passed else 'FAIL'}")
        print(f"{'='*60}\n")

        # Save results
        metrics = {
            "position_tracking": {
                "max_error_m": float(max_error),
                "rmse_m": float(rmse),
                "final_error_m": float(final_error),
                "status": "PASS" if passed else "FAIL"
            },
            "mpc_performance": {
                "avg_solve_time_ms": float(avg_solve_time),
                "max_solve_time_ms": float(max_solve_time),
                "success_rate_percent": float(success_rate)
            },
            "simulation": {
                "real_time_seconds": float(elapsed_time),
                "simulated_time_seconds": float(duration),
                "real_time_factor": float(duration/elapsed_time)
            }
        }

        self.save_test_results(
            iteration=self.current_iteration,
            passed=passed,
            metrics=metrics,
            notes="MPC figure-8 trajectory tracking test at z=" + str(z_altitude) + "m"
        )

        # Plot results
        self.plot_hover_results(test_name="Figure-8 Test", show_plot=self.show_plots)

        return passed

    def close(self):
        """Clean up environment"""
        self.env.close()


def main():
    """Run MPC controller tests"""
    import argparse

    parser = argparse.ArgumentParser(
        description="MPC Controller Test Suite for Quadrotor Trajectory Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run test with visualization (3D PyBullet window)
  python tests/test_mpc_controller.py --gui

  # Run test without visualization (faster, headless)
  python tests/test_mpc_controller.py

  # Run longer test with visualization
  python tests/test_mpc_controller.py --gui --duration 20

Results:
  - Plots will be saved to: results/phase_02/mpc_hover_test.png
  - JSON results saved to: results/phase_02/test_iteration_XX.json
  - CSV results saved to: results/phase_02/test_iteration_XX.csv
        """
    )
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Enable PyBullet 3D visualization window'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=10.0,
        help='Test duration in seconds (default: 10.0)'
    )
    parser.add_argument(
        '--altitude',
        type=float,
        default=1.0,
        help='Target hover altitude in meters (default: 1.0)'
    )
    parser.add_argument(
        '--iteration',
        type=int,
        default=2,
        help='Test iteration number for file naming (default: 2)'
    )
    parser.add_argument(
        '--test',
        type=str,
        default='hover',
        choices=['hover', 'figure8'],
        help='Test type to run: hover or figure8 (default: hover)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable matplotlib plot windows (still saves PNG files)'
    )

    args = parser.parse_args()

    print("="*60)
    print("MPC Controller Test Suite")
    print("="*60)
    print(f"Test Type: {args.test.upper()}")
    print(f"GUI Mode: {'ENABLED' if args.gui else 'DISABLED (headless)'}")
    print(f"Test Duration: {args.duration}s")
    print(f"Target Altitude: {args.altitude}m")
    print("="*60)

    # Check if config exists
    config_path = "configs/mpc_crazyflie.yaml"
    if not Path(config_path).exists():
        print(f"ERROR: Config file not found: {config_path}")
        print("Please ensure the config file exists.")
        return

    # Create test suite
    try:
        test = MPCTest(config_path=config_path, gui=args.gui, show_plots=not args.no_plots)

        # Update iteration number in test
        test.current_iteration = args.iteration

        # Run selected test
        if args.test == 'hover':
            success = test.test_hover(duration=args.duration, target_altitude=args.altitude)
        elif args.test == 'figure8':
            success = test.test_figure8(duration=args.duration, z_altitude=args.altitude)
        else:
            print(f"ERROR: Unknown test type: {args.test}")
            return 1

        # Clean up
        test.close()

        print("\n" + "="*60)
        print(f"Overall Test Result: {'PASS' if success else 'FAIL'}")
        print("="*60)

        return 0 if success else 1

    except Exception as e:
        print(f"\nERROR: Test failed with exception:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
