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

    def __init__(self, config_path="configs/mpc_crazyflie.yaml", gui=False):
        """
        Initialize test suite

        Args:
            config_path: Path to MPC configuration file
            gui: Whether to show PyBullet GUI
        """
        self.config_path = config_path
        self.gui = gui

        # Create MPC controller
        print(f"Loading MPC controller from {config_path}...")
        self.mpc = MPCController(config_path)
        print(f"MPC initialized: N={self.mpc.N}, dt={self.mpc.dt}")

        # Create PyBullet environment
        print("Creating PyBullet environment...")
        self.env = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
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
        Convert MPC control to motor RPMs for PyBullet

        Args:
            control: [thrust, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd]

        Returns:
            rpms: 4D array of motor RPMs
        """
        # Simple control allocation
        # This is a placeholder - should match actual drone mixing
        thrust = control[0]

        # Convert thrust to RPM (simplified)
        # For CF2X: thrust = k * rpm^2, where k is motor constant
        base_rpm = np.sqrt(thrust / (4 * 1.0e-9)) if thrust > 0 else 0

        # Apply angular rate commands (simplified differential)
        roll_cmd = control[1]
        pitch_cmd = control[2]
        yaw_cmd = control[3]

        # Differential allocation (simplified)
        rpm1 = base_rpm + roll_cmd * 1000 + pitch_cmd * 1000 - yaw_cmd * 1000
        rpm2 = base_rpm - roll_cmd * 1000 + pitch_cmd * 1000 + yaw_cmd * 1000
        rpm3 = base_rpm - roll_cmd * 1000 - pitch_cmd * 1000 - yaw_cmd * 1000
        rpm4 = base_rpm + roll_cmd * 1000 - pitch_cmd * 1000 + yaw_cmd * 1000

        rpms = np.array([rpm1, rpm2, rpm3, rpm4])

        # Clip to valid range
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
        pass_rmse = rmse < 0.1
        pass_final = final_error < 0.05
        pass_solve = avg_solve_time < 20.0

        passed = pass_rmse and pass_final and pass_solve

        print(f"\n{'='*60}")
        print(f"Test Result: {'PASS' if passed else 'FAIL'}")
        print(f"{'='*60}\n")

        # Plot results
        self.plot_hover_results()

        return passed

    def plot_hover_results(self):
        """Plot hover test results"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        time = np.array(self.results['time'])
        positions = np.array(self.results['position'])
        reference = np.array(self.results['reference'])

        # Position plot
        axes[0].plot(time, positions[:, 0], label='X', linewidth=2)
        axes[0].plot(time, positions[:, 1], label='Y', linewidth=2)
        axes[0].plot(time, positions[:, 2], label='Z', linewidth=2)
        axes[0].axhline(y=reference[0, 2], color='r', linestyle='--',
                       label='Z Reference', alpha=0.7)
        axes[0].set_ylabel('Position (m)', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('MPC Hover Test - Position Tracking', fontsize=14, fontweight='bold')

        # Velocity plot
        velocities = np.array(self.results['velocity'])
        axes[1].plot(time, velocities[:, 0], label='Vx', linewidth=2)
        axes[1].plot(time, velocities[:, 1], label='Vy', linewidth=2)
        axes[1].plot(time, velocities[:, 2], label='Vz', linewidth=2)
        axes[1].set_ylabel('Velocity (m/s)', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # Control plot
        controls = np.array(self.results['control'])
        axes[2].plot(time, controls[:, 0], label='Thrust', linewidth=2)
        axes[2].plot(time, controls[:, 1], label='Roll rate cmd', linewidth=1.5)
        axes[2].plot(time, controls[:, 2], label='Pitch rate cmd', linewidth=1.5)
        axes[2].plot(time, controls[:, 3], label='Yaw rate cmd', linewidth=1.5)
        axes[2].set_xlabel('Time (s)', fontsize=12)
        axes[2].set_ylabel('Control', fontsize=12)
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        results_dir = Path(__file__).parent.parent / "results" / "phase_02"
        results_dir.mkdir(parents=True, exist_ok=True)
        fig_path = results_dir / "mpc_hover_test.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {fig_path}")

        plt.show()

    def close(self):
        """Clean up environment"""
        self.env.close()


def main():
    """Run MPC controller tests"""
    print("="*60)
    print("MPC Controller Test Suite")
    print("="*60)

    # Check if config exists
    config_path = "configs/mpc_crazyflie.yaml"
    if not Path(config_path).exists():
        print(f"ERROR: Config file not found: {config_path}")
        print("Please ensure the config file exists.")
        return

    # Create test suite
    try:
        test = MPCTest(config_path=config_path, gui=False)

        # Run hover test
        success = test.test_hover(duration=10.0, target_altitude=1.0)

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
