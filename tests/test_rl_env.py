"""
Test script for RL environment
Verifies MPCTuningEnv works correctly before training

Author: Dr. Abdul Manan Khan
Phase: 6 - RL Integration
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.mpc_tuning_env import MPCTuningEnv


def test_environment_creation():
    """Test environment can be created"""
    print("Testing environment creation...")
    env = MPCTuningEnv(platform='crazyflie', trajectory_type='circular', gui=False)
    print(f"  [OK] Environment created successfully")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    env.close()
    return True


def test_environment_reset():
    """Test environment reset"""
    print("\nTesting environment reset...")
    env = MPCTuningEnv(platform='crazyflie', trajectory_type='circular', gui=False)
    obs, info = env.reset()
    print(f"  [OK] Environment reset successfully")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Observation dtype: {obs.dtype}")
    print(f"  - Info keys: {list(info.keys())}")
    env.close()
    return True


def test_environment_step():
    """Test environment step"""
    print("\nTesting environment step...")
    env = MPCTuningEnv(platform='crazyflie', trajectory_type='circular', gui=False)
    obs, info = env.reset()

    # Random action
    action = env.action_space.sample()

    obs_new, reward, terminated, truncated, info = env.step(action)

    print(f"  [OK] Environment step successful")
    print(f"  - New observation shape: {obs_new.shape}")
    print(f"  - Reward: {reward:.4f}")
    print(f"  - Terminated: {terminated}")
    print(f"  - Truncated: {truncated}")
    print(f"  - Position error: {info.get('position_error', 0.0):.4f} m")
    print(f"  - Solve time: {info.get('solve_time_ms', 0.0):.2f} ms")

    env.close()
    return True


def test_episode_rollout():
    """Test full episode rollout"""
    print("\nTesting full episode rollout (10 steps)...")
    env = MPCTuningEnv(platform='crazyflie', trajectory_type='circular', gui=False)
    obs, info = env.reset()

    episode_rewards = []
    episode_errors = []

    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        episode_rewards.append(reward)
        episode_errors.append(info.get('position_error', 0.0))

        if terminated or truncated:
            print(f"  ! Episode terminated at step {step+1}")
            break

    mean_reward = np.mean(episode_rewards)
    mean_error = np.mean(episode_errors)

    print(f"  [OK] Episode completed")
    print(f"  - Steps: {len(episode_rewards)}")
    print(f"  - Mean reward: {mean_reward:.4f}")
    print(f"  - Mean position error: {mean_error:.4f} m")

    env.close()
    return True


def test_bryson_initialization():
    """Test Bryson's Rule initialization"""
    print("\nTesting Bryson's Rule initialization...")
    env = MPCTuningEnv(platform='crazyflie', trajectory_type='circular', gui=False)

    print(f"  [OK] Bryson's Rule weights computed")
    print(f"  - Q_bryson shape: {env.Q_bryson.shape}")
    print(f"  - Q_bryson values: {env.Q_bryson}")
    print(f"  - R_bryson shape: {env.R_bryson.shape}")
    print(f"  - R_bryson values: {env.R_bryson}")

    env.close()
    return True


def main():
    """Run all tests"""
    print("="*70)
    print("RL Environment Test Suite")
    print("="*70)

    tests = [
        test_environment_creation,
        test_environment_reset,
        test_bryson_initialization,
        test_environment_step,
        test_episode_rollout,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  [FAIL] Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "="*70)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("="*70)

    if all(results):
        print("\n[OK] All tests passed! Environment is ready for training.")
        return 0
    else:
        print("\n[FAIL] Some tests failed. Please fix errors before training.")
        return 1


if __name__ == "__main__":
    exit(main())
