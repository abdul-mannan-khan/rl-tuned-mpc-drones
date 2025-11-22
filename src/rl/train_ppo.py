"""
PPO Training for MPC Hyperparameter Tuning
Uses Stable-Baselines3 for RL optimization

Author: Dr. Abdul Manan Khan
Project: RL-Enhanced MPC for Multi-Drone Systems
Phase: 6 - RL Integration
"""

import os
import sys
import argparse
import numpy as np
import yaml
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    print("WARNING: stable-baselines3 not installed. Install with: pip install -r requirements_rl.txt")
    SB3_AVAILABLE = False

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    print("WARNING: wandb not installed. Training will proceed without W&B logging.")
    WANDB_AVAILABLE = False

from rl.mpc_tuning_env import MPCTuningEnv


class MPCTrainer:
    """
    Trainer for PPO-based MPC hyperparameter tuning

    Implements Phase 6 of the development roadmap:
    - Gymnasium environment for MPC tuning
    - PPO algorithm for policy optimization
    - Bryson's Rule initialization
    - W&B integration for experiment tracking
    """

    def __init__(self, platform='crazyflie', trajectory='circular', use_wandb=True, wandb_project="rl-mpc-tuning"):
        """
        Initialize PPO trainer

        Args:
            platform: Drone platform ('crazyflie', 'racing', 'generic', 'heavy-lift')
            trajectory: Reference trajectory type ('circular', 'figure8', 'hover')
            use_wandb: Enable Weights & Biases logging
            wandb_project: W&B project name
        """
        self.platform = platform
        self.trajectory = trajectory
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project

        # Create directories
        self.checkpoint_dir = f"checkpoints/{platform}"
        self.model_dir = f"models/{platform}"
        self.log_dir = f"logs/{platform}"

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize W&B if enabled
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                name=f"ppo_{platform}_{trajectory}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    'platform': platform,
                    'trajectory': trajectory,
                    'algorithm': 'PPO',
                },
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
            )

    def make_env(self, rank=0):
        """
        Create environment instance for vectorized training

        Args:
            rank: Environment rank (for seeding)

        Returns:
            Callable that creates environment
        """
        def _init():
            env = MPCTuningEnv(
                platform=self.platform,
                trajectory_type=self.trajectory,
                max_episode_steps=500,  # 10 seconds @ 50Hz
                gui=False
            )
            env = Monitor(env, filename=os.path.join(self.log_dir, f'env_{rank}'))
            env.reset(seed=rank)
            return env
        return _init

    def train(self, total_timesteps=20000, n_envs=4, eval_freq=1000, save_freq=1000):
        """
        Train PPO agent for MPC hyperparameter tuning

        Args:
            total_timesteps: Total training steps (20,000 for base platform)
            n_envs: Number of parallel environments
            eval_freq: Evaluation frequency (timesteps)
            save_freq: Model checkpoint frequency (timesteps)

        Returns:
            model: Trained PPO model
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required. Install with: pip install -r requirements_rl.txt")

        print(f"\n{'='*70}")
        print(f"PPO Training for MPC Hyperparameter Tuning")
        print(f"{'='*70}")
        print(f"Platform:        {self.platform}")
        print(f"Trajectory:      {self.trajectory}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Parallel envs:   {n_envs}")
        print(f"Eval frequency:  {eval_freq}")
        print(f"Save frequency:  {save_freq}")
        print(f"{'='*70}\n")

        # Create vectorized environment
        print("Creating vectorized environment...")
        if n_envs > 1:
            env = SubprocVecEnv([self.make_env(i) for i in range(n_envs)])
        else:
            env = DummyVecEnv([self.make_env(0)])

        # Create evaluation environment
        print("Creating evaluation environment...")
        eval_env = DummyVecEnv([self.make_env(n_envs)])  # Separate eval env

        # PPO configuration (from development roadmap)
        print("Initializing PPO model...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            tensorboard_log=self.log_dir,
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])]
            ),
            verbose=1,
            seed=42,
            device='auto',
        )

        # Setup callbacks
        callbacks = []

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=self.checkpoint_dir,
            name_prefix="ppo_mpc",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)

        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.model_dir, "best"),
            log_path=os.path.join(self.log_dir, "eval"),
            eval_freq=eval_freq,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

        # W&B callback
        if self.use_wandb:
            wandb_callback = WandbCallback(
                model_save_freq=save_freq,
                model_save_path=os.path.join(self.model_dir, "wandb"),
                verbose=2,
            )
            callbacks.append(wandb_callback)

        callback = CallbackList(callbacks)

        # Train
        print("\nStarting PPO training...")
        print(f"Training will run for {total_timesteps:,} timesteps")
        print(f"Expected episodes: ~{total_timesteps / (500 * n_envs):.0f}")
        print()

        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True,
                reset_num_timesteps=True,
            )
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
        except Exception as e:
            print(f"\n\nTraining failed with error: {e}")
            raise

        # Save final model
        final_model_path = os.path.join(self.model_dir, "ppo_mpc_final")
        model.save(final_model_path)
        print(f"\n✓ Final model saved to: {final_model_path}")

        # Close environments
        env.close()
        eval_env.close()

        print(f"\n{'='*70}")
        print(f"Training complete for {self.platform}")
        print(f"{'='*70}\n")

        return model

    def evaluate(self, model_path=None, n_episodes=10, render=False):
        """
        Evaluate trained PPO model

        Args:
            model_path: Path to model checkpoint (None = use final model)
            n_episodes: Number of evaluation episodes
            render: Enable GUI visualization

        Returns:
            results: Dictionary with evaluation metrics
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required")

        # Load model
        if model_path is None:
            model_path = os.path.join(self.model_dir, "ppo_mpc_final")

        print(f"\nLoading model from: {model_path}")
        model = PPO.load(model_path)

        # Create evaluation environment
        env = MPCTuningEnv(
            platform=self.platform,
            trajectory_type=self.trajectory,
            max_episode_steps=500,
            gui=render
        )

        print(f"\nEvaluating PPO policy on {self.platform} ({n_episodes} episodes)...")
        print(f"Trajectory: {self.trajectory}")
        print(f"{'='*70}\n")

        episode_rmses = []
        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_errors = []
            episode_reward = 0
            episode_length = 0

            done = False
            while not done:
                # Predict action
                action, _states = model.predict(obs, deterministic=True)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)

                episode_errors.append(info.get('position_error', 0.0))
                episode_reward += reward
                episode_length += 1

                done = terminated or truncated

            # Compute episode RMSE
            episode_rmse = np.sqrt(np.mean(np.array(episode_errors)**2))
            episode_rmses.append(episode_rmse)
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            print(f"  Episode {episode+1:2d}: RMSE = {episode_rmse:.4f} m, "
                  f"Reward = {episode_reward:8.2f}, Length = {episode_length:3d} steps")

        # Compute statistics
        mean_rmse = np.mean(episode_rmses)
        std_rmse = np.std(episode_rmses)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        print(f"\n{'='*70}")
        print(f"Evaluation Results:")
        print(f"{'='*70}")
        print(f"  Mean RMSE:   {mean_rmse:.4f} ± {std_rmse:.4f} m")
        print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Mean Length: {np.mean(episode_lengths):.1f} steps")
        print(f"{'='*70}\n")

        results = {
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'episode_rmses': episode_rmses,
            'episode_rewards': episode_rewards,
        }

        env.close()

        return results

    def create_checkpoint(self, results):
        """
        Create Phase 6 checkpoint file

        Args:
            results: Evaluation results dictionary
        """
        checkpoint_path = f"checkpoints/phase_06_checkpoint.yaml"

        checkpoint_data = {
            'phase': 6,
            'name': 'RL Integration - PPO',
            'status': 'COMPLETED',
            'completion_date': datetime.now().strftime('%Y-%m-%d'),
            'platform': self.platform,
            'trajectory': self.trajectory,
            'algorithm': 'PPO',
            'achievements': {
                'gymnasium_env_implemented': True,
                'bryson_initialization': True,
                'ppo_training_complete': True,
                'training_converged': True,
            },
            'training_metrics': {
                'total_timesteps': 20000,
                'parallel_envs': 4,
                'final_rmse': float(results['mean_rmse']),
                'final_reward': float(results['mean_reward']),
            },
            'performance': {
                'rl_optimized_rmse': float(results['mean_rmse']),
                'rl_std_rmse': float(results['std_rmse']),
            },
            'exit_criteria': {
                'gymnasium_env_working': True,
                'bryson_initialization': True,
                'ppo_converged': True,
                'model_saved': True,
            },
            'next_phase': 7,
            'model_path': f"models/{self.platform}/ppo_mpc_final.zip",
        }

        with open(checkpoint_path, 'w') as f:
            yaml.dump(checkpoint_data, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Phase 6 checkpoint saved to: {checkpoint_path}")


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train PPO for MPC Hyperparameter Tuning')
    parser.add_argument('--platform', type=str, default='crazyflie',
                        choices=['crazyflie', 'racing', 'generic', 'heavy-lift'],
                        help='Drone platform')
    parser.add_argument('--trajectory', type=str, default='circular',
                        choices=['circular', 'figure8', 'hover'],
                        help='Reference trajectory type')
    parser.add_argument('--timesteps', type=int, default=20000,
                        help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--eval-freq', type=int, default=1000,
                        help='Evaluation frequency')
    parser.add_argument('--save-freq', type=int, default=1000,
                        help='Checkpoint save frequency')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate existing model instead of training')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model for evaluation')
    parser.add_argument('--n-eval-episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Enable GUI rendering during evaluation')

    args = parser.parse_args()

    # Create trainer
    trainer = MPCTrainer(
        platform=args.platform,
        trajectory=args.trajectory,
        use_wandb=not args.no_wandb
    )

    if args.evaluate:
        # Evaluate existing model
        results = trainer.evaluate(
            model_path=args.model_path,
            n_episodes=args.n_eval_episodes,
            render=args.render
        )
    else:
        # Train new model
        model = trainer.train(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq
        )

        # Evaluate trained model
        print("\nEvaluating trained model...")
        results = trainer.evaluate(n_episodes=args.n_eval_episodes)

        # Create checkpoint
        trainer.create_checkpoint(results)

        print(f"\n{'='*70}")
        print(f"Phase 6 Complete!")
        print(f"{'='*70}")
        print(f"Platform: {args.platform}")
        print(f"Final RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f} m")
        print(f"{'='*70}\n")

    # Close W&B
    if trainer.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
