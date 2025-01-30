import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
import ray
from ray.tune.registry import register_env
import sys
from no_thanks_env import NoThanksEnv

# ----------------------------
# Initialize and Configure RLlib Trainer
# ----------------------------
def main(num_players=3, iterations=100, checkpoint_path=None):
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train No Thanks! RL Agent.")
    parser.add_argument(
        "--num_players",
        type=int,
        default=num_players,
        help="Total number of players in the game.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=iterations,
        help="Number of training iterations.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=checkpoint_path,
        help="Path to the checkpoint to resume training from.",
    )
    args = parser.parse_args()

    # Update variables with parsed arguments
    num_players = args.num_players
    iterations = args.iterations
    checkpoint_path = args.checkpoint_path

    # Initialize Ray
    ray.shutdown()  # Shutdown any existing Ray instances
    ray.init(ignore_reinit_error=True)

    # Register the environment
    def env_creator(config):
        return NoThanksEnv(num_players=num_players)

    register_env("no_thanks_env", env_creator)

    # Create a temporary environment instance to access observation and action spaces
    temp_env = NoThanksEnv(num_players=num_players)
    observation_space = temp_env.observation_space
    action_space = temp_env.action_space

    # Define separate policies for each agent
    policies = {
        f"policy_{i}": (
            None,
            observation_space,
            action_space,
            {
                "model": {
                    "fcnet_hiddens": [256, 256, 256],
                    "fcnet_activation": "relu",
                },
                "framework": "torch",
            },
        ) for i in range(num_players)
    }

    # Policy mapping function
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        # Extract the numeric part from the agent ID
        agent_num = int(agent_id.split('_')[1])
        return f"policy_{agent_num}"

    # Configure RLlib trainer
    config = (
        PPOConfig()
        .environment(env="no_thanks_env")
        .framework("torch")
        .rollouts(
            num_rollout_workers=6,
            rollout_fragment_length="auto",
        )
        .training(
            train_batch_size=4096,
            sgd_minibatch_size=512,
            num_sgd_iter=20,  # Increased from 10 to encourage more thorough learning (but too high may lead to overfitting)
            lr=1e-4,  # Reduced from 1e-3
            lr_schedule=[
                [0, 1e-4],
                [4000, 5e-5],
                [10000, 1e-5],
            ],
            clip_param=0.2,
            entropy_coeff=0.005,  # Reduce from 0.01 to 0.005; to balance exploration and exploitation
            entropy_coeff_schedule=[
                [0, 0.005],
                [5000, 0.002],
                [10000, 0.001],
                [14000, 0.0005],
            ],
            vf_clip_param=5.0,
            vf_loss_coeff=1.0, #To balance the importance between policy loss and value function loss
            use_gae=True, # Generalized Advantage Estimation reduces variance in advantage estimates
            lambda_=0.95, # gea_lambda
            grad_clip=0.5, #helps Nan issue in training (moved up from .5)
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .resources(
            num_gpus=0
        )
    )

    # Create the PPO trainer
    trainer = config.build()

    # Restore from checkpoint if provided and not empty string
    if checkpoint_path and checkpoint_path != "":
        try:
            trainer.restore(checkpoint_path)
            print(f"Restored trainer state from checkpoint: {checkpoint_path}")

        except Exception as e:
            print(f"Failed to restore from checkpoint: {e}")
            sys.exit(1)

    print(f"Starting training for {iterations} iterations with {num_players} players.")

    # Initialize variables to track the best model
    best_reward_mean = {policy_id: float('-inf') for policy_id in policies.keys()}
    best_checkpoint_path = {policy_id: None for policy_id in policies.keys()}

    log_file = open('best_model_log.txt', 'w')

    # Open a log file to save the best checkpoints
    for i in range(1, iterations + 1):
        result = trainer.train()

        print(f"Iteration: {i}")
        
        # Iterate over each policy to log metrics
        for policy_id in policies.keys():
            learner_stats = result['info']['learner'][policy_id]['learner_stats']
            policy_loss = learner_stats.get('policy_loss', 'N/A')
            vf_loss = learner_stats.get('vf_loss', 'N/A')
            entropy = learner_stats.get('entropy', 'N/A')
            print(f"  {policy_id} - Policy Loss: {policy_loss}")
            print(f"  {policy_id} - Value Function Loss: {vf_loss}")
            print(f"  {policy_id} - Entropy: {entropy}")

        # Access metrics from 'env_runners' for each policy
        env_runner_metrics = result.get('env_runners', {})
        if env_runner_metrics:
            for policy_id, metrics in env_runner_metrics.get('policy_reward_mean', {}).items():
                print(f"  Policy Reward Mean ({policy_id}): {metrics:.2f}")
                # Check if current reward mean is better than the best so far for this policy
                if metrics > best_reward_mean[policy_id]:
                    best_reward_mean[policy_id] = metrics
                    # Save checkpoint
                    save_result = trainer.save()
                    checkpoint_path = save_result.checkpoint.path
                    best_checkpoint_path[policy_id] = checkpoint_path
                    print(f"  New best model for {policy_id} found at iteration {i}, checkpoint saved at {checkpoint_path}")
                    # Log the path and score context
                    log_file.write(f"Iteration {i}, Best Reward Mean for {policy_id}: {best_reward_mean[policy_id]:.2f}, Checkpoint Path: {checkpoint_path}\n")
                    log_file.flush()  # Ensure it's written to disk

        # Save the model at specified intervals (optional)
        # if i % 500 == 0:
        #     save_result = trainer.save()
        #     checkpoint_path = save_result.checkpoint.path
        #     print(f"  Checkpoint saved at {checkpoint_path}")

        print("-" * 50)  # Separator for readability

    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
