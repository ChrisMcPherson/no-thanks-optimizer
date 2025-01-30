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

    # Create a shared policy for all agents
    temp_env = NoThanksEnv(num_players=num_players)
    # policies = {
    #     "shared_policy": (
    #         None,  # Use default policy class (PPO)
    #         temp_env.observation_space,  # Updated observation space
    #         temp_env.action_space,  # Action space remains the same
    #         {
    #             "model": {
    #                 "fcnet_hiddens": [256, 256, 256],  # Network architecture
    #                 "fcnet_activation": "relu",
    #             },
    #             "framework": "torch",
    #         },
    #     )
    # }

    # # Policy mapping function: All agents use the shared policy
    # def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    #     return "shared_policy"
    
    policies = {
        f"policy_{i}": (
            None,
            temp_env.observation_space,
            temp_env.action_space,
            {
                "model": {
                    "fcnet_hiddens": [256, 256, 256],
                    "fcnet_activation": "relu",
                },
                "framework": "torch",
            },
        ) for i in range(num_players)
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
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
            train_batch_size=8192,
            sgd_minibatch_size=1024,
            num_sgd_iter=40,  # Increased from 10 to encourage more thorough learning
            lr=1e-3, #5e-5 # Increased from 5e-5 to encourage policy updates
            lr_schedule=[
                [0, 1e-3],     # Start with 1e-3
                [10000, 5e-4], # Decay to 5e-4 at iteration 10000
                [20000, 1e-4], # Decay to 1e-4 at iteration 20000
            ],
            clip_param=0.2,
            entropy_coeff=0.01,  # Increased from 0.001 to encourage exploration
            entropy_coeff_schedule=[
                [0, 0.01],      
                [5000, 0.007],  
                [10000, 0.005], 
                [20000, 0.001], # Decay to exploit more later
            ],
            vf_clip_param=10.0,
            vf_loss_coeff=1.0,
            use_gae=True, # Generalized Advantage Estimation reduces variance in advantage estimates
            lambda_=0.95, # gea_lambda
            #grad_clip=0.5,
            # normalize_rewards=True,
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

            # Optionally modify hyperparameters after restoring
            # For example, adjust learning rate or entropy coefficient
            # trainer.get_policy().config['lr'] = 1e-5  # New learning rate
            # trainer.get_policy().config['entropy_coeff'] = 0.0005  # New entropy coefficient

        except Exception as e:
            print(f"Failed to restore from checkpoint: {e}")
            sys.exit(1)

    print(f"Starting training for {iterations} iterations with {num_players} players.")

    # Initialize variables to track the best model
    best_reward_mean = float('-inf')  # Initialize to negative infinity
    best_checkpoint_path = None

    # Open a log file to record the best models
    log_file = open('best_model_log.txt', 'w')

    # Training loop
    for i in range(1, iterations + 1):
        result = trainer.train()

        print(f"Iteration: {i}")
        print(f"  Policy Loss: {result['info']['learner']['shared_policy']['learner_stats']['policy_loss']}")
        print(f"  Value Function Loss: {result['info']['learner']['shared_policy']['learner_stats']['vf_loss']}")
        print(f"  Entropy: {result['info']['learner']['shared_policy']['learner_stats']['entropy']}")

         # Access metrics from 'env_runners'
        env_runner_metrics = result.get('env_runners', {})
        episode_reward_mean = env_runner_metrics.get('episode_reward_mean', None)
        episode_len_mean = env_runner_metrics.get('episode_len_mean', None)
        policy_reward_mean = env_runner_metrics.get('policy_reward_mean', {}).get('shared_policy', None)

        # Print a formatted summary of metrics
        print(f"Iteration: {i}")
        if episode_reward_mean is not None:
            print(f"  Episode Reward Mean: {episode_reward_mean:.2f}")
        if episode_len_mean is not None:
            print(f"  Episode Length Mean: {episode_len_mean:.2f}")
        if policy_reward_mean is not None:
            print(f"  Policy Reward Mean (shared_policy): {policy_reward_mean:.2f}")

        # Check if current reward mean is better than the best so far
        if episode_reward_mean is not None and episode_reward_mean > best_reward_mean:
            best_reward_mean = episode_reward_mean
            # Save checkpoint
            save_result = trainer.save()
            #checkpoint_path = save_result['checkpoint'].to_directory(save_result['checkpoint'].uri)
            checkpoint_path = save_result.checkpoint.path
            best_checkpoint_path = checkpoint_path
            print(f"  New best model found at iteration {i}, checkpoint saved at {checkpoint_path}")
            # Log the path and score context
            log_file.write(f"Iteration {i}, Best Reward Mean: {best_reward_mean:.2f}, Best Policy Reward: {policy_reward_mean:.2f}, Checkpoint Path: {checkpoint_path}\n")
            log_file.flush()  # Ensure it's written to disk

        # Save the model at specified intervals (optional)
        # if i % 500 == 0:
        #     save_result = trainer.save()
        #     checkpoint_path = save_result['checkpoint'].to_directory(save_result['checkpoint'].uri)
        #     print(f"  Checkpoint saved at {checkpoint_path}")

        print("-" * 50)  # Separator for readability

    # Close the log file
    log_file.close()

    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
