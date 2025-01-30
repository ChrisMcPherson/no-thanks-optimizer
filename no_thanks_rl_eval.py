import argparse
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from no_thanks_env import NoThanksEnv 

def main():
    parser = argparse.ArgumentParser(description="Evaluate No Thanks! RL Agent.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the RLlib checkpoint directory.",
    )
    parser.add_argument(
        "--num_players",
        type=int,
        default=3,
        help="Total number of players in the game.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes to run.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during evaluation.",
    )
    args = parser.parse_args()

    # Initialize Ray
    ray.shutdown()  # Ensure any existing Ray instances are shut down
    ray.init(ignore_reinit_error=True)

    # Register the environment
    def env_creator(config):
        return NoThanksEnv(num_players=args.num_players)

    register_env("no_thanks_env", env_creator)

    # Create a shared policy for all agents
    temp_env = NoThanksEnv(num_players=args.num_players)
    policies = {
        "shared_policy": (
            None,  # Use default policy class (PPO)
            temp_env.observation_space,
            temp_env.action_space,
            {
                "model": {
                    "fcnet_hiddens": [256, 256, 256],
                    "fcnet_activation": "relu",
                },
                "framework": "torch",
            },
        )
    }

    # Policy mapping function: All agents use the shared policy
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"

    # Configure RLlib trainer (same as during training)
    config = (
        PPOConfig()
        .environment(env="no_thanks_env")
        .framework("torch")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .rollouts(
            num_rollout_workers=0,  # No workers needed for evaluation
        )
        .resources(
            num_gpus=0
        )
    )

    # Build the trainer and restore from checkpoint
    trainer = config.build()
    try:
        trainer.restore(args.checkpoint)
        print(f"Successfully loaded checkpoint from {args.checkpoint}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        sys.exit(1)

    # Initialize metrics storage
    total_rewards = []
    total_lengths = []
    agent_rewards = defaultdict(list)
    action_counts = defaultdict(int)
    entropy_values = []  # Store average entropy per episode

    # Import torch for softmax
    import torch
    import torch.nn.functional as F

    # Run evaluation episodes
    for episode in range(1, args.num_episodes + 1):
        env = NoThanksEnv(num_players=args.num_players)
        observations, infos = env.reset()
        done = {"__all__": False}
        episode_reward = {agent: 0.0 for agent in env.agents}
        episode_length = 0
        episode_entropies = []  # To store entropies for this episode

        while not done["__all__"]:
            actions = {}
            current_agent = env.current_agent

            # Get the action for the current agent
            obs = observations[current_agent]
            policy_id = policy_mapping_fn(current_agent, None, None)
            action, _, action_info = trainer.compute_single_action(
                obs, policy_id=policy_id, explore=False, full_fetch=True
            )
            actions[current_agent] = action

            # Extract action probabilities
            logits = action_info['action_dist_inputs']
            action_probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
            action_prob = action_probs[action]
            action_counts[action] += 1

            # Compute entropy for the current action probabilities
            entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))  # Add small value to avoid log(0)
            episode_entropies.append(entropy)

            # Step the environment
            observations, rewards, dones, truncateds, infos = env.step(actions)

            # Update episode reward and length
            for agent_id, reward in rewards.items():
                episode_reward[agent_id] += reward
                agent_rewards[agent_id].append(reward)

            episode_length += 1
            done = dones

            # Render if requested
            if args.render:
                env.render()

        # Aggregate rewards across agents
        total_episode_reward = sum(episode_reward.values()) / args.num_players
        total_rewards.append(total_episode_reward)
        total_lengths.append(episode_length)

        # Compute average entropy for the episode
        avg_episode_entropy = np.mean(episode_entropies)
        entropy_values.append(avg_episode_entropy)

        print(f"Episode {episode} - Reward: {total_episode_reward:.2f}, Length: {episode_length}, Entropy: {avg_episode_entropy:.4f}")

    # Calculate and print average metrics
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(total_lengths)
    avg_entropy = np.mean(entropy_values)
    print("\nEvaluation Results:")
    print(f"Average Reward per Episode: {avg_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.2f}")
    print(f"Average Policy Entropy: {avg_entropy:.4f}")

    # Plotting results
    plot_evaluation_results(total_rewards, entropy_values, action_counts)
    print(config.to_dict())


    # Shutdown Ray
    ray.shutdown()

def plot_evaluation_results(total_rewards, entropy_values, action_counts):
    episodes = np.arange(1, len(total_rewards) + 1)

    # Plot total rewards per episode
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(episodes, total_rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards')
    plt.legend()

    # Plot entropy values over episodes
    plt.subplot(1, 3, 2)
    plt.plot(episodes, entropy_values, label='Policy Entropy')
    plt.xlabel('Episode')
    plt.ylabel('Entropy')
    plt.title('Policy Entropy over Episodes')
    plt.legend()

    # Plot action distribution
    plt.subplot(1, 3, 3)
    actions = list(action_counts.keys())
    counts = list(action_counts.values())
    plt.bar(actions, counts, tick_label=['Take (0)', 'Pass (1)'])
    plt.xlabel('Actions')
    plt.ylabel('Counts')
    plt.title('Action Distribution')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
