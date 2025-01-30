import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.registry import register_env
import ray
import numpy as np
import sys
import random
import argparse
import json
from no_thanks_env import NoThanksEnv  # Import the environment from your module

# ----------------------------
# Define the Gameplay Script
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Play No Thanks! against trained RLlib models.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the RLlib checkpoint directory.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="no_thanks_env",
        help="Name of the registered environment.",
    )
    parser.add_argument(
        "--num_players",
        type=int,
        default=3,
        help="Total number of players (including the human).",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default=None,
        help="Comma-separated list of policy indices to use (e.g., '1,2'). If not specified, policies will be selected randomly.",
    )
    parser.add_argument(
        "--policy_info_file",
        type=str,
        default=None,
        help="Path to a JSON file containing policy performance metrics (e.g., mean rewards).",
    )
    args = parser.parse_args()

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register the environment
    register_env(args.env, lambda config: NoThanksEnv(num_players=args.num_players))

    # Create the environment instance to access observation and action spaces
    env = NoThanksEnv(num_players=args.num_players)
    observations, infos = env.reset()

    # Get observation and action spaces
    observation_space = env.observation_space
    action_space = env.action_space

    # Define the total number of trained policies (assuming policies are named from policy_0 upwards)
    # Update this to match the number of policies you trained
    total_trained_policies = 3  # Example: Update this if you have more policies

    # Load policy performance metrics if provided
    policy_performance = {}
    if args.policy_info_file:
        try:
            with open(args.policy_info_file, 'r') as f:
                policy_performance = json.load(f)
        except Exception as e:
            print(f"Failed to load policy performance data: {e}")
            sys.exit(1)

    # Parse the policies to use
    if args.policies:
        policy_indices = [int(idx) for idx in args.policies.split(',')]
        if len(policy_indices) != args.num_players - 1:
            print(f"Error: Number of specified policies ({len(policy_indices)}) does not match the number of AI agents ({args.num_players - 1}).")
            sys.exit(1)
    else:
        # Randomly select policies
        policy_indices = random.sample(range(total_trained_policies), args.num_players - 1)

    # Define the policies dictionary including all trained policies
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
        ) for i in range(total_trained_policies)
    }

    # Policy mapping for AI agents
    model_agents = [f"player_{i}" for i in range(1, args.num_players)]
    policy_mapping = {}
    for agent_id, policy_idx in zip(model_agents, policy_indices):
        policy_mapping[agent_id] = f"policy_{policy_idx}"

    # Policy mapping function
    def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
        if agent_id == "player_0":
            return None  # Human player does not use a policy
        else:
            return policy_mapping[agent_id]

    # Configure the trainer
    config = (
        PPOConfig()
        .environment(env=args.env)
        .framework("torch")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
    )

    # Build the trainer
    trainer = config.build()

    # Load the trained checkpoint
    try:
        trainer.restore(args.checkpoint)
        print(f"Successfully loaded checkpoint from {args.checkpoint}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        sys.exit(1)

    # Define player IDs
    human_agent = "player_0"

    print("Welcome to No Thanks! You are Player 0.")
    print("Enter your actions as follows:")
    print("  1 - Pass (place a chip)")
    print("  2 - Take the card")
    print("\nGame Setup:")
    print(f"  Total Players: {args.num_players}")
    print(f"  Human Player: {human_agent}")

    # Display policy assignments and performance metrics
    print("  AI Players and Assigned Policies:")
    for agent_id in model_agents:
        policy_id = policy_mapping[agent_id]
        print(f"    {agent_id} uses {policy_id}", end='')
        if policy_performance:
            performance_info = policy_performance.get(policy_id, {})
            mean_reward = performance_info.get('mean_reward')
            if mean_reward is not None:
                print(f" (Mean Reward: {mean_reward:.2f})")
            else:
                print(" (Mean Reward: N/A)")
        else:
            print()
    print("Let's start the game!\n")

    env.render()

    done = False
    while not done:
        # Determine whose turn it is
        current_agent = env.current_agent

        if current_agent == human_agent:
            # Human's turn
            while True:
                try:
                    action_input = int(input("Your Action (1: Pass, 2: Take): "))
                    if action_input in [1, 2]:
                        # Map user input to action space: 1 (Pass) -> 1, 2 (Take) -> 0
                        action = 1 if action_input == 1 else 0
                        break
                    else:
                        print("Invalid input. Please enter 1 or 2.")
                except ValueError:
                    print("Invalid input. Please enter a number (1 or 2).")
        else:
            # Model's turn
            # Prepare the observation for the model
            model_obs = observations[current_agent]

            # Get the policy ID for the current agent
            policy_id = policy_mapping_fn(current_agent)

            # Compute the action using the trained policy
            model_action = trainer.compute_single_action(
                model_obs, policy_id=policy_id
            )
            action = model_action
            action_desc = 'Pass' if action == 1 else 'Take'
            print(f"{current_agent} ({policy_id}) chooses to {action_desc}.")

        # Prepare the action dictionary
        action_dict = {current_agent: action}

        # Step the environment
        observations, rewards, terminateds, truncateds, infos = env.step(action_dict)

        # Render the environment
        env.render()

        # Check if the game is over
        done = terminateds.get("__all__", False)

    # Game Over: Calculate and display final scores
    print("Game Over! Final Scores:")
    final_scores = env.calculate_final_scores()
    for agent in env.agents:
        print(
            f"  {agent}: Cards = {env.player_cards[agent]}, Chips = {env.player_chips[agent]}, Score = {final_scores[agent]}"
        )

    # Determine the winner
    min_score = min(final_scores.values())
    winners = [agent for agent, score in final_scores.items() if score == min_score]

    if len(winners) == 1:
        print(f"\nWinner: {winners[0]} with a score of {min_score}")
    else:
        print(f"\nIt's a tie between: {', '.join(winners)} with a score of {min_score}")

    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    main()