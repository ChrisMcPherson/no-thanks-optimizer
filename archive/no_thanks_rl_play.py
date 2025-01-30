import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
import ray
import numpy as np
import sys

# ----------------------------
# Define the No Thanks! Env
# ----------------------------
class NoThanksEnv(MultiAgentEnv):
    def __init__(self, config=None, num_players=3):
        super().__init__()
        self.num_players = num_players
        self.player_ids = [f"player_{i}" for i in range(num_players)]

        # Define shared observation and action spaces
        self.observation_space = spaces.Box(low=0, high=35, shape=(36,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0: Take, 1: Pass

        self.reset()

    def reset(self, *, seed=None, options=None):
        self.agents = self.player_ids.copy()
        self.current_agent_idx = 0
        self.current_agent = self.agents[self.current_agent_idx]

        # Initialize deck and remove 9 random cards
        self.deck = list(range(3, 36))
        np.random.shuffle(self.deck)
        self.deck = self.deck[:-9]  # Remove 9 random cards

        # Initialize game state
        self.current_card = self.deck.pop()
        self.chips_on_card = 0

        # Initialize players
        self.player_chips = {agent: 11 for agent in self.agents}
        self.player_cards = {agent: [] for agent in self.agents}

        # Initialize rewards, terminateds, truncateds, infos
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminateds = {agent: False for agent in self.agents}
        self.truncateds = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Return initial observations and infos
        observations = self._get_obs()
        infos = self.infos.copy()

        return observations, infos  # Return both observations and infos

    def step(self, action_dict):
        agent = self.current_agent
        action = action_dict.get(agent, None)

        if action is None:
            raise ValueError(f"Action for agent {agent} is missing in action_dict.")

        # Reset rewards for all agents
        self.rewards = {agent_id: 0.0 for agent_id in self.agents}

        if action == 0:  # Take
            self.player_cards[agent].append(self.current_card)
            self.player_chips[agent] += self.chips_on_card
            self.rewards[agent] += self.chips_on_card  # Reward for gaining chips

            # Draw a new card
            if self.deck:
                self.current_card = self.deck.pop()
                self.chips_on_card = 0
            else:
                # Game over
                for ag in self.agents:
                    self.terminateds[ag] = True

        elif action == 1:  # Pass
            if self.player_chips[agent] <= 0:
                # Must take the card
                self.player_cards[agent].append(self.current_card)
                self.player_chips[agent] += self.chips_on_card
                self.rewards[agent] -= self.current_card  # Penalty for taking card without chips

                # Draw a new card
                if self.deck:
                    self.current_card = self.deck.pop()
                    self.chips_on_card = 0
                else:
                    # Game over
                    for ag in self.agents:
                        self.terminateds[ag] = True
            else:
                # Pass and place a chip
                self.player_chips[agent] -= 1
                self.chips_on_card += 1
                self.rewards[agent] -= 0.1  # Small penalty for passing

        else:
            raise ValueError(f"Invalid action {action} by agent {agent}")

        # Update agent turn
        if not all(self.terminateds.values()):
            self.current_agent_idx = (self.current_agent_idx + 1) % self.num_players
            self.current_agent = self.agents[self.current_agent_idx]

        # Prepare next observations
        observations = self._get_obs()
        rewards = self.rewards.copy()
        infos = self.infos.copy()

        # Prepare terminateds and truncateds dictionaries
        terminateds = self.terminateds.copy()
        truncateds = self.truncateds.copy()

        # Set "__all__" key for terminateds and truncateds
        terminateds["__all__"] = all(self.terminateds.values())
        truncateds["__all__"] = False  # Assuming we don't have truncation in this game

        return observations, rewards, terminateds, truncateds, infos

    def _get_obs(self):
        obs = {}
        for agent in self.agents:
            obs[agent] = np.concatenate([
                np.array([self.current_card - 3], dtype=np.float32),
                np.array([self.chips_on_card], dtype=np.float32),
                np.array([self.player_chips[agent]], dtype=np.float32),
                self._cards_to_binary(self.player_cards[agent])
            ])
        return obs

    def _cards_to_binary(self, cards):
        binary = np.zeros(33, dtype=np.float32)
        for card in cards:
            binary[card - 3] = 1.0
        return binary

    def render(self, mode='human'):
        # Simple text-based rendering
        print("\n--- Current Game State ---")
        print(f"Current Card: {self.current_card}")
        print(f"Chips on Card: {self.chips_on_card}")
        print("Player Status:")
        for agent in self.agents:
            print(f"  {agent}: Chips = {self.player_chips[agent]}, Cards = {self.player_cards[agent]}")
        print("--------------------------\n")

    def close(self):
        pass

# ----------------------------
# Define the Gameplay Script
# ----------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Play No Thanks! against a trained RLlib model.")
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
    args = parser.parse_args()

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register the environment
    register_env(args.env, lambda config: NoThanksEnv(num_players=args.num_players))

    # Create the PPO trainer and load the checkpoint
    trainer = PPO(env=args.env)

    # Load the trained checkpoint
    try:
        trainer.restore(args.checkpoint)
        print(f"Successfully loaded checkpoint from {args.checkpoint}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        sys.exit(1)

    # Initialize the environment
    env = NoThanksEnv(num_players=args.num_players)
    observations, infos = env.reset()

    # Define player IDs
    human_agent = "player_0"
    model_agents = [f"player_{i}" for i in range(1, args.num_players)]

    print("Welcome to No Thanks! You are Player 0.")
    print("Enter your actions as follows:")
    print("  0 - Take the card")
    print("  1 - Pass (place a chip)")
    print("Let's start the game!")

    env.render()

    done = False
    while not done:
        # Determine whose turn it is
        current_agent = env.current_agent

        if current_agent == human_agent:
            # Human's turn
            while True:
                try:
                    action = int(input("Your Action (0: Take, 1: Pass): "))
                    if action in [0, 1]:
                        break
                    else:
                        print("Invalid input. Please enter 0 or 1.")
                except ValueError:
                    print("Invalid input. Please enter a number (0 or 1).")
        else:
            # Model's turn
            # Prepare the observation for the model
            model_obs = observations[current_agent]
            model_action = trainer.compute_single_action(model_obs, policy_id="shared_policy")
            action = model_action
            print(f"{current_agent} chooses to {'Take' if action == 0 else 'Pass'}.")

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
    for agent in env.agents:
        num_cards = len(env.player_cards[agent])
        total_value = sum(env.player_cards[agent])
        chips = env.player_chips[agent]
        score = total_value - chips
        print(f"  {agent}: Cards = {env.player_cards[agent]}, Chips = {chips}, Score = {score}")

    # Determine the winner
    scores = {agent: sum(env.player_cards[agent]) - env.player_chips[agent] for agent in env.agents}
    min_score = min(scores.values())
    winners = [agent for agent, score in scores.items() if score == min_score]

    if len(winners) == 1:
        print(f"\nWinner: {winners[0]} with a score of {min_score}")
    else:
        print(f"\nIt's a tie between: {', '.join(winners)} with a score of {min_score}")

    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    main()