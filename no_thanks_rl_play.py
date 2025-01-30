import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.registry import register_env
import ray
import numpy as np
import sys
import random

# ----------------------------
# Define the No Thanks! Environment
# ----------------------------
class NoThanksEnv(MultiAgentEnv):
    def __init__(self, config=None, num_players=3):
        super().__init__()
        self.num_players = num_players
        self.player_ids = [f"player_{i}" for i in range(num_players)]

        # Total chips in the game
        total_chips = self.num_players * 11  # Each player starts with 11 chips
        num_opponents = self.num_players - 1

        # Observation space bounds
        obs_low = np.array(
            [3, 0, 0, -35]  # current_card, chips_on_card, player_chips, adjusted_card_value
            + [0] * num_opponents  # opponent_chips lower bounds
            + [0] * num_opponents  # opponent_card_counts lower bounds
            + [0] * 33,  # card_binary lower bounds
            dtype=np.float32,
        )

        obs_high = np.array(
            [35, total_chips, total_chips, total_chips]  # current_card, chips_on_card, player_chips, adjusted_card_value
            + [total_chips] * num_opponents  # opponent_chips upper bounds
            + [33] * num_opponents  # opponent_card_counts upper bounds
            + [1] * 33,  # card_binary upper bounds
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)  # 0: Take, 1: Pass

        self.reset()

    def reset(self, *, seed=None, options=None):
        self.agents = self.player_ids.copy()
        self.current_agent_idx = 0
        self.current_agent = self.agents[self.current_agent_idx]

        # Initialize deck and remove 9 random cards
        self.deck = list(range(3, 36))
        random.shuffle(self.deck)
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
            # No incremental rewards to align with training environment

            # Draw a new card
            if self.deck:
                self.current_card = self.deck.pop()
                self.chips_on_card = 0
            else:
                # Game over: Calculate and assign final rewards
                final_scores = self.calculate_final_scores()
                for ag in self.agents:
                    self.terminateds[ag] = True
                    # Assign negative final score as reward
                    self.rewards[ag] = -final_scores[ag]
        elif action == 1:  # Pass
            if self.player_chips[agent] <= 0:
                # Must take the card
                self.player_cards[agent].append(self.current_card)
                self.player_chips[agent] += self.chips_on_card
                # No incremental rewards

                # Draw a new card
                if self.deck:
                    self.current_card = self.deck.pop()
                    self.chips_on_card = 0
                else:
                    # Game over: Calculate and assign final rewards
                    final_scores = self.calculate_final_scores()
                    for ag in self.agents:
                        self.terminateds[ag] = True
                        self.rewards[ag] = -final_scores[ag]
            else:
                # Pass and place a chip
                self.player_chips[agent] -= 1
                self.chips_on_card += 1
                # No incremental rewards
        else:
            raise ValueError(f"Invalid action {action} by agent {agent}")

        # Update agent turn if game is not over
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
        truncateds["__all__"] = False  # Assuming no truncation in this game

        return observations, rewards, terminateds, truncateds, infos

    def calculate_final_scores(self):
        final_scores = {}
        for agent in self.agents:
            final_scores[agent] = self.calculate_agent_score(agent)
        return final_scores

    def calculate_agent_score(self, agent):
        # Calculate the agent's final score, accounting for sequences
        sorted_cards = sorted(set(self.player_cards[agent]))
        total = 0
        i = 0
        while i < len(sorted_cards):
            seq_start = sorted_cards[i]
            while (
                i + 1 < len(sorted_cards)
                and sorted_cards[i + 1] == sorted_cards[i] + 1
            ):
                i += 1
            total += seq_start  # Only add the lowest card in the sequence
            i += 1
        total -= self.player_chips[agent]
        return total

    def calculate_adjusted_card_value(self, agent, card):
        # New method calculating direct impact
        cards = set(self.player_cards[agent])
        extends_sequence = False

        if (card - 1) in cards or (card + 1) in cards:
            extends_sequence = True

        # Score impact
        if extends_sequence:
            score_impact = 0
        else:
            score_impact = card  # Card adds to the score

        # Net impact
        adjusted_value = -score_impact + self.chips_on_card
        return adjusted_value

    def _get_obs(self):
        obs = {}
        num_opponents = self.num_players - 1
        total_chips = self.num_players * 11  # Each player starts with 11 chips
        for agent in self.agents:
            # Current card (3-35)
            current_card = self.current_card

            # Chips on card
            chips_on_card = self.chips_on_card

            # Player's remaining chips
            player_chips = self.player_chips[agent]

            # Adjusted card value for the agent
            adjusted_card_value = self.calculate_adjusted_card_value(
                agent, self.current_card
            )

            # Opponents' chips and card counts
            opponent_chips = [
                self.player_chips[opponent]
                for opponent in self.agents
                if opponent != agent
            ]
            opponent_card_counts = [
                len(self.player_cards[opponent])
                for opponent in self.agents
                if opponent != agent
            ]

            # Binary vector for player's collected cards
            card_binary = self._cards_to_binary(self.player_cards[agent])

            # Concatenate all observation components
            obs[agent] = np.concatenate(
                [
                    np.array([current_card], dtype=np.float32),
                    np.array([chips_on_card], dtype=np.float32),
                    np.array([player_chips], dtype=np.float32),
                    np.array([adjusted_card_value], dtype=np.float32),
                    np.array(opponent_chips, dtype=np.float32),
                    np.array(opponent_card_counts, dtype=np.float32),
                    card_binary,
                ]
            )
        return obs

    def _cards_to_binary(self, cards):
        binary = np.zeros(33, dtype=np.float32)  # Cards 3 to 35
        for card in cards:
            if 3 <= card <= 35:
                binary[card - 3] = 1.0
        return binary

    def render(self, mode='human'):
        # Simple text-based rendering
        print("\n--- Current Game State ---")
        print(f"Current Card: {self.current_card}")
        print(f"Chips on Card: {self.chips_on_card}")
        print("Player Status:")
        for agent in self.agents:
            print(
                f"  {agent}: Chips = {self.player_chips[agent]}, Cards = {self.player_cards[agent]}"
            )
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

    # Create the environment instance to access observation and action spaces
    env = NoThanksEnv(num_players=args.num_players)
    observations, infos = env.reset()

    # Reconstruct the same configuration used during training
    num_opponents = args.num_players - 1
    total_chips = args.num_players * 11

    obs_low = np.array(
        [3, 0, 0, -35]  # current_card, chips_on_card, player_chips, adjusted_card_value
        + [0] * num_opponents  # opponent_chips lower bounds
        + [0] * num_opponents  # opponent_card_counts lower bounds
        + [0] * 33,  # card_binary lower bounds
        dtype=np.float32,
    )

    obs_high = np.array(
        [35, total_chips, total_chips, total_chips]  # current_card, chips_on_card, player_chips, adjusted_card_value
        + [total_chips] * num_opponents  # opponent_chips upper bounds
        + [33] * num_opponents  # opponent_card_counts upper bounds
        + [1] * 33,  # card_binary upper bounds
        dtype=np.float32,
    )

    observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    config = (
        PPOConfig()
        .environment(env=args.env)
        .framework("torch")
        .multi_agent(
            policies={
                "shared_policy": (
                    None,
                    observation_space,
                    env.action_space,
                    {
                        "model": {
                            "fcnet_hiddens": [256, 256],
                            "fcnet_activation": "relu",
                        },
                        "framework": "torch",
                    },
                )
            },
            policy_mapping_fn=lambda agent_id, **kwargs: "shared_policy",
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

            # Compute the action using the trained policy
            model_action = trainer.compute_single_action(
                model_obs, policy_id="shared_policy"
            )
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
