import random
import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class NoThanksEnv(MultiAgentEnv):
    def __init__(self, config=None, num_players=3):
        super().__init__()
        self.num_players = num_players
        self.player_ids = [f"player_{i}" for i in range(num_players)]

        # Define observation and action spaces
        num_opponents = self.num_players - 1
        total_chips = self.num_players * 11  # Each player starts with 11 chips

        # Observation:
        # - Current card (3-35)
        # - Chips on card (0 to total_chips)
        # - Player's remaining chips (0 to total_chips)
        # - Adjusted card value (-35 to total_chips)
        # - Opponents' chips (0 to total_chips per opponent)
        # - Opponents' card counts (0 to 33 per opponent)
        # - Binary vector of player's collected cards (33 elements for cards 3-35)
        # - Number of cards remaining in the deck (0 to 24)
        obs_low = np.array(
            [3, 0, 0, -35] + [0] * num_opponents + [0] * num_opponents + [0] * 33 + [0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [35, total_chips, total_chips, total_chips]  # Adjusted upper bounds
            + [total_chips] * num_opponents  # Opponents' chips upper bound
            + [33] * num_opponents  # Opponents' card counts upper bound
            + [1] * 33
            + [24],  # Maximum number of cards remaining
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
        # Check if the card extends any existing sequence
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
    
    def _get_agent_observation(self, agent):
        """Helper method to get the observation for a specific agent."""
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

        # Number of cards remaining in the deck
        cards_remaining = len(self.deck)

        # Concatenate all observation components
        observation = np.concatenate(
            [
                np.array([current_card], dtype=np.float32),
                np.array([chips_on_card], dtype=np.float32),
                np.array([player_chips], dtype=np.float32),
                np.array([adjusted_card_value], dtype=np.float32),
                np.array(opponent_chips, dtype=np.float32),
                np.array(opponent_card_counts, dtype=np.float32),
                card_binary,
                np.array([cards_remaining], dtype=np.float32),
            ]
        )
        return observation
    
    def get_agent_score(self, agent):
        """Gets the score for the current agent"""
        agent_score = self.calculate_agent_score(agent)
        return agent_score

    def _get_obs(self):
        """Gets the observations for all agents."""
        obs = {}
        for agent in self.agents:
            obs[agent] = self._get_agent_observation(agent)
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
                f"  {agent}: Chips = {self.player_chips[agent]}, Cards = {self.player_cards[agent]}, Score = {self.get_agent_score(agent)}"
            )
        print("--------------------------\n")

    def close(self):
        pass

def calculate_total_points(cards, chips):
    """
    Calculates the total points for a player, considering sequences and chips.
    """
    if not cards:
        return 0 - chips  # Only chips are considered if no cards

    total = 0
    sequences = []
    for card in sorted(cards):
        if sequences and card == sequences[-1][-1] + 1:
            sequences[-1].append(card)
        else:
            sequences.append([card])
    for seq in sequences:
        total += seq[0]  # Only add the lowest card in each sequence
    return total - chips

def calculate_total_points(cards, chips):
    """
    Calculates the total points for a player, considering sequences and chips.
    """
    if not cards:
        return 0 - chips  # Only chips are considered if no cards

    total = 0
    sequences = []
    for card in sorted(cards):
        if sequences and card == sequences[-1][-1] + 1:
            sequences[-1].append(card)
        else:
            sequences.append([card])
    for seq in sequences:
        total += seq[0]  # Only add the lowest card in each sequence
    return total - chips

def _get_agent_observation(self, agent):
    """Helper method to get the observation for a specific agent."""
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

    # Number of cards remaining in the deck
    cards_remaining = len(self.deck)

    # Concatenate all observation components
    observation = np.concatenate(
        [
            np.array([current_card], dtype=np.float32),
            np.array([chips_on_card], dtype=np.float32),
            np.array([player_chips], dtype=np.float32),
            np.array([adjusted_card_value], dtype=np.float32),
            np.array(opponent_chips, dtype=np.float32),
            np.array(opponent_card_counts, dtype=np.float32),
            card_binary,
            np.array([cards_remaining], dtype=np.float32),
        ]
    )
    return observation

def get_agent_score(self, agent):
    """Gets the score for the current agent"""
    agent_score = self.calculate_agent_score(agent)
    return agent_score
def _get_obs(self):
    """Gets the observations for all agents."""
    obs = {}
    for agent in self.agents:
        obs[agent] = self._get_agent_observation(agent)
    return obs

def _cards_to_binary(cards):
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
            f"  {agent}: Chips = {self.player_chips[agent]}, Cards = {self.player_cards[agent]}, Score = {self.get_agent_score(agent)}"
        )
    print("--------------------------\n")

def close(self):
    pass

def env_creator(config):
    return NoThanksEnv(num_players=num_players)

def create_ppo_trainer(num_players=3, checkpoint_path=None, policies=None):
    from ray.rllib.algorithms.ppo import PPOConfig
    temp_env = NoThanksEnv(num_players=num_players)

    if policies == None:
      policies = {
            "shared_policy": (
                None,  # Use default policy class (PPO)
                temp_env.observation_space,  # Updated observation space
                temp_env.action_space,  # Action space remains the same
                {
                    "model": {
                        "fcnet_hiddens": [256, 256, 256],  # Network architecture
                        "fcnet_activation": "relu",
                    },
                    "framework": "torch",
                },
            )
        }
      def policy_mapping_fn(agent_id, episode, worker, **kwargs):
          return "shared_policy"
    else:
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

    return trainer
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
