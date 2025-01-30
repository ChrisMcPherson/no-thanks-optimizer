import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
import ray
from pprint import pprint

# Define the No Thanks! environment
class NoThanksEnv(MultiAgentEnv):
    def __init__(self, num_players=3):
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
        pass  # Implement visualization if desired

    def close(self):
        pass


# Initialize Ray
ray.shutdown()  # Shutdown any existing Ray instances
ray.init(ignore_reinit_error=True)

# Register the environment
from ray.tune.registry import register_env

def env_creator(config):
    return NoThanksEnv(num_players=3)

register_env("no_thanks_env", env_creator)

# Create a shared policy for all agents
obs_space = gym.spaces.Box(low=0, high=35, shape=(36,), dtype=np.float32)
act_space = gym.spaces.Discrete(2)

policies = {
    "shared_policy": (
        None,
        obs_space,
        act_space,
        {
            "model": {
                "fcnet_hiddens": [128, 128],
                "fcnet_activation": "relu",
            },
            "framework": "torch",
        },
    )
}

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "shared_policy"


config = (
    PPOConfig()
    .environment(env="no_thanks_env")
    .framework("torch")
    .rollouts(
        num_rollout_workers=1,
        rollout_fragment_length="auto",
    )
    .training(
        train_batch_size=512,
        sgd_minibatch_size=128,
        num_sgd_iter=10,
        lr=5e-4,
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
    )
)

# Create the PPO trainer
trainer = config.build()

# Training loop
for i in range(5001):
    result = trainer.train()

    #result.pop("config")
    #pprint(result)

    env_runner_metrics = result.get('env_runners', {})
    episode_reward_mean = env_runner_metrics.get('episode_reward_mean', None)
    episode_len_mean = env_runner_metrics.get('episode_len_mean', None)
    policy_reward_mean = env_runner_metrics.get('policy_reward_mean', {}).get('shared_policy', None)

    # Access custom metrics if available
    custom_episode_reward = result.get('custom_metrics', {}).get('episode_reward', None)
    custom_episode_length = result.get('custom_metrics', {}).get('episode_length', None)

    # Print a formatted summary of metrics
    print(f"Iteration: {i}")
    if episode_reward_mean is not None:
        print(f"  Episode Reward Mean: {episode_reward_mean:.2f}")
    if episode_len_mean is not None:
        print(f"  Episode Length Mean: {episode_len_mean:.2f}")
    if policy_reward_mean is not None:
        print(f"  Policy Reward Mean (shared_policy): {policy_reward_mean:.2f}")
    if custom_episode_reward is not None:
        print(f"  Custom Episode Reward: {custom_episode_reward:.2f}")
    if custom_episode_length is not None:
        print(f"  Custom Episode Length: {custom_episode_length}")

    # Save the model every 50 iterations
    if i % 50 == 0:
        checkpoint = trainer.save()
        print(f"  Checkpoint saved at {checkpoint}")

    print("-" * 50)  # Separator for readability

# Shutdown Ray
ray.shutdown()