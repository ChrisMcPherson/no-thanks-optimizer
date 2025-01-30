# No Thanks Optimizer

## Overview

This project implements Reinforcement Learning (RL) and Monte Carlo Tree Search (MCTS) algorithms to optimize playing the card game "No Thanks!". The project uses the Ray RLlib library for training the agents.

## Features

-   **Reinforcement Learning (RL):** Uses the Proximal Policy Optimization (PPO) algorithm to train agents to play the game.
-   **Monte Carlo Tree Search (MCTS):** Implements the MCTS algorithm to make decisions.
-   **Multi-agent environment:** The project supports multi-agent environments where multiple agents play against each other.

## Getting Started

To run the RL training:

1. Run `no_thanks_rl.py`
2. Use the `--num_players` and `--iterations` arguments to specify the number of players and iterations.
3. Use `--checkpoint_path` argument to restore from a checkpoint if one is available.

To run the MCTS simulation:

1. Run `mcts/no_thanks_mcts.py`
2. Select the mode you want to use, either "Simulation with AI players" or "Play interactively against AI".
3. You can customize the number of iterations per player in `no_thanks_mcts.py`

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvement, please submit a pull request.

## License

This project is licensed under the MIT License.