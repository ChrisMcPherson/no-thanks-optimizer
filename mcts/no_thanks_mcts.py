import copy
import math
import random


class NoThanksGame:
    """
    Class representing the No Thanks! game.
    """

    def __init__(self, num_players, iterations_list=None, players=None):
        self.deck = [i for i in range(3, 36)]
        random.shuffle(self.deck)
        self.deck = self.deck[:-9]  # Remove 9 random cards
        random.shuffle(self.deck)

        if players is not None:
            self.players = players
        else:
            if iterations_list is None:
                iterations_list = [100] * num_players  # Default iterations for all players
            self.players = [Player(player_id, iterations=iterations_list[player_id]) for player_id in range(num_players)]

        self.current_card = None
        self.chips_on_card = 0
        self.current_player_index = 0
        self.last_action = None

    def start_game(self):
        """Starts the game loop."""
        self.current_card = self.deck.pop() if self.deck else None
        while not self.is_game_over():
            self.play_turn()

    def play_turn(self):
        """Handles a single turn in the game."""
        player = self.players[self.current_player_index]
        action = player.decide_action(self.current_card, self.chips_on_card, self)
        if action == 'take':
            self.process_take_action(player)
        elif action == 'pass':
            self.process_pass_action(player)
        else:
            raise ValueError("Invalid action")

    def process_take_action(self, player):
        """Processes the 'take' action."""
        print(f"\nPlayer {player.id} decides to TAKE the card.")
        player.take_card(self.current_card, self.chips_on_card)
        print(f"Player {player.id} takes card {self.current_card} with {self.chips_on_card} chips.")
        print(f"Player {player.id} now has cards: {sorted(player.cards)} and chips: {player.chips}")
        self.current_card = self.deck.pop() if self.deck else None
        self.chips_on_card = 0
        self.last_action = 'take'

    def process_pass_action(self, player):
        """Processes the 'pass' action."""
        if player.chips <= 0:
            # Player must take the card if they have no chips
            print(f"\nPlayer {player.id} has no chips and must TAKE the card.")
            self.process_take_action(player)
        else:
            player.chips -= 1
            self.chips_on_card += 1
            print(f"\nPlayer {player.id} decides to PASS and places a chip on card {self.current_card}.")
            print(f"Chips on card: {self.chips_on_card}, Player {player.id} now has {player.chips} chips.")
            self.last_action = 'pass'
            self.next_player()

    def next_player(self):
        """Advances to the next player."""
        self.current_player_index = (self.current_player_index + 1) % len(self.players)

    def is_game_over(self):
        """Checks if the game is over."""
        return self.current_card is None and not self.deck

    def copy(self):
        """Creates a deep copy of the game state."""
        return copy.deepcopy(self)

    def apply_action(self, action):
        """Applies an action to the game state."""
        player = self.players[self.current_player_index]
        if action == 'take':
            player.take_card(self.current_card, self.chips_on_card)
            self.current_card = self.deck.pop() if self.deck else None
            self.chips_on_card = 0
            self.last_action = 'take'
        elif action == 'pass':
            if player.chips <= 0:
                player.take_card(self.current_card, self.chips_on_card)
                self.current_card = self.deck.pop() if self.deck else None
                self.chips_on_card = 0
                self.last_action = 'take'
            else:
                player.chips -= 1
                self.chips_on_card += 1
                self.last_action = 'pass'
                self.next_player()
        else:
            raise ValueError("Invalid action")


class Player:
    """
    Class representing an AI player.
    """

    def __init__(self, player_id, iterations=100):
        self.id = player_id
        self.cards = []
        self.chips = 11
        self.iterations = iterations

    def decide_action(self, current_card, chips_on_card, game_state):
        """Decides on an action using MCTS."""
        root_state = game_state.copy()
        root_node = MCTSNode(root_state)
        best_child = mcts(root_node, iterations=self.iterations)
        best_action = best_child.action

        # Print decision statistics
        print(f"\nPlayer {self.id}'s Decision on Card {current_card} with {chips_on_card} chips (Iterations: {self.iterations}):")
        for child in root_node.children:
            action = child.action
            visits = child.visits
            value = child.value
            avg_value = value / visits if visits > 0 else 0
            print(f"  Action: {action}, Visits: {visits}, Avg Value: {avg_value:.2f}")
        print(f"Player {self.id} decides to {best_action.upper()} the card.")
        return best_action

    def take_card(self, card, chips):
        """Updates the player's cards and chips when taking a card."""
        self.cards.append(card)
        self.chips += chips

    def score(self):
        """Calculates the player's score."""
        return calculate_total_points(self.cards, self.chips)


class HumanPlayer(Player):
    """
    Class representing a human player.
    """

    def __init__(self, player_id):
        super().__init__(player_id)
        self.name = f"Human {player_id}"  # You can modify this to ask for the player's name if desired

    def decide_action(self, current_card, chips_on_card, game_state):
        """Prompts the human player for an action."""
        # Display the game state to the user
        print(f"\nIt's your turn, Player {self.id}!")
        print(f"Current Card: {current_card}")
        print(f"Chips on Card: {chips_on_card}")
        print(f"Your Chips: {self.chips}")
        print(f"Your Cards: {sorted(self.cards)}")

        # Available actions
        if self.chips > 0:
            actions = ['take', 'pass']
        else:
            actions = ['take']  # Must take if no chips

        # Prompt the user for action
        while True:
            action = input(f"Do you want to 'take' or 'pass' the card? ").strip().lower()
            if action in actions:
                return action
            else:
                print(f"Invalid action. Please enter one of: {', '.join(actions)}")


class MCTSNode:
    """
    Class representing a node in the MCTS tree.
    """

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0

    def expand(self):
        """Expands the node by generating all possible child nodes."""
        if not self.children:
            possible_actions = ['take']
            current_player = self.state.players[self.state.current_player_index]
            if current_player.chips > 0:
                possible_actions.append('pass')
            for action in possible_actions:
                new_state = self.state.copy()
                new_state.apply_action(action)
                child = MCTSNode(new_state, parent=self, action=action)
                self.children.append(child)

    def is_terminal(self):
        """Checks if the node represents a terminal state."""
        return self.state.is_game_over()

    def best_child(self, c_param=math.sqrt(2)):
        """Selects the best child node using the UCB1 formula."""
        best_score = float('-inf')
        best_children = []
        for child in self.children:
            if child.visits == 0:
                ucb_score = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
                ucb_score = exploitation + exploration
            if ucb_score > best_score:
                best_score = ucb_score
                best_children = [child]
            elif ucb_score == best_score:
                best_children.append(child)
        return random.choice(best_children)


def mcts(root, iterations):
    """
    Performs Monte Carlo Tree Search from the root node.
    """
    root_player_index = root.state.current_player_index
    for _ in range(iterations):
        node = root
        # Selection
        while node.children and not node.is_terminal():
            node = node.best_child()
        # Expansion
        if not node.is_terminal():
            node.expand()
            node = random.choice(node.children)
        # Simulation
        result = simulate(node.state)
        # Backpropagation
        player_score = result[root_player_index]
        reward = -player_score  # Lower scores are better
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
    # Return the child with the highest visit count
    return max(root.children, key=lambda c: c.visits)


def heuristic_decide_action(state, player):
    """
    Decides an action based on the 'total points' heuristic with adaptive probability.
    """
    current_card = state.current_card
    chips_on_card = state.chips_on_card

    # Copy player's current state
    player_cards = player.cards.copy()
    player_chips = player.chips

    # Calculate current total points
    current_total_points = calculate_total_points(player_cards, player_chips)

    # Option 1: Take the card
    # Update cards and chips
    new_player_cards = player_cards + [current_card]
    new_player_chips = player_chips + chips_on_card

    # Calculate new total points if the player takes the card
    total_points_if_take = calculate_total_points(new_player_cards, new_player_chips)

    # Option 2: Pass the card
    # Update chips
    if player_chips > 0:
        new_player_chips_pass = player_chips - 1
    else:
        # If the player has no chips, they must take the card
        return 'take'

    # Estimate the probability that the card will return
    probability_card_returns = estimate_card_return_probability(state, player)

    # Expected total points if the card returns to the player
    expected_chips_on_card = chips_on_card + len(state.players)
    expected_new_player_chips = new_player_chips_pass + expected_chips_on_card
    expected_new_player_cards = player_cards + [current_card]
    total_points_if_card_returns = calculate_total_points(expected_new_player_cards, expected_new_player_chips)

    # Expected total points if the player passes
    total_points_if_pass = (probability_card_returns * total_points_if_card_returns) + \
                           ((1 - probability_card_returns) * current_total_points)

    # Decide based on which action leads to lower expected total points
    if total_points_if_take <= total_points_if_pass:
        return 'take'
    else:
        return 'pass'


def estimate_card_return_probability(state, player):
    """
    Estimates the probability that the current card will return to the player if they pass.
    """
    current_card = state.current_card
    chips_on_card = state.chips_on_card
    num_players = len(state.players)
    opponent_indices = [i for i in range(num_players) if i != player.id]
    opponents = [state.players[i] for i in opponent_indices]
    
    # Base probability
    probability = 1.0
    
    for opponent in opponents:
        # If opponent has no chips, they must take the card
        if opponent.chips <= 0:
            probability *= 0.0
            continue
        
        # Estimate the opponent's likelihood to take the card
        desirability = 0
        
        # Lower card values are more desirable
        if current_card <= 15:
            desirability += 1
        elif current_card >= 30:
            desirability -= 1
        
        # More chips on the card increase desirability
        if chips_on_card >= 3:
            desirability += 1
        
        # Opponent's chip count
        if opponent.chips <= 3:
            # Less able to pass, more likely to take
            desirability += 1
        
        # Convert desirability to probability of taking
        opponent_take_prob = sigmoid(desirability)
        
        # Probability that the opponent passes
        opponent_pass_prob = 1 - opponent_take_prob
        
        # Update the cumulative probability that the card returns
        probability *= opponent_pass_prob
    
    return probability


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + math.exp(-x))


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


def simulate(state):
    """
    Simulates a game from the given state using the 'total points' heuristic policy.
    """
    simulation_state = state.copy()
    while not simulation_state.is_game_over():
        current_player = simulation_state.players[simulation_state.current_player_index]
        action = heuristic_decide_action(simulation_state, current_player)
        simulation_state.apply_action(action)
    scores = [player.score() for player in simulation_state.players]
    return scores


if __name__ == '__main__':
    # Mode selection
    print("Select mode:")
    print("1. Simulation with AI players")
    print("2. Play interactively against AI")
    mode = input("Enter the number of the mode you want to run: ").strip()

    if mode == '1':
        # Run a standard simulation with AI players

        # Number of players
        num_players = 3

        # Iterations per player (can be customized)
        iterations_per_player = [500] * num_players

        # Initialize AI players
        players = [
            Player(
                player_id=i,
                iterations=iterations_per_player[i]
            )
            for i in range(num_players)
        ]

        # Create and start the game
        game = NoThanksGame(num_players=num_players, players=players)
        game.start_game()

        # After the game
        print("\nGame Over!")
        for player in game.players:
            print(f"\nPlayer {player.id} Final Score: {player.score()}")
            print(f"Cards: {sorted(player.cards)}")
            print(f"Chips: {player.chips}")

    elif mode == '2':
        # Interactive play against AI

        # Number of players (including human)
        num_players = 3

        # Iterations per AI player (can be customized)
        iterations_per_player = [1000] * num_players

        # Initialize players (replace one AI with HumanPlayer)
        players = []
        for i in range(num_players):
            if i == 0:
                # The first player is the human
                players.append(HumanPlayer(player_id=i))
            else:
                # The other players are AI
                players.append(
                    Player(
                        player_id=i,
                        iterations=iterations_per_player[i]
                    )
                )

        # Create and start the game
        game = NoThanksGame(num_players=num_players, players=players)
        game.start_game()

        # After the game
        print("\nGame Over!")
        for player in game.players:
            if isinstance(player, HumanPlayer):
                print(f"\nPlayer {player.id} (You) Final Score: {player.score()}")
            else:
                print(f"\nPlayer {player.id} Final Score: {player.score()}")
            print(f"Cards: {sorted(player.cards)}")
            print(f"Chips: {player.chips}")

    else:
        print("Invalid mode selected.")