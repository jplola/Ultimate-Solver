
import numpy as np
import random



class MonteSearch:
    def __init__(self, state, parent=None, sim_class=None):
        self.state = sim_class(state) if sim_class else state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_reward = 0
        self.untried_moves = [move for move in self.state.get_legal_moves()]
        self.class_type = sim_class

    def get_best_move(self):
        if not self.children:
            return None
        best_move = max(self.children.keys(), key=lambda move: self.children[move].visits)
        return best_move

    def get_board_prob_val(self):
        total_visits = sum([self.children[move].visits for move in self.children.keys()])
        probabilities = {}

        for move in sorted(self.children.keys()):
            probabilities[move] = self.children[move].visits / total_visits

        board = self.state.visualise_board()
        value = self.total_reward / total_visits

        return board,probabilities,value

def ucb1(node, c=np.sqrt(2)):
    if node.visits == 0:
        return float('inf')
    exploitation = node.total_reward / node.visits
    exploration = c * np.sqrt(np.log(node.parent.visits) / node.visits)
    return exploitation + exploration


def backpropagate(node, result):
    while node is not None:
        node.visits += 1
        node.total_reward += result
        node = node.parent
        result = -result


def expand_node(node, sim_class):

    # Choose a random untried move
    move = random.choice(node.untried_moves)
    node.untried_moves.remove(move)

    new_state = sim_class(node.state)
    new_state.step_forward(move)
    new_state.current_player *= -1  # Switch player after move

    child_node = MonteSearch(new_state, parent=node, sim_class=sim_class)
    node.children[move] = child_node
    return child_node


def simulate(state, sim_class, rollouts=1):
    """Run a single simulation to the end and return result."""

    parents_player = -state.current_player
    total = 0
    for _ in range(rollouts):
        game = sim_class(state)
        # Play random moves until terminal
        while not game.is_terminal():
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break

            move = random.choice(legal_moves)
            game.step_forward(move)
            game.current_player *= -1

        # Return result from parent player's perspective
        if game.winner == parents_player:
            total += 1.0
        elif game.winner == -parents_player:
            total += -1.0
        else:  # Draw
            total -= 0.
    return total / rollouts


def mcts_search(root, sim_class, depth=1000, c=np.sqrt(2),rollouts=1):
    for _ in range(depth):
        node = root

        # SELECTION
        while node.untried_moves == [] and node.children:
            node = max(node.children.values(), key=lambda n: ucb1(n, c))

        # EXPANSION
        if node.untried_moves and not node.state.is_terminal():
            node = expand_node(node, sim_class)

        # SIMULATION
        if node:
            result = simulate(node.state, sim_class,rollouts)

            # BACKPROPAGATION: Update all nodes with result
            backpropagate(node, result)

    return root


class MonteCarloPlayer:
    def __init__(self, sim_class, iterations=1000, c=np.sqrt(2),rollouts=1):
        self.sim_class = sim_class
        self.iterations = iterations
        self.c = c
        self.rollouts = rollouts

    def next_move(self, state):
        if state.is_terminal() or not state.get_legal_moves():
            return None

        # Create a fresh search tree with the current state
        root = MonteSearch(state, sim_class=self.sim_class)

        # Run MCTS to determine the best move
        root = mcts_search(
            root=root,
            sim_class=self.sim_class,
            depth=self.iterations,
            c=self.c,
            rollouts = self.rollouts
        )
        board,probabilities,value = root.get_board_prob_val()
        print(board)
        print(probabilities)
        print(value)

        # Get and return the best move
        return root.get_best_move()


# Usage example for testing
if __name__ == '__main__':
    from SimpleToeFile import SimpleTicTacToe
    from UltimateToeFile import UltimateToe

    game = SimpleTicTacToe()


    mcts_player = MonteCarloPlayer(
        sim_class=SimpleTicTacToe,
        iterations=100,
        c=1.4
    )



    class RandomPlayer:
        def next_move(self, state):
            legal_moves = state.get_legal_moves()
            if legal_moves:
                return random.choice(legal_moves)
            return None


    random_player = RandomPlayer()


    mcts_wins = 0
    random_wins = 0
    draws = 0
    total_games = 100

    for i in range(total_games):
        game = SimpleTicTacToe()
        current_player = 1


        if i % 2 == 0:
            mcts_is_player = 1
        else:
            mcts_is_player = -1
            game.current_player = -1

        while not game.is_terminal():
            if game.current_player == mcts_is_player:
                move = mcts_player.next_move(game)
            else:
                move = random_player.next_move(game)

            if move is None:
                break

            game.step_forward(move)
            game.current_player *= -1

        # Determine winner
        if game.winner == mcts_is_player:
            mcts_wins += 1
        elif game.winner == -mcts_is_player:
            random_wins += 1
        else:
            draws += 1

        print(f"Game {i + 1}: MCTS={mcts_wins}, Random={random_wins}, Draws={draws}")

    print(f"\nFinal results after {total_games} games:")
    print(f"MCTS wins: {mcts_wins} ({mcts_wins / total_games * 100:.1f}%)")
    print(f"Random wins: {random_wins} ({random_wins / total_games * 100:.1f}%)")
    print(f"Draws: {draws} ({draws / total_games * 100:.1f}%)")