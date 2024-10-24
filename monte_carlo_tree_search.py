import numpy as np
from joblib import Parallel, delayed
from copy import deepcopy
from Simple_Simulator import SimpleTicTacToe
from UltimateToeFile import UltimateToe


class SearchTreeNode:
    def __init__(self, state, parent=None, sim_class=SimpleTicTacToe):
        self.state = sim_class(state)
        if parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_reward = 0
        self.untried_moves = [move for move in state.get_legal_moves()]
        self.is_terminal = state.is_terminal()
        if state.turns:
            self.current_player = state.turns[-1]
        # self.current_player = 1

    def value(self):
        return self.total_reward / self.visits

    def get_state_probabilities(self):
        all_moves = np.zeros((3, 3))

        possible_moves = self.children.keys()
        total_visits = self.visits
        for elem in possible_moves:
            row, col = divmod(elem, 3)
            all_moves[row, col] = self.children[elem].visits / total_visits
        return all_moves


def ucb1(node, c=np.sqrt(2)):  # c is the exploration constant (sqrt(2) is common)
    if node.visits == 0:
        return float('inf')  # Always explore unvisited nodes
    exploitation = node.total_reward / node.visits
    exploration = c * np.sqrt(np.log(node.parent.visits) / node.visits)
    return exploitation + exploration


def select_best_node(node, c=np.sqrt(2)):
    if node.state.current_player == 1:
        return max(node.children.values(), key=lambda n: ucb1(n, c))
    else:
        return min(node.children.values(), key=lambda n: ucb1(n, -c))


def expand_node(node, sim_class):
    if node.state.get_legal_moves() or not node.state.is_terminal():
        move = node.untried_moves.pop()

        next_state = sim_class(deepcopy(node.state).step_forward(move))

        next_state.current_player *= -1
        child_node = SearchTreeNode(next_state, sim_class=sim_class, parent=node)
        node.children[move] = child_node
        return child_node
    return None


def simulate(state, simulations, sim_class):
    rewards = 0
    game = sim_class(state)
    for i in range(simulations):
        game.reset(state)
        rewards += game.simulate()

    rewards /= simulations

    return rewards  # Returns 1, -1, or draw


def backpropagate__(node, result, bias=0.3):
    for i in range(node.depth):
        node.visits += 1
        # Update the average reward using the incremental averaging formula
        node.total_reward += result - bias  # (result - node.total_reward) / node.visits
        node = node.parent

    node.visits += 1
    # Update the average reward using the incremental averaging formula
    node.total_reward += result - bias  # (result - node.total_reward) / node.visits
    return node

def backpropagate(node, result,bias=0.3):
    while node is not None:
        node.visits += 1
        node.total_reward += result  # result is +1, -1, or 0 for draw
        node = node.parent
    return node


def mcts_search(root, sim_class, deepness=20, simulations=100, bias=0.3, c=np.sqrt(2)):
    for i in range(deepness):
        node = root

        # Selection
        while node.children and not node.untried_moves:
            node = select_best_node(node, c)

        # Simulation
        if node.state.is_terminal():
            print(f'arrived to terrminal {node.state.winner}')
            if node.state.winner != 0.:
                print('eccomi')
        # Expansion
        child = expand_node(node, sim_class)
        if child:
            node = child

        # Simulation
        if node.state.is_terminal():
            print(f'arrived to terrminal {node.state.winner}')
            if node.state.winner != 0.:
                print('eccomi')

        result = simulate(node.state, simulations, sim_class)

        # Backpropagation
        node = backpropagate(node, result, bias)

    # Best Move (Highest visit count)
    best_move = get_best_move(root)
    return best_move, root


def get_best_move(root):
    best_move = max(root.children.keys(), key=lambda move: root.children[move].visits)
    return best_move


def self_play_mcts(mcts_search, game_class, num_games=10, deepness=1000, simulations=100, sim_class=SimpleTicTacToe):
    game = game_class()

    root = SearchTreeNode(game)

    best_move, root = mcts_search(root, deepness=deepness, simulations=simulations, sim_class=sim_class)
    # Loop over a fixed number of self-play games
    for game_num in range(num_games):
        # Initialize the game
        game = game_class()  # This could be Tic-Tac-Toe, Chess, etc.
        current_player = 1  # Player 1 starts

        # Play the game until it reaches a terminal state
        while not game.is_terminal():
            if root.depth == 0:
                best_move = game_num % 9
            # Perform MCTS search from the current game state
            else:
                best_move, root = mcts_search(root, deepness=deepness, simulations=simulations, sim_class=sim_class)

            # Apply the move to the game
            game.step_forward(best_move)

            game.current_player *= -1

            # Update the root for the next turn
            root = root.children[best_move]

            # Switch to the next player
            current_player *= -1

        # Once the game ends, backpropagate the result up the MCTS tree
        result = game.winner  # +1 for Player 1 win, -1 for Player 2 win, 0 for draw
        root = backpropagate(root, result)

    return root





def make_a_choice(root,my_move = None,deepness=1000,simulations=100):
    if my_move is None:
        best_move, root = mcts_search(root,UltimateToe,deepness,simulations)
        return best_move,root
    else:
        root = root.children[(my_move[0],my_move[1])]
        root.parent = None
        best_move, root = mcts_search(root, UltimateToe, deepness, simulations)
        return best_move, root




game = UltimateToe()

root = SearchTreeNode(game, sim_class=UltimateToe)

best_move,root = make_a_choice(root,my_move = None,deepness=100,simulations=1000)
