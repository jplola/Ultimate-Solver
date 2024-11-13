from copy import deepcopy

import numpy as np
from Simple_Simulator import SimpleTicTacToe
from Eval import evaluate_tic_tac_toe
from UltimateToeFile import UltimateToe
from Simple_Simulator import SimpleTicTacToe


class MonteSearch:
    def __init__(self,state,parent=None,sim_class=SimpleTicTacToe):
        self.state = sim_class(state)
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_reward = 0
        self.untried_moves = [move for move in state.get_legal_moves()]
        self.class_type = sim_class
    def get_best_move(self):
        best_move = max(self.children.keys(), key=lambda move: self.children[move].visits)
        return best_move

    def visualise_board(self):
        if self.class_type == SimpleTicTacToe:
            return self.state.board
        elif self.class_type == UltimateToe:
            board = np.zeros((9,9))
            for i in range(9):
                row,col = divmod(i,3)
                board[int(row * 3): int(row*3 + 3),int(col * 3):int(col*3 + 3)] = \
                    self.state.small_boards[i].board

        return board

    @staticmethod
    def get_state_probabilities_simple(root):
        probabilities = np.zeros((3, 3))

        total_visits = sum([root.children[move].visits for move in root.children.keys()])
        for move in root.children.keys():
            row, col = divmod(move, 3)

            probabilities[row, col] = root.children[move].visits / total_visits

        return root.state.board * root.state.current_player, probabilities,abs(root.total_reward/root.visits)

    @staticmethod
    def ucb1(node, c=np.sqrt(2)):  # c is the exploration constant (sqrt(2) is common)
        if node.visits == 1:
            return float('inf')  # Always explore unvisited nodes
        exploitation = node.total_reward / node.visits
        exploration = c * np.sqrt(np.log(node.parent.visits) / node.visits)
        return exploitation + exploration


    def ucb1_with_eval(self,node, current_player, c=np.sqrt(2)):  # c is the exploration constant (sqrt(2) is common)
        if node.visits == 1:
            return float('inf')  # Always explore unvisited nodes
        exploitation = abs(node.total_reward) / node.visits
        exploration = c * np.sqrt(np.log(node.parent.visits) / node.visits)

        res = exploitation + exploration
        return res
    @staticmethod
    def backpropagate_deprecated(node, result):
        condition = True
        while condition:
            node.visits += 1
            if result * node.state.current_player < 0:
                node.total_reward += abs(result)  # result is +1, -1, or 0 for draw
            if node.parent is not None:
                node = node.parent
            else:
                condition = False
        return node

    @staticmethod
    def backpropagate(node, result):
        condition = True
        while condition:
            node.visits += 1
            if result * node.state.current_player < 0.0:
                node.total_reward += abs(result)  # Reward the current player's win
            else:
                node.total_reward -= abs(result)  # Penalize the current player's loss
            if node.parent is not None:
                node = node.parent
            else:
                condition = False
        return node

    @staticmethod
    def expand_node(node, sim_class):
        if node.state.get_legal_moves() or not node.state.is_terminal():
            index = np.random.choice(np.arange(len(node.untried_moves)))
            move = node.untried_moves[index]
            node.untried_moves.remove(move)
            next_state = sim_class(deepcopy(node.state).step_forward(move))
            next_state.current_player *= -1
            child_node = MonteSearch(next_state, sim_class=sim_class, parent=node)
            node.children[move] = child_node
            return child_node
        return None
    @staticmethod
    def simulate(state, simulations, sim_class):
        rewards = 0
        game = sim_class(state)
        for i in range(simulations):
            game.reset(state)
            rewards += game.simulate()
        return rewards / simulations

    @staticmethod
    def select_best_node_ucb(node , c=np.sqrt(2)):
            return max(node.children.values(), key=lambda n: MonteSearch.ucb1(n, c))

    @staticmethod
    def select_best_node_ucb_with_eval(node, current_player, c=np.sqrt(2)):
            return max(node.children.values(), key=lambda n: node.ucb1_with_eval(n, current_player, c))
    @staticmethod
    def mcts_search(root, sim_class, deepness=20, simulations=100, c=np.sqrt(2), selection_func=None,eval = False):
        for i in range(deepness):
            node = root
            # Selection
            while node.children and not node.untried_moves:
                if selection_func is None:
                    node = max(node.children.values(), key=lambda n: MonteSearch.ucb1(n, c))
                else:
                    node = selection_func(node)
                    # Check terminal state before expansion
            if node.state.is_terminal():
                result = node.state.winner  # Get the game result
                if result == -5:
                    result = 0
                node = MonteSearch.backpropagate(node, result)
                continue
            # Expansion
            child = MonteSearch.expand_node(node, sim_class)
            if child:
                node = child
            # Simulation
            result = MonteSearch.simulate(node.state, simulations, sim_class)

            if eval:
                if evaluate_tic_tac_toe(root.state.current_player,node.state.board) \
                        and not  \
                    evaluate_tic_tac_toe(-root.state.current_player,node.state.board):
                    result *= 2

            node = MonteSearch.backpropagate(node, result)
        return root

    @staticmethod
    def get_state_probabilities(root):

        probabilities = np.zeros((9, 9))
        total_visits = 0
        for move in root.children.keys():
            total_visits += root.children[move].visits
        for move in root.children.keys():
            big_row, big_col = divmod(move[0], 3)
            small_row, small_col = divmod(move[1], 3)
            current_subboard = probabilities[big_row * 3: big_row * 3 + 3, big_col * 3: big_col * 3 + 3]

            current_subboard[small_row, small_col] = root.children[move].visits / total_visits

        board = root.visualise_board()
        not_legal_moves = np.ones((81))

        for key in root.children.keys():
            numba = (key[0] // 3) * 27 + (key[1] // 3) * 9 + (key[0] % 3) * 3 + (key[1] % 3)
            not_legal_moves[numba] = 0

        not_legal_moves = not_legal_moves.reshape((9,9))

        return board,not_legal_moves,probabilities,abs(root.total_reward/root.visits)


class MonteSearchModel:
    def __init__(self, deepness=20, simulations=1, c=np.sqrt(2),
                 sim_class=SimpleTicTacToe,eval=False):
        game = sim_class()
        self.tree = MonteSearch(game, parent=None, sim_class=sim_class)
        self.tree.mcts_search(root=self.tree,
                              sim_class=sim_class,
                              deepness=deepness,
                              simulations=simulations,
                              eval=eval)
        self.deepness = deepness
        self.simulations = simulations
        self.sim_class = sim_class
        self.c = c
        self.eval = eval

    def next_move(self, state):

        game = self.sim_class(state)

        self.tree = MonteSearch(game, sim_class=self.sim_class)

        self.tree = MonteSearch.mcts_search(self.tree, self.sim_class,
                                            deepness=self.deepness,
                                            simulations=self.simulations,
                                            eval=self.eval
                                            )

        if not state.is_terminal() and len(self.tree.children) > 0:
            my_move = self.tree.get_best_move()
            state.step_forward(my_move)

            return my_move
        return False



"""
from monte_carlo_tree_search import RandomModel
from UltimateToeFile import UltimateToe
sim_class = SimpleTicTacToe

UpgradedModel = MonteSearchModel(sim_class=sim_class,deepness=100,simulations=10,eval=False)
OldModel = RandomModel()


upgraded_model_player = 1
tot_games = 100

OldModel_score = 0
UpgradedModel_score = 0
for i in range(tot_games):
    game = sim_class()

    if i % 2 == 0:
        game.current_player *= -1
        upgraded_model_player *= -1

    while not game.is_terminal():
        if int(game.current_player) == upgraded_model_player:
            move = UpgradedModel.next_move(game)
        else:
            move = OldModel.next_move(game)

        a = game.board
        # Ensure game.current_player is toggled in the main loop, not inside `next_move`
        game.current_player *= -1

    # Evaluate results
    if game.winner == -upgraded_model_player:
        OldModel_score += 1
    elif game.winner == upgraded_model_player:
        UpgradedModel_score += 1

    print(f'game numba = {i}: UpgradedModel = {UpgradedModel_score}, OldModel = {OldModel_score}')

print(f'perc = {UpgradedModel_score/tot_games}')"""