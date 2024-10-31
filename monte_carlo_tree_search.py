import numpy as np
from joblib import Parallel, delayed
from copy import deepcopy
from Simple_Simulator import SimpleTicTacToe


def ucb1(node, c=np.sqrt(2)):  # c is the exploration constant (sqrt(2) is common)
    if node.visits == 0:
        return float('inf')  # Always explore unvisited nodes
    exploitation = node.total_reward / node.visits
    exploration = c * np.sqrt(np.log(node.parent.visits) / node.visits)
    return exploitation + exploration


class SearchTreeNode:
    def __init__(self, state, parent=None, sim_class=SimpleTicTacToe, selection_func=None):
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
        self.class_type = sim_class
        if state.turns:
            self.current_player = state.turns[-1]
        # self.current_player = 1
        self.selection_func = selection_func

    # Merging two trees (self and other) into a new tree
    def merge_trees(self, other):
        """
        Merges the current tree with another tree (other) and returns the merged tree.
        It assumes both trees are of the same game state at the root.
        """
        # if self.state != other.state:
        #    raise ValueError("Cannot merge trees with different root states.")

        # Create a new node for the merged tree
        merged_node = SearchTreeNode(self.state, self.parent, sim_class=self.class_type)

        # Sum up visits and rewards
        merged_node.visits = self.visits + other.visits
        merged_node.total_reward = self.total_reward + other.total_reward

        # Merge children recursively
        all_children_states = sorted(set(self.children.keys()).union(other.children.keys()))
        for child_state in all_children_states:
            if child_state in self.children and child_state in other.children:
                # Both trees have this child, merge them recursively
                merged_node.children[child_state] = self.children[child_state].merge_trees(
                    other.children[child_state])
            elif child_state in self.children:
                # Only the current tree has this child
                merged_node.children[child_state] = self.children[child_state]
            else:
                # Only the other tree has this child
                merged_node.children[child_state] = other.children[child_state]

        return merged_node

    def value(self):
        return self.total_reward / self.visits

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

        return probabilities

    @staticmethod
    def get_state_probabilities_simple(root):
        probabilities = np.zeros((3, 3))

        total_visits = sum([root.children[move].visits for move in root.children.keys()])
        for move in root.children.keys():
            row, col = divmod(move, 3)

            probabilities[row, col] = root.children[move].visits / total_visits

        return root.state.board, probabilities

    def get_best_move(self):
        best_move = max(self.children.keys(), key=lambda move: self.children[move].visits)
        return best_move

    @staticmethod
    def select_best_node_ucb(node, c=np.sqrt(2)):
        if node.state.current_player == 1:
            return max(node.children.values(), key=lambda n: ucb1(n, c))
        else:
            return min(node.children.values(), key=lambda n: ucb1(n, -c))

    @staticmethod
    def expand_node(node, sim_class):
        if node.state.get_legal_moves() or not node.state.is_terminal():
            index = np.random.choice(np.arange(len(node.untried_moves)))
            move = node.untried_moves[index]
            node.untried_moves.remove(move)

            next_state = sim_class(deepcopy(node.state).step_forward(move))

            next_state.current_player *= -1
            child_node = SearchTreeNode(next_state, sim_class=sim_class, parent=node)
            node.children[move] = child_node
            return child_node
        return None

    @staticmethod
    def simulate(state, simulations, sim_class, parallel_sim=None):
        if parallel_sim is None:
            parallel_sim = simulations >= 5000  # noted that it is beneficial to simulate in parallel when
            # the number of simulations is above 5000

        def direct_simulations(state, simulations, sim_class_=sim_class):
            rewards = 0
            game = sim_class_(state)
            for i in range(simulations):
                game.reset(state)
                rewards += game.simulate()
            return rewards

        if not parallel_sim:
            rewards = direct_simulations(state, simulations, sim_class_=sim_class)
        else:
            batch_size = 1000

            def sim(state):
                rewards = direct_simulations(state, batch_size, sim_class_=sim_class)  # sim_class(state).simulate()
                return rewards

            res = Parallel(n_jobs=-2)(delayed(sim)(state) for _ in range(int(np.ceil(simulations / batch_size))))

            rewards = sum(res)
        return rewards / simulations  # Returns 1, -1, or draw

    @staticmethod
    def backpropagate(node, result):
        condition = True
        while condition:
            node.visits += 1
            node.total_reward += result  # result is +1, -1, or 0 for draw
            if node.parent is not None:
                node = node.parent
            else:
                condition = False
        return node

    @staticmethod
    def parallelised_mcts_search(root, sim_class, deepness=20, simulations=100, batch_size=100, c=np.sqrt(2),
                                 selection_func=None):
        def mcts_search_para(root):
            for _ in range(batch_size):
                node = root

                # Selection
                while node.children and not node.untried_moves:
                    if selection_func is None:
                        node = SearchTreeNode.select_best_node_ucb(node, c)
                    else:
                        node = selection_func(node)

                # Expansion
                child = SearchTreeNode.expand_node(node, sim_class)

                if child:
                    node = child

                result = SearchTreeNode.simulate(node.state, simulations, sim_class)

                # Backpropagation
                node = SearchTreeNode.backpropagate(node, result)
            return root

        res = Parallel(n_jobs=-2, verbose=2)(
            delayed(mcts_search_para)(root) for _ in range(int(np.ceil(deepness / batch_size))))
        game = sim_class(root.state)
        new_root = SearchTreeNode(game, sim_class=sim_class)
        for elem in res:
            new_root = new_root.merge_trees(elem)
        return new_root

    @staticmethod
    def mcts_search(root, sim_class, deepness=20, simulations=100, c=np.sqrt(2), selection_func=None):
        for i in range(deepness):
            node = root

            # Selection
            while node.children and not node.untried_moves:
                if selection_func is None:
                    node = SearchTreeNode.select_best_node_ucb(node, c)
                else:
                    node = selection_func(node)
            # Expansion
            child = SearchTreeNode.expand_node(node, sim_class)
            if child:
                node = child

            # Simulation
            result = SearchTreeNode.simulate(node.state, simulations, sim_class)

            # Backpropagation
            node = SearchTreeNode.backpropagate(node, result)
        return root

    @staticmethod
    def make_a_choice(root, my_move=None, deepness=1000, simulations=100, sim_class=SimpleTicTacToe,
                      batch_size=100, c=np.sqrt(2), parallelise=True, selection_func=None):
        if my_move is None:
            pass
        else:
            root = root.children[my_move]
            root.parent = None

        if parallelise:
            root = SearchTreeNode.parallelised_mcts_search(root, sim_class, deepness=deepness, simulations=simulations,
                                                           batch_size=batch_size, c=c,
                                                           selection_func=selection_func)
        else:
            root = SearchTreeNode.mcts_search(root, sim_class, deepness=deepness, simulations=simulations, c=np.sqrt(2),
                                              selection_func=selection_func)

        if root.children:
            best_move = root.get_best_move()
            return best_move, root
        else:
            return 'Terminal', root


class MCTSmodel:
    def __init__(self, deepness=1000, simulations=1, batch_size=100, c=np.sqrt(2),
                 sim_class=SimpleTicTacToe, selection_func=None):
        game = sim_class()
        self.tree = SearchTreeNode(game, parent=None, sim_class=sim_class)
        self.tree.mcts_search(root=self.tree,sim_class=sim_class,deepness=deepness,simulations=simulations,selection_func=selection_func)
        self.deepness = deepness
        self.simulations = simulations
        self.batch_size = batch_size
        self.sim_class = sim_class
        self.c = c
        self.selection_function = selection_func

    def next_move(self, state,in_place=False):


        #
        if not in_place:
            game = self.sim_class(state)
            self.tree = SearchTreeNode(game, sim_class=self.sim_class)

        self.tree = SearchTreeNode.mcts_search(self.tree, self.sim_class, deepness=self.deepness,
                                               simulations=self.simulations,
                                               c=self.c, selection_func=self.selection_function)
        if not state.is_terminal():
            my_move = self.tree.get_best_move()
            state.step_forward(my_move)
            self.tree = self.tree.children[my_move]
            state.current_player *= -1
            return my_move
        else:
            return False

    def make_opponent_move(self,move,game):
        if move in self.tree.children.keys():
            self.tree = self.tree.children[move]
        else:
            self.tree.children[move] = SearchTreeNode(game,parent=self.tree,sim_class=self.sim_class,
                                                      selection_func=self.selection_function)

            self.tree = self.tree.children[move]


    def return_to_root(self):
        while self.tree.parent is not None:
            self.tree = self.tree.parent

    def get_probabilities_for_visited_nodes(self, min_visits=0,node=None, probabilities_dict=None):
        """
        Recursively traverse the tree and collect state probabilities
        for nodes with a minimum number of visits.

        Args:
            node (SearchTreeNode): The current node being visited.
            probabilities_dict (dict): Dictionary to store the probability matrices.

        Returns:
            dict: A dictionary where keys are states and values are probability matrices.
        """
        if probabilities_dict is None:
            probabilities_dict = {}

        if node is None:
            node = self.tree

        # Check if node meets the visit criteria
        if node.visits >= min_visits:
            # Calculate probabilities for the current node
            if len(node.state.moves) > 0:
                probabilities_dict[node.state] = (node.state.moves[-1],SearchTreeNode.get_state_probabilities(node))

        # Traverse the children recursively
        for child_node in node.children.values():
            self.get_probabilities_for_visited_nodes(min_visits,child_node, probabilities_dict)

        return probabilities_dict

    def get_probabilities_for_visited_nodes_list(self,min_visits=0):
        probabilities_dict = self.get_probabilities_for_visited_nodes(min_visits)
        list_ = []
        for node in probabilities_dict.keys():
            last_move_probabilities = probabilities_dict[node]
            list_.append((node.visualise_board(),last_move_probabilities[0],last_move_probabilities[1]))
        return list_



class RandomModel:
    def next_move(self, state):
        if not state.is_terminal():
            index = np.random.choice(np.arange(len(state.legal_moves)))
            my_move = state.legal_moves[index]
            state.step_forward(my_move)
            state.current_player *= -1
            return my_move
        else:
            return False





