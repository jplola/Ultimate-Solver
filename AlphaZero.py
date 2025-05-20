
import random
import torch
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import ast
from MonteCarloSearch import simulate
import numpy as np

from UltimateToeFile import UltimateToe
from UltimateNet import UltimatePolicyValueNet

def print_ultimate_tictactoe(board):
    """
    Prints an Ultimate Tic-Tac-Toe board from a 9x9 numpy array.

    Parameters:
    board (numpy.ndarray): A 9x9 array where:
        - 1 represents player X
        - -1 represents player O
        - 0 represents an empty cell
    """
    # Check input dimensions
    if board.shape != (9, 9):
        raise ValueError(f"Board must be a 9x9 array, got {board.shape}")

    # Color codes
    BLUE = "\033[94m"  # Blue color for X
    RED = "\033[91m"  # Red color for O
    RESET = "\033[0m"  # Reset to default terminal color

    # Symbol mapping with colors
    symbols = {
        0: ' ',
        1: f"{BLUE}X{RESET}",  # X in blue
        -1: f"{RED}O{RESET}"  # O in red
    }

    # Horizontal line separators
    thick_line = "━" * 11
    print(f"┏{thick_line}┓┏{thick_line}┓┏{thick_line}┓")

    # Print the board row by row
    for i in range(9):
        # Start of a major 3x3 block - print a thicker horizontal line
        if i % 3 == 0 and i > 0:
            print(f"┣{thick_line}┫┣{thick_line}┫┣{thick_line}┫")

        # Print each row within the current block
        for block_col in range(3):
            start_col = block_col * 3
            cells = [symbols[board[i, start_col + j]] for j in range(3)]

            # Connect blocks with appropriate vertical separators
            if block_col == 0:
                print(f"┃ {cells[0]} │ {cells[1]} │ {cells[2]} ┃", end="")
            elif block_col == 1:
                print(f"┃ {cells[0]} │ {cells[1]} │ {cells[2]} ┃", end="")
            else:  # block_col == 2
                print(f"┃ {cells[0]} │ {cells[1]} │ {cells[2]} ┃")



    # Print a border at the bottom
    print(f"┗{thick_line}┛┗{thick_line}┛┗{thick_line}┛")




class AlphaZero:
    def __init__(self, state, parent=None, sim_class=None):
        self.state = sim_class(state) if sim_class else state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_reward = 0
        self.untried_moves = [move for move in self.state.get_legal_moves()]
        self.class_type = sim_class
        self.prior = 1e-8  # Default prior probability

    def get_best_move(self):
        if not self.children:
            return None
        best_move = max(self.children.keys(), key=lambda move: self.children[move].visits)
        return best_move

    def get_state_probabilities(self):
        """Returns the board state and move probabilities based on visit counts."""
        if not self.children:
            return self.state.visualise_board(), np.zeros((9, 9))

        total_visits = sum(child.visits for child in self.children.values())
        if total_visits == 0:
            return self.state.visualise_board(), np.zeros((9, 9))

        if self.class_type.__name__ == "SimpleTicTacToe":
            size = 3
            probs = np.zeros((size, size))
            for move, child in self.children.items():
                row, col = divmod(move, size)
                probs[row, col] = child.visits / total_visits
            return self.state.board, probs

        elif self.class_type.__name__ == "UltimateToe":
            probs = np.zeros((9, 9))
            for move, child in self.children.items():
                big_row, big_col = divmod(move[0], 3)
                small_row, small_col = divmod(move[1], 3)
                probs[big_row * 3 + small_row, big_col * 3 + small_col] = child.visits / total_visits
            return self.state.visualise_board(), probs

        return self.state.visualise_board(), np.zeros((9, 9))


def ucb_score(parent, child, c_puct=1.0):
    """UCB score for AlphaZero selection phase."""
    if child.visits == 0:
        q_value = 0
    else:
        q_value = child.total_reward / child.visits

    prior = getattr(child, "prior", 1e-8)
    u_score = c_puct * prior * np.sqrt(parent.visits) / (1 + child.visits)
    return q_value + u_score


def backpropagate(node, result):
    """Backpropagate the result through the tree."""
    while node is not None:
        node.visits += 1
        node.total_reward += result
        node = node.parent
        result = -result


def expand_node(node, sim_class, model):
    """Expand a node using the policy network for priors."""
    if not node.untried_moves:
        return None

    # Prepare input for the neural network
    state_tensor = node.state.to_tensor()

    # Get policy and value from neural network
    model.eval()
    with torch.no_grad():
        value, policy = model(state_tensor)

    policy = torch.softmax(policy.squeeze(), dim=0).detach().cpu().numpy()


    move = random.choice(node.untried_moves)
    node.untried_moves.remove(move)

    # Create new state and switch player
    new_state = sim_class(node.state)
    new_state.step_forward(move)
    new_state.current_player *= -1  # Switch player

    # Create child node
    child_node = AlphaZero(new_state, parent=node, sim_class=sim_class)


    if isinstance(move, tuple) and len(move) == 2:
        # For UltimateToe - complex move structure
        from Utils import from_tuple_to_int
        try:
            idx = from_tuple_to_int(move[0], move[1])
            child_node.prior = policy.flatten()[idx] if idx < len(policy.flatten()) else 1e-6
        except Exception:
            child_node.prior = 1e-6
    else:
        # For SimpleTicTacToe - simple integer move
        try:
            child_node.prior = policy.flatten()[move] if move < len(policy.flatten()) else 1e-6
        except Exception:
            child_node.prior = 1e-6

    # Ensure prior is never too small
    if child_node.prior < 1e-6:
        child_node.prior = 1e-6

    node.children[move] = child_node
    return child_node


def alpha_mcts_search(root, model, depth=100, c_puct=1.0,train=False):
    """
    Performs MCTS search using neural network for guidance.
    """
    for _ in range(depth):
        node = root

        if train and node.children:
                alpha = 0.3
                eps = 0.3

                children_keys = list(node.children)
                children = [node.children[k] for k in children_keys]
                priors = np.array([child.prior for child in children])

                noise = np.random.dirichlet([alpha] * len(children))
                new_priors = (1 - eps) * priors + eps * noise

                for child, new_p in zip(children, new_priors):
                    child.prior = new_p

        # Selection phase - choose path with highest UCB score
        while node.untried_moves == [] and node.children:
            node = max(
                node.children.values(),
                key=lambda n: ucb_score(node, n, c_puct)
            )

        # Expansion phase - expand if not terminal
        if node.untried_moves and not node.state.is_terminal():
            node = expand_node(node, node.class_type, model)

        # Evaluation phase - use neural network instead of simulation
        if node:


            if train:
                value = simulate(node.state, node.class_type, rollouts=50)
            else:
                state_tensor = node.state.to_tensor()
                with torch.no_grad():
                    value, _ = model(state_tensor)
                value = value.item()

            # Backpropagation phase
            backpropagate(node, value)

    return root


def add_symmetries_to_memory(state_tensor, pi_flat, value, memory):
    """
    Adds all 8 symmetries (4 rotations and their flips) to the memory.
    state_tensor: torch.Tensor of shape (C, H, W) or (B, C, H, W)
    pi_flat: numpy array of shape (H*W,)
    """


    # Ensure pi_flat is properly sized
    board_size = int(np.sqrt(pi_flat.size))
    assert board_size * board_size == pi_flat.size, \
        f"pi_flat length {pi_flat.size} is not a perfect square"

    pi_matrix = pi_flat.reshape(board_size, board_size)

    # Always rotate/flip the *last two* dimensions
    spatial_dims = (-2, -1)  # Works for both 3D and 4D tensors
    flip_dim = spatial_dims[1]  # Width axis

    for k in range(4):
        # 1) Rotate
        rotated_state = torch.rot90(state_tensor, k=k, dims=spatial_dims)
        rotated_pi = np.rot90(pi_matrix, k=k)
        flat_pi = rotated_pi.flatten()

        # Add to memory
        memory.append((rotated_state.clone(), flat_pi.copy(), value))

        # 2) Flip horizontally
        flipped_state = torch.flip(rotated_state, dims=[flip_dim])
        flipped_pi = np.fliplr(rotated_pi)
        flat_fp = flipped_pi.flatten()

        # Add to memory
        memory.append((flipped_state.clone(), flat_fp.copy(), value))


def play_self_game(sim_class, model, simulations=100, temp=1.0, max_memory_size=10000):
    """
    Plays a complete self-play game and returns the game trajectory.
    """
    game = sim_class()

    memory = []

    while not game.is_terminal():
        # Search and select best move

        root = AlphaZero(game, sim_class=sim_class)

        root = alpha_mcts_search(root, model, depth=simulations, c_puct=temp,train=True)
        move = root.get_best_move()

        if move is None:
            break

        # Store state and policy before making the move
        board, pi = root.get_state_probabilities()
        state_tensor,_ = game.to_tensor(return_flipped=True)
        pi_flat = pi.flatten()

        # Make the move
        game.step_forward(move)
        game.current_player *= -1  # Switch player

        # Store state and policy (with added symmetries)
        value = root.total_reward / max(root.visits, 1)  # Avoid division by zero

        add_symmetries_to_memory(state_tensor, pi_flat, game.current_player * value, memory)

        # Memory management - trim if too large
        if len(memory) > max_memory_size:
            memory = memory[-max_memory_size:]

        # Print info
        print(f"Current player: {game.current_player}, Estimated value: {value:.3f}")
        print(f'reward for alpha: {(root.total_reward / root.visits):.4f} move:  {root.get_best_move()}')





    winner = game.winner
    print(f"Game finished. Winner: {winner}")

    # Return trajectory with final result labels
    return [(s, p, winner * c) for s, p, c in memory]


def train_network(model, memory, optimizer, epochs=1, batch_size=512, l2_const=1e-5, test_ratio=0.2):
    """
    Trains the neural network on the collected game data.
    """
    # Split memory into train and test sets
    if len(memory) <= batch_size:
        print("Not enough samples to train (need at least batch_size)")
        return

    train_memory, test_memory = train_test_split(memory, test_size=test_ratio, shuffle=True)

    if len(test_memory) < batch_size:
        test_memory = train_memory[:batch_size]  # Fallback if test set is too small

    value_loss_fn = torch.nn.MSELoss()
    policy_loss_fn = torch.nn.CrossEntropyLoss()

    best_model = model
    best_policy_loss = 1e4
    best_value_loss = 1e4

    for epoch in range(epochs):
        # TRAINING
        model.train()
        np.random.shuffle(train_memory)
        total_train_value_loss = 0.0
        total_train_policy_loss = 0.0
        train_batches = max(len(train_memory) // batch_size, 1)

        for i in range(0, len(train_memory), batch_size):
            batch = train_memory[i:i + batch_size]
            if len(batch) < batch_size:
                continue  # Skip incomplete batches

            states, pis, zs = zip(*batch)

            # Handle potential shape issues
            states = torch.stack([s for s in states])
            if states.dim() > 4:
                states = states.squeeze(1)

            target_pis = torch.stack([torch.tensor(p, dtype=torch.float32) for p in pis])
            target_vs = torch.tensor(zs, dtype=torch.float32)

            pred_vs, log_pis = model(states)

            # Ensure consistent shapes
            if pred_vs.dim() > 1:
                pred_vs = pred_vs.squeeze()

            # Handle shape mismatch
            if pred_vs.shape != target_vs.shape:
                if pred_vs.dim() == 0 and target_vs.dim() == 1:
                    pred_vs = pred_vs.expand_as(target_vs)
                elif pred_vs.dim() == 1 and target_vs.dim() == 0:
                    target_vs = target_vs.expand_as(pred_vs)

            loss_v = value_loss_fn(pred_vs, target_vs)
            loss_p = policy_loss_fn(log_pis, target_pis)

            # L2 regularization
            l2_reg = sum(param.pow(2).sum() for param in model.parameters()) * l2_const
            loss = loss_v + loss_p + l2_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_value_loss += loss_v.item()
            total_train_policy_loss += loss_p.item()

        avg_train_value_loss = total_train_value_loss / train_batches
        avg_train_policy_loss = total_train_policy_loss / train_batches

        # TESTING
        model.eval()
        total_test_value_loss = 0.0
        total_test_policy_loss = 0.0
        test_batches = max(len(test_memory) // batch_size, 1)

        with torch.no_grad():
            for i in range(0, len(test_memory), batch_size):
                batch = test_memory[i:i + batch_size]
                if len(batch) < batch_size:
                    continue  # Skip incomplete batches

                states, pis, zs = zip(*batch)

                # Handle potential shape issues
                states = torch.stack([s for s in states])
                if states.dim() > 4:
                    states = states.squeeze(1)

                target_pis = torch.stack([torch.tensor(p, dtype=torch.float32) for p in pis])
                target_vs = torch.tensor(zs, dtype=torch.float32)

                pred_vs, log_pis = model(states)

                # Ensure consistent shapes
                if pred_vs.dim() > 1:
                    pred_vs = pred_vs.squeeze()

                # Handle shape mismatch
                if pred_vs.shape != target_vs.shape:
                    if pred_vs.dim() == 0 and target_vs.dim() == 1:
                        pred_vs = pred_vs.expand_as(target_vs)
                    elif pred_vs.dim() == 1 and target_vs.dim() == 0:
                        target_vs = target_vs.expand_as(pred_vs)

                loss_v = value_loss_fn(pred_vs, target_vs)
                loss_p = policy_loss_fn(log_pis, target_pis)

                total_test_value_loss += loss_v.item()
                total_test_policy_loss += loss_p.item()

        avg_test_value_loss = total_test_value_loss / test_batches
        avg_test_policy_loss = total_test_policy_loss / test_batches

        # PRINT RESULTS
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train  | Value Loss: {avg_train_value_loss:.4f}, Policy Loss: {avg_train_policy_loss:.4f}")
        print(f"  Test   | Value Loss: {avg_test_value_loss:.4f}, Policy Loss: {avg_test_policy_loss:.4f}")
        if (avg_train_value_loss + avg_test_value_loss) / 2 < best_value_loss \
            and (avg_train_policy_loss+avg_test_policy_loss)/2 < best_policy_loss:
            best_model = model
            best_value_loss = (avg_train_value_loss + avg_test_value_loss) / 2
            best_policy_loss = (avg_train_policy_loss+avg_test_policy_loss)/2
        model = best_model


def train_alpha_zero(sim_class, iterations=10, games_per_iter=1, model=None, depth=100,
                     temperature=1.0, max_memory_size=10000, learning_rate=1e-4,epochs=20,batch_size=128):
    """
    Main AlphaZero training loop.
    """
    # Initialize model if not provided
    if model is None:
        from UltimateNet import UltimatePolicyValueNet
        model = UltimatePolicyValueNet()

    optimizer = Adam(model.parameters(), lr=learning_rate)



    for i in range(iterations):
        print(f"Self-play iteration {i + 1}/{iterations}")
        memory = []
        for j in range(games_per_iter):
            print(f"  Game {j + 1}/{games_per_iter}")
            game_data = play_self_game(
                sim_class=sim_class,
                model=model,
                simulations=depth,
                temp=temperature,
                max_memory_size=max_memory_size // games_per_iter
            )
            memory.extend(game_data)

            # Memory management
            if len(memory) > max_memory_size:
                # Keep most recent examples
                memory = memory[-max_memory_size:]

        # Train on collected data
        print(f"Training on {len(memory)} examples")
        train_network(model, memory, optimizer, epochs=epochs,batch_size=batch_size)
        print("Training complete for this iteration\n")

    return model

class AlphaZeroTrainer:
    def __init__(self,epochs=10,learning_rate=1e-4,temperature=1,depth=100,games_per_iter=1,iterations=5,sim_class=UltimateToe):
        self.sim_class = sim_class
        self.iterations = iterations
        self.games_per_iter = games_per_iter
        self.depth = depth
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.temperature = temperature

class AlphaZeroPlayer:
    """
    Player that uses AlphaZero algorithm to select moves.
    """

    def __init__(self, model, sim_class, depth=100, c_puct=1.0):
        self.model = model
        self.sim_class = sim_class
        self.depth = depth
        self.c_puct = c_puct

    def next_move(self, state):
        if state.is_terminal() or not state.get_legal_moves():
            return None

        # Create root node
        root = AlphaZero(state, sim_class=self.sim_class)

        # Run MCTS with neural network guidance
        root = alpha_mcts_search(
            root=root,
            model=self.model,
            depth=self.depth,
            c_puct=self.c_puct
        )
        print(f'reward for alpha: {-(root.total_reward/root.visits):.4f} move:  {root.get_best_move()}')

        # Get and return best move
        return root.get_best_move()
class HumanModel:
    def __init__(self):
        pass

    def next_move(self, state):
        print(f'legal moves = {state.get_legal_moves()}')
        print_ultimate_tictactoe(state.get_board())
        move_input = input("Enter your move as a tuple (e.g., (2, 3)): ")

        try:
            move = ast.literal_eval(move_input)
            if not isinstance(move, tuple) or not all(isinstance(x, int) for x in move):
                raise ValueError
            if move not in state.get_legal_moves():
                print('illegal move, retry')
                raise ValueError

        except (ValueError, SyntaxError):
            print("Invalid input. Please enter a tuple of integers like (2, 3).")
            return self.next_move(state)


        return move

# Create random player for comparison
class RandomPlayer:
    def __init__(self):
        pass
    @staticmethod
    def next_move(state):
        legal_moves = state.get_legal_moves()
        if legal_moves:
            return random.choice(legal_moves)
        return None


def make_models_clash(model_1,model_2,model_1_name='model 1',model_2_name='model 2',number_of_games=100):
    # Play games between AlphaZero and random player
    model_1_wins = 0
    model_2_wins = 0
    draws = 0


    for i in range(number_of_games):
        game = UltimateToe()

        # Alternate who goes first
        if (-1) ** i < 0:
            alpha_is_player = 1
        else:
            alpha_is_player = -1

        while not game.is_terminal():
            if game.current_player == alpha_is_player:
                move = model_1.next_move(game)
            else:
                move = model_2.next_move(game)

            if move is None:
                break

            game.step_forward(move)
            game.current_player *= -1

        # Determine winner
        if game.winner == alpha_is_player:
            model_1_wins += 1
        elif game.winner == -alpha_is_player:

            model_2_wins += 1
        else:
            draws += 1

        print(f"Game {i + 1}: {model_1_name}={model_1_wins}, {model_2_name}={model_2_wins}, Draws={draws}")

    print(f"\nFinal results after {number_of_games} games:")
    print(f"{model_1_name} wins: {model_1_wins} ({model_1_wins / number_of_games * 100:.1f}%)")
    print(f"{model_2_name} wins: {model_2_wins} ({model_2_wins / number_of_games * 100:.1f}%)")
    print(f"Draws: {draws} ({draws / number_of_games * 100:.1f}%)")

# Example usage
if __name__ == '__main__':

    new_model_path = '/Users/pietropezzoli/Desktop/Thesis Pietro Pezzoli/tesi/pythonProject/Ultimate-Solver/checkpoints/SmallAlphaCheckPoints/alphazero_1.pth'
    # Create neural network

    TRAIN_AGAIN = False
    if TRAIN_AGAIN:
        model = UltimatePolicyValueNet(board_side_size=9, channels=3, intermediate_channels=16)

        # Train model (short training for demonstration)
        trained_model = train_alpha_zero(
            sim_class=UltimateToe,
            iterations=5,
            games_per_iter=1,
            model=model,
            depth=162,
            temperature=1.0,
            epochs=30
        )
        trained_model.save_model(
            new_model_path)



