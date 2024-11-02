import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from create_training_data import make_MCTS_combat
class UltimateTTTDataset(Dataset):
    def __init__(self, game_data):
        """
        game_data: list of tuples (board, last_move, next_move_probs)
        board: 9x9 numpy array
        last_move: integer 0-80
        next_move_probs: 9x9 numpy array
        """
        self.games = game_data

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        board, last_move, next_move_probs = self.games[idx]

        # Create contiguous copies of numpy arrays before converting to tensors
        board = np.ascontiguousarray(board)
        next_move_probs = np.ascontiguousarray(next_move_probs)

        # Convert to torch tensors
        board_tensor = torch.FloatTensor(board).reshape(1, 9, 9)  # Add channel dimension

        # Create last move marker
        last_move_tensor = torch.zeros(9, 9)
        row, col = last_move // 9, last_move % 9
        last_move_tensor[row, col] = 1
        last_move_tensor = last_move_tensor.unsqueeze(0)  # Add channel dimension

        # Stack board state and last move as input channels
        input_tensor = torch.cat([board_tensor, last_move_tensor], dim=0)

        return input_tensor, torch.FloatTensor(next_move_probs)
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_bn=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut
        return F.relu(x)

class UltimateTTTCNN(nn.Module):
    def __init__(self):
        super(UltimateTTTCNN, self).__init__()

        # Convolutional layers with ResNet-style residual blocks
        self.res_block1 = ResidualBlock(2, 32)
        self.res_block2 = ResidualBlock(32, 64)
        self.res_block3 = ResidualBlock(64, 128)
        self.res_block4 = ResidualBlock(128, 64)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=1)

        # Fully connected layers
        self.fc1 = nn.Linear(9 * 9, 128)
        self.fc_extra = nn.Linear(128, 64)  # New fully connected layer
        self.fc2 = nn.Linear(64, 81)  # Output layer for 9x9 board

        # Dropout layer
        self.dropout = nn.Dropout(0.5)  # Dropout with probability 0.5

    def forward(self, x):
        # Pass through residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.conv5(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc_extra(x))  # New fully connected layer
        x = self.fc2(x)

        # Reshape and apply softmax
        x = x.view(-1, 9, 9)
        x = F.softmax(x.view(-1, 81), dim=1).view(-1, 9, 9)

        return x

"""
class UltimateTTTCNN(nn.Module):
    def __init__(self):
        super(UltimateTTTCNN, self).__init__()

        # Input: 2 channels (board state and last move marker)
        # First conv layer processes 3x3 sub-boards
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second conv layer to process the entire board
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third conv layer for higher-level features
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Final layers to produce move probabilities
        self.conv4 = nn.Conv2d(128, 64, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Process local 3x3 patterns
        x = F.relu(self.bn1(self.conv1(x)))

        # Process broader patterns
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Generate move probabilities
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)

        # Reshape to 9x9 and apply softmax
        x = x.view(-1, 9, 9)
        x = F.softmax(x.reshape(-1, 81), dim=1).view(-1, 9, 9)

        return x

"""

def train_model(model, game_data, num_epochs=10, batch_size=32, learning_rate=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create dataset and dataloader
    dataset = UltimateTTTDataset(game_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Reshape for CrossEntropyLoss
            loss = criterion(outputs.view(-1, 81), targets.view(-1, 81))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

    return model

def get_move_probabilities(model, current_board, last_move, device=None):
    """
    Get move probabilities from the model for a given board state and last move.

    Args:
        model: Trained UltimateTTTCNN model
        current_board: 9x9 numpy array with -1, 0, 1 values
        last_move: Integer from 0-80 representing the last move position
        device: torch.device to use (if None, will use CUDA if available)

    Returns:
        9x9 numpy array of probabilities for each possible move
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()  # Set to evaluation mode

    with torch.no_grad():
        # Prepare board input
        board_tensor = torch.FloatTensor(
            np.ascontiguousarray(current_board)
        ).reshape(1, 1, 9, 9).to(device)

        # Prepare last move marker
        last_move_tensor = torch.zeros(1, 1, 9, 9).to(device)
        row, col = last_move // 9, last_move % 9
        last_move_tensor[0, 0, row, col] = 1

        # Combine inputs
        input_tensor = torch.cat([board_tensor, last_move_tensor], dim=1)

        # Get model predictions
        move_probabilities = model(input_tensor)

        # Convert to numpy array
        move_probabilities = move_probabilities.cpu().numpy()[0]

        # Zero out probabilities for already filled positions
        move_probabilities[current_board != 0] = 0

        # Renormalize probabilities if any valid moves exist
        if move_probabilities.sum() > 0:
            move_probabilities /= move_probabilities.sum()

    return move_probabilities






def save_model_checkpoint(model, optimizer, epoch, loss, filename):
    """
    Save model checkpoint including model state, optimizer state, epoch, and loss.

    Args:
        model: The UltimateTTTCNN model
        optimizer: The optimizer used in training
        epoch: Current epoch number
        loss: Current loss value
        filename: Path where to save the checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save model state dict separately
    model_state = model.state_dict()
    torch.save(model_state, f"{filename}_model")

    # Save optimizer state dict separately
    if optimizer is not None:
        optimizer_state = optimizer.state_dict()
        torch.save(optimizer_state, f"{filename}_optimizer")

    # Save metadata
    metadata = {
        'epoch': epoch,
        'loss': loss
    }
    torch.save(metadata, f"{filename}_metadata")

    print(f"Model checkpoint saved to {filename}")

def load_model_checkpoint(filename, model=None, optimizer=None):
    """
    Load a saved model checkpoint.

    Args:
        filename: Path to the checkpoint file base name
        model: Optional - The UltimateTTTCNN model to load state into
        optimizer: Optional - The optimizer to load state into

    Returns:
        model: Loaded model (or new model if none provided)
        optimizer: Loaded optimizer (or None if none provided)
        epoch: The epoch number when checkpoint was saved
        loss: The loss value when checkpoint was saved
    """
    # Check if files exist
    model_file = f"{filename}_model"
    optimizer_file = f"{filename}_optimizer"
    metadata_file = f"{filename}_metadata"

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"No model checkpoint found at {model_file}")

    try:
        # Load model state with weights_only=True
        model_state = torch.load(model_file, map_location='cpu', weights_only=True)

        # Create new model if none provided
        if model is None:
            model = UltimateTTTCNN()

        # Load model state
        model.load_state_dict(model_state)

        # Load optimizer state if provided and file exists
        if optimizer is not None and os.path.exists(optimizer_file):
            optimizer_state = torch.load(optimizer_file, map_location='cpu', weights_only=True)
            optimizer.load_state_dict(optimizer_state)

        # Load metadata
        if os.path.exists(metadata_file):
            metadata = torch.load(metadata_file, map_location='cpu', weights_only=True)
            epoch = metadata.get('epoch', 0)
            loss = metadata.get('loss', float('inf'))
        else:
            print("Warning: Metadata file not found. Using default values.")
            epoch = 0
            loss = float('inf')

        return model, optimizer, epoch, loss

    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {str(e)}")



def train_model_with_checkpoints(model, game_data, num_epochs=10, batch_size=32,
                               learning_rate=0.01, checkpoint_dir='checkpoints',
                               checkpoint_frequency=1, resume_from=None):
    """
    Train model with periodic checkpointing and option to resume training.

    Args:
        model: The UltimateTTTCNN model
        game_data: Training data
        num_epochs: Number of epochs to train
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        checkpoint_dir: Directory to save checkpoints
        checkpoint_frequency: Save checkpoint every N epochs
        resume_from: Optional checkpoint file to resume training from

    Returns:
        model: Trained model
        best_loss: Best loss achieved during training
    """
    if game_data == []:
        print('Gamedata Not Available')
        return -1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load checkpoint if resuming training
    start_epoch = 0
    best_loss = float('inf')
    if resume_from:
        try:
            model, optimizer, start_epoch, best_loss = load_model_checkpoint(
                resume_from, model, optimizer
            )
            print(f"Resuming training from epoch {start_epoch+1}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {str(e)}")
            print("Starting training from beginning...")

    # Create dataset and dataloader
    dataset = UltimateTTTDataset(game_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs.view(-1, 81), targets.view(-1, 81))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

        # Save checkpoint if needed
        if (epoch + 1) % checkpoint_frequency == 0:
            checkpoint_base = os.path.join(
                checkpoint_dir,
                f'model_checkpoint_epoch_{epoch+1}'
            )
            save_model_checkpoint(model, optimizer, epoch+1, avg_loss, checkpoint_base)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_base = os.path.join(checkpoint_dir, 'best_model')
            save_model_checkpoint(model, optimizer, epoch+1, avg_loss, best_model_base)

    return model, best_loss








class CNNpolicyModel:
    def __init__(self,first_deepness=20,second_deepnees=10,
                 min_visits=10,first_simulations=10,
                 second_simulation=20,num_games=10,
                 epochs=20):
        self.first_deepness = first_deepness
        self.second_deepness = second_deepnees
        self.first_simulations = first_simulations
        self.second_simulation = second_simulation
        self.num_games = num_games
        self.min_visits = min_visits
        self.epochs = epochs
        self.len_training_data = 0

    def load_model(self):
        model=UltimateTTTCNN()
        loaded_model, _, _, _ = load_model_checkpoint('checkpoints/best_model', model)
        self.model = loaded_model
    def resume_training(self,learning_rate=0.01):

        self.load_model()
        # Train model with checkpointing
        game_data = make_MCTS_combat(games_played=self.num_games,
                                     first_deepness=self.first_deepness,
                                     second_deepness=self.second_deepness,
                                     min_visits=self.min_visits)
        trained_model, best_loss = train_model_with_checkpoints(
            self.model,
            game_data,
            num_epochs=20,
            checkpoint_frequency=5,  # Save every 5 epochs
            learning_rate=learning_rate
        )
        self.len_training_data += len(game_data)
        self.model = trained_model
    def reset_and_start_training(self,learning_rate=0.01):
        print("starting creating dataset...")
        model = UltimateTTTCNN()

        game_data = make_MCTS_combat(games_played=self.num_games,
                                     first_deepness=self.first_deepness,
                                     second_deepness=self.second_deepness,
                                     second_sim=self.second_simulation,
                                     first_sim=self.first_simulations,
                                     min_visits=self.min_visits)
        print("starting training...")
        trained_model, best_loss = train_model_with_checkpoints(
            model,
            game_data,
            num_epochs=self.epochs,
            checkpoint_frequency=5,  # Save every 5 epochs
            learning_rate=learning_rate
        )
        self.len_training_data = len(game_data)
        self.model = trained_model

    @staticmethod
    def from_tuple_to_one_to81(move: tuple) -> int:
        big_row, big_col = divmod(move[0], 3)
        small_row, small_col = divmod(move[1], 3)
        array = np.array([[0, 1, 2],
                          [9, 10, 11],
                          [18, 19, 20]])
        relative = array[small_row, small_col]
        perfect = relative + 27 * big_row + 3 * big_col
        return int(perfect)

    @staticmethod
    def from_one_to81_to_tuple(move: int) -> tuple:
        big_row,big_col = divmod(move,9)
        big_row,big_col = big_row//3, big_col//3
        relative = move - (27 * big_row + 3 * big_col)
        small_row,small_col = np.where(np.array([[0, 1, 2],
                          [9, 10, 11],
                          [18, 19, 20]]) == relative)
        small_row, small_col = small_row[0],small_col[0]
        return int(big_col +3 * big_row), int(small_col + 3 * small_row)

    def next_move(self,state):
        if not state.is_terminal():
            last_move = self.from_tuple_to_one_to81(state.moves[-1])
            probabilities = get_move_probabilities(self.model, state.visualise_board(), last_move, device=None)
            # Get the most likely move
            #best_move = np.unravel_index(np.argmax(probabilities), probabilities.shape)
            #print(f"Most likely move: row={best_move[0]}, col={best_move[1]}")

            last = state.moves[-1]
            # Get top 3 moves
            flat_probs = probabilities.flatten()
            top_moves = np.argsort(flat_probs)[::-1]
            legal_chosen = []
            for i,move in enumerate(top_moves):
                move_tuple = self.from_one_to81_to_tuple(move)
                if move_tuple in state.legal_moves:
                    legal_chosen.append(move_tuple)
                    state.step_forward(move_tuple)
                    state.current_player *= -1
                    return move_tuple
                if i > 20:
                    break

            index = np.random.choice(np.arange(len(state.legal_moves)))
            my_move = state.legal_moves[index]
            state.step_forward(my_move)
            state.current_player *= -1
            return my_move





        else:
            return False

from monte_carlo_tree_search import MCTSmodel,RandomModel
from UltimateToeFile import UltimateToe


CNN_score = 0
random_score = 0
sim_class = UltimateToe
total_game_numbers = 100

CNN = CNNpolicyModel(first_deepness=20,second_deepnees=20,num_games=1,min_visits=0)
CNN.reset_and_start_training(learning_rate=0.5)
#CNN.resume_training(learning_rate=0.001)
#CNN.load_model()


random_model = MCTSmodel(deepness=10,sim_class=UltimateToe)

for i in range(total_game_numbers):
    game = sim_class()
    first_to_go = game.current_player
    while not game.is_terminal():
        cur = game.current_player
        if game.current_player == -1:
            move = CNN.next_move(game)
        else:
            move = random_model.next_move(game)



    if game.winner == -1:
        CNN_score += 1
    elif game.winner == 1:
        random_score += 1

    print(
        f'game numba = {i}: CNN_score = {CNN_score}, '
        f'random_score = {random_score}')




