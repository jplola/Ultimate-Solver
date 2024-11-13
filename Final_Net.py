import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import create_training_data


def from_numpy_to_onehot(board: np.array):
    # Convert the board to a tensor
    board_tensor = torch.tensor(board, dtype=torch.int64)

    # Map the board values to indices:
    # - Map -1 to index 0 (representing -1 pieces)
    # - Map 1 to index 1 (representing 1 pieces)
    mapped_board_1 = torch.where(board_tensor == -1, 1., 0.,)
    mapped_board_2 = torch.where(board_tensor == 1, 1., 0.)
    board_onehot = torch.stack((mapped_board_1, mapped_board_2))

    board_onehot = board_onehot.view((1,2,board.shape[0],board.shape[1]))
    return board_onehot


class PolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.value_1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=2, padding=1, padding_mode='zeros', stride=1)
        self.value_2 = nn.BatchNorm2d(num_features=16)
        self.value_3 = nn.ReLU()
        self.value_4 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=2, padding=0, padding_mode='zeros')
        self.value_5 = nn.BatchNorm2d(num_features=2)
        self.value_6 = nn.Flatten()
        self.value_linear =  nn.Linear(in_features=18, out_features=25)
        self.value_linear_1 = nn.ReLU()
        self.value_linear_2 = nn.Linear(in_features=25, out_features=10)
        self.value_linear_3 = nn.ReLU()
        self.value_linear_4 = nn.Linear(in_features=10, out_features=1)

        self.policy_1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=2, padding=1, padding_mode='zeros', stride=1)
        self.policy_2 = nn.BatchNorm2d(num_features=32)
        self.policy_3 = nn.ReLU()
        self.policy_4 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, padding=0, padding_mode='zeros')
        self.policy_5 = nn.BatchNorm2d(num_features=8)
        self.policy_6 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, padding=0, padding_mode='zeros')
        self.policy_7 = nn.Flatten()

        self.policy_linear_1 = nn.Linear(in_features=16,out_features=64)
        self.policy_linear_2 = nn.ReLU()
        self.policy_linear_3 = nn.Linear(in_features=64,out_features=32)
        self.policy_linear_4 = nn.ReLU()
        self.policy_linear_5 = nn.Linear(in_features=32,out_features=9)




    def forward(self, board):
        value = self.value_1(board)
        value = self.value_2(value)
        value = self.value_3(value)
        value = self.value_4(value)
        value = self.value_5(value)
        value = self.value_6(value)
        value = self.value_linear(value)
        value = self.value_linear_1(value)
        value = self.value_linear_2(value)
        value = self.value_linear_3(value)
        value = self.value_linear_4(value)

        prob = self.policy_1(board)
        prob = self.policy_2(prob)
        prob = self.policy_3(prob)
        prob = self.policy_4(prob)
        prob = self.policy_5(prob)
        prob = self.policy_6(prob)
        prob = self.policy_7(prob)

        prob = self.policy_linear_1(prob)
        prob = self.policy_linear_2(prob)
        prob = self.policy_linear_3(prob)
        prob = self.policy_linear_4(prob)
        prob = self.policy_linear_5(prob)
        # Sum along the channel dimension (dim=1) to get a (batch_size, 3, 3) board occupancy
        board_occupied = board.sum(dim=1)
        board_occupied_flat = board_occupied.view(board.size(0), -1)  # Flatten to (batch_size, 9)

        # Mask the probabilities where the board is occupied
        prob = torch.where(board_occupied_flat != 0, torch.tensor(-10000, device=prob.device), prob)

        # Apply softmax to masked probabilities
        prob = F.softmax(prob, dim=1)

        return value,prob


def save_model_weights(model, path):
    """
    Save the model weights to a specified path.

    Args:
        model (PolicyValueNet): The model instance.
        path (str): The file path where to save the model weights.
    """
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")


def load_model_weights(path):
    """
    Load model weights from a specified path into a new PolicyValueNet instance.

    Args:
        path (str): The file path from where to load the model weights.

    Returns:
        PolicyValueNet: A model instance with loaded weights.
    """
    # Initialize a new model instance
    model = PolicyValueNet()

    # Load the state dictionary
    model.load_state_dict(torch.load(path,weights_only=True))
    print(f"Model weights loaded from {path}")

    return model


def one_hot_encode_tensor(board):
    """
    Convert a 3x3 board with values 1, -1, and 0 to a one-hot encoded PyTorch tensor.

    Args:
        board (np.array): A 3x3 numpy array with values 1, -1, and 0.

    Returns:
        torch.Tensor: A (2, 3, 3) one-hot encoded tensor.
    """
    # Create a one-hot encoding for 1s and -1s
    player1 = (board == 1).astype(np.float32)
    player_minus1 = (board == -1).astype(np.float32)

    # Stack to create a (2, 3, 3) array and convert to tensor
    one_hot = np.stack([player1, player_minus1], axis=0)
    one_hot_tensor = torch.tensor(one_hot, dtype=torch.float32)

    return one_hot_tensor

def prepare_dataset(data_list):
    """
    Prepare a dataset from a list of tuples.

    Args:
        data_list (list of tuples): Each tuple contains:
            - board (3x3 numpy array): The board state
            - probabilities (3x3 numpy array): The action probabilities
            - value (scalar): The target value

    Returns:
        TensorDataset: A dataset that can be used with DataLoader for batching.
    """
    # Unzip the data into separate lists
    boards, probabilities, values = zip(*data_list)

    # Apply one_hot_encode_tensor to each board and stack the results
    boards_tensor = torch.stack([one_hot_encode_tensor(board) for board in boards])

    # Convert probabilities and values to tensors
    probabilities_tensor = torch.tensor([p.flatten() for p in probabilities], dtype=torch.float32)
    values_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

    # Return as a TensorDataset
    dataset = TensorDataset(boards_tensor, probabilities_tensor, values_tensor)
    return dataset

def train_model(model,num_games,lr,batch_size,num_epochs):
    # Example usage
    # Replace `example_data` with your actual data in the form of a list of tuples
    example_data = create_training_data.create_dataset_simple_toe(games_number=num_games)

    # Prepare dataset
    dataset = prepare_dataset(example_data)

    # Now you can use it with a DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss functions
    value_loss_fn = torch.nn.MSELoss()  # For scalar value output
    policy_loss_fn = torch.nn.CrossEntropyLoss()  # For probability output

    # Training loop

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_value_loss = 0.0
        total_policy_loss = 0.0

        for boards, target_probs, target_value in data_loader:
            optimizer.zero_grad()

            # Forward pass
            pred_value, pred_policy = model(boards)

            # Compute value loss
            value_loss = value_loss_fn(pred_value, target_value)

            # Compute policy loss
            policy_loss = policy_loss_fn(pred_policy, target_probs.argmax(dim=1))  # Assuming target_probs are probabilities

            # Total loss
            loss = value_loss + policy_loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Accumulate loss for monitoring
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()

        # Print average losses per epoch
        avg_value_loss = total_value_loss / len(data_loader)
        avg_policy_loss = total_policy_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Value Loss: {avg_value_loss:.4f}, Policy Loss: {avg_policy_loss:.4f}")

    print("Training complete!")

    save_model_weights(model,path='/Users/pietropezzoli/Desktop/Thesis Pietro Pezzoli/tesi/pythonProject/Ultimate-Solver/checkpoints/simple_model_5000.pth')

    return model




class SimpleNetworkModel:
    def __init__(self,path=None):
        if path is not None:
            self.model = load_model_weights(path)
        else:
            self.model = PolicyValueNet()

    def next_move(self,state):
        board = state.board * state.current_player
        model_in = one_hot_encode_tensor(board)
        model_in = model_in.view((1,2,3,3))
        val,probs = self.model(model_in)
        argmax = torch.argmax(probs).item()
        state.step_forward(argmax)


"""
from Simple_Simulator import SimpleTicTacToe
from monte_carlo_tree_search import RandomModel
from MonteCarloSearch import MonteSearchModel
sim_class = SimpleTicTacToe

UpgradedModel = SimpleNetworkModel()

UpgradedModel.model = train_model(UpgradedModel.model,120,lr=0.001,batch_size=64,num_epochs=25)
OldModel = RandomModel()#MonteSearchModel(deepness=20,simulations=10)

upgraded_model_player = -1
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

print(f'perc = {UpgradedModel_score / tot_games}')"""

