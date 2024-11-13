
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from create_training_data import make_MCTS_combat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UltimateTicTacToeNet(nn.Module):
    def __init__(self):
        super(UltimateTicTacToeNet, self).__init__()

        # Convolutional branch 1: 3x3 kernel, stride of 3, followed by batch normalization
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=3)
        self.bn1 = nn.BatchNorm2d(32)

        # Convolutional branch 2: 2x2 kernel, stride of 1, followed by batch normalization
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Fully connected layers for both branches combined
        self.fc1 = nn.Linear(32 * 3 * 3 + 32 * 8 * 8 + 10, 128)  # Include the embedding dimension here
        self.bn_fc1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 81)  # Output 81 values (9x9 board)

        # Embedding layer for last move (to handle the integer input)
        self.move_embedding = nn.Embedding(81, 10)  # Embedding dimension of 10 for last move

    def forward(self, board, last_move):
        # Convolutional branch 1 with batch normalization and ReLU activation
        x1 = F.relu(self.bn1(self.conv1(board)))  # Output shape: (batch_size, 32, 3, 3)
        x1 = x1.view(x1.size(0), -1)  # Flatten to (batch_size, 32*3*3)

        # Convolutional branch 2 with batch normalization and ReLU activation
        x2 = F.relu(self.bn2(self.conv2(board)))  # Output shape: (batch_size, 32, 8, 8)
        x2 = x2.view(x2.size(0), -1)  # Flatten to (batch_size, 32*8*8)

        # Get embedding for last move
        move_emb = self.move_embedding(last_move)  # Shape: (batch_size, 10)

        # Concatenate both branches with move embedding
        x = torch.cat((x1, x2, move_emb), dim=1)

        # First fully connected layer with batch normalization and ReLU activation
        x = F.relu(self.bn_fc1(self.fc1(x)))

        # Second fully connected layer to output 81 values
        x = self.fc2(x)  # Shape: (batch_size, 81)

        # Determine occupancy by summing across the channels
        board_occupancy = board.sum(dim=1)  # Shape: (batch_size, 9, 9)

        # Flatten to match the output layer's shape
        board_flattened = board_occupancy.view(board.size(0), -1)  # Shape: (batch_size, 81)

        # Create mask: 1 for empty positions, 0 for occupied positions
        mask = (board_flattened == 0).float()  # Shape: (batch_size, 81)

        # Apply the mask by setting logits for occupied positions to a very low value
        x = x * mask   # Applying mask to logits

        # Apply softmax across the last dimension
        x = F.softmax(x, dim=1)

        return x




class UltimateTicTacToeDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data: A list of tuples, each containing (board_state, last_move, target_probs).
                  board_state: 9x9 grid with values -1, 1, and 0.
                  last_move: Integer from 0 to 80.
                  target_probs: 9x9 grid of probabilities.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board_state, last_move, target_probs = self.data[idx]

        # Convert board state to tensor with two layers
        board_tensor = torch.zeros(2, 9, 9)
        board_tensor[0] = torch.tensor(board_state == 1, dtype=torch.float32)  # Player 1
        board_tensor[1] = torch.tensor(board_state == -1, dtype=torch.float32)  # Player 2

        last_move = torch.tensor(last_move, dtype=torch.long)  # Keep as int for embedding
        target_probs = torch.tensor(target_probs, dtype=torch.float32)  # Target 9x9 probabilities

        return board_tensor, last_move, target_probs


# Save the model
def save_model(model, path='ultimate_tictactoe_model.pth'):
    # Save the model's state_dict
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# Load the model with `weights_only=True`
def load_model(path='ultimate_tictactoe_model.pth'):
    # Initialize an instance of the model
    model = UltimateTicTacToeNet()

    # Load only the weights using `weights_only=True`
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {path}")

    return model











class CnnUltimateToe:
    def __init__(self,path = None):
        if path is None:
            self.model = UltimateTicTacToeNet()

        else:
            self.model = load_model(path)

        # Hyperparameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 20

        # dataset
        self.data = None

    def generate_training_data(self,total_games=10,first_deep=10,second_deep=10):
        data = make_MCTS_combat(games_played=total_games,
                                first_deepness=first_deep,second_deepness=second_deep,
                                first_sim=10,
                                second_sim=10,)

        self.data = data

    def resume_training(self,learning_rate=None,batch_size= None,num_epochs=None):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if batch_size is not None:
            self.batch_size = batch_size
        if learning_rate is not None:
            self.num_epochs = num_epochs

        dataset = UltimateTicTacToeDataset(self.data)  # Replace `data` with your actual dataset variable
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0

            for board, last_move, target_probs in dataloader:
                # Forward pass
                optimizer.zero_grad()
                output = self.model(board, last_move)

                # Reshape target_probs to match output shape (batch_size, 81) and output to match target
                output = output.view(-1, 81)  # Flatten output to (batch_size, 81)
                target_probs = target_probs.view(-1, 81)  # Flatten target to (batch_size, 81)

                # Cross-entropy expects class indices as targets, not probabilities, so we convert it:
                _, target_labels = target_probs.max(dim=1)  # Get the target indices with highest probability

                # Compute loss
                loss = criterion(output, target_labels)
                loss.backward()
                optimizer.step()

                # Accumulate loss
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        save_model(self.model,
                   path='/Users/pietropezzoli/Desktop/Thesis Pietro Pezzoli/tesi/pythonProject/Ultimate-Solver/checkpoints/ultimate_tictactoe_model.pth')

    def predict_move_probabilities(self,board, last_move):
        # Convert the board into a 2-layer tensor (one for each player)
        board_tensor = torch.zeros((1, 2, 9, 9))  # (batch_size=1, channels=2, height=9, width=9)
        board_tensor[0, 0] = torch.tensor(board == 1, dtype=torch.float32)  # Player one layer
        board_tensor[0, 1] = torch.tensor(board == -1, dtype=torch.float32)  # Player two layer

        # Convert last move to a tensor
        last_move_tensor = torch.tensor([last_move], dtype=torch.long)

        # Set model to evaluation mode
        self.model.eval()

        with torch.no_grad():  # Disable gradient calculation for inference
            output = self.model(board_tensor, last_move_tensor)  # Output shape: (1, 9, 9)

        # Extract the probability grid from the output
        probabilities = output.squeeze().numpy()  # Remove batch dimension and convert to numpy

        return probabilities

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
            probabilities = self.predict_move_probabilities(state.visualise_board(), last_move)
            # Get the most likely move
            #best_move = np.unravel_index(np.argmax(probabilities), probabilities.shape)
            #print(f"Most likely move: row={best_move[0]}, col={best_move[1]}")
            a = probabilities.reshape((9,9))
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

                    return move_tuple


            index = np.random.choice(np.arange(len(state.legal_moves)))
            my_move = state.legal_moves[index]
            state.step_forward(my_move)

            return my_move


from monte_carlo_tree_search import RandomModel
from UltimateToeFile import UltimateToe
CNN_score = 0
random_score = 0
sim_class = UltimateToe
total_game_numbers = 5
num_epochs = 25

CNN = CnnUltimateToe()
CNN.generate_training_data(total_games=total_game_numbers,first_deep=36,second_deep=36)
num_epochs = int(np.floor(len(CNN.data)/32))
CNN.resume_training(batch_size=32,num_epochs=num_epochs,learning_rate=0.01)


random_model = RandomModel()

for i in range(1000):
    game = sim_class()
    first_to_go = game.current_player
    while not game.is_terminal():
        cur = game.current_player
        if game.current_player == -1:
            move = CNN.next_move(game)
        else:
            move = random_model.next_move(game)

        game.current_player *=  -1



    if game.winner == -1:
        CNN_score += 1
    elif game.winner == 1:
        random_score += 1

    print(
        f'game numba = {i}: CNN_score = {CNN_score}, '
        f'random_score = {random_score}')


