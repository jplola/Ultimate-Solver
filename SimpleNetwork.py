
import torch.optim as optim

from create_training_data import make_MCTS_combat_policy_and_value
from Simple_Simulator import SimpleTicTacToe
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F




class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super(PolicyValueNetwork, self).__init__()

        # Convolutional layer for the 3x3 board input
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

        # Fully connected layers after the convolutional processing
        self.fc1 = nn.Linear(16 * 3 * 3 + 1, 64)  # Including the player indicator (1 additional feature)
        self.fc2 = nn.Linear(64, 64)

        # Policy head for action probabilities
        self.policy_head = nn.Linear(64, 9)

        # Value head for win/loss prediction
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        # Ensure x has a batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add a batch dimension if not present

        # Separate the input into board and player indicator
        board = x[:, :9].view(-1, 1, 3, 3)  # Reshape first 9 elements to (batch_size, 1, 3, 3)
        player_indicator = x[:, 9].view(-1, 1)  # Get the last element as a separate tensor

        # Apply the convolutional layer to the board
        board = F.relu(self.conv1(board))
        board = board.view(-1, 16 * 3 * 3)  # Flatten the conv output

        # Concatenate board features with the player indicator
        x = torch.cat((board, player_indicator), dim=1)

        # Pass through the fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Policy output: probabilities for each action
        policy_output = F.softmax(self.policy_head(x), dim=-1)

        # Value output: probability of winning
        value_output = torch.sigmoid(self.value_head(x))

        return policy_output, value_output


# Instantiate the network
network = PolicyValueNetwork()
optimizer = optim.Adam(network.parameters(), lr=0.001)
training_data = make_MCTS_combat_policy_and_value(games_played=100,first_sim=100,second_sim=100,sim_class=SimpleTicTacToe)
# Combined training loop
num_epochs = 25
for epoch in range(num_epochs):
    total_loss = 0  # Track total loss for the epoch
    batch = training_data[int(epoch * np.floor( len(training_data)/num_epochs)):int((epoch + 1) * np.floor( len(training_data) / num_epochs))]
    for data in batch:
        board_state, target_policy, target_value, current_player = data[0],data[1],data[2],data[3]
        # Convert data to tensors
        # Concatenate board state with current player indicator
        target_value_tensor = torch.tensor(target_value,dtype=torch.float32)
        target_policy_tensor = torch.tensor(target_policy, dtype=torch.float32)

        # Convert board state to a flat tensor and concatenate the current player indicator
        board_state_flat = torch.tensor(board_state,
                                        dtype=torch.float32).flatten()  # Flatten the 3x3 board to 9 elements
        current_player_tensor = torch.tensor([current_player],
                                             dtype=torch.float32)  # Current player as a single-element tensor
        board_tensor = torch.cat((board_state_flat, current_player_tensor))  # Combine into a single tensor of size [10]

        # Forward pass
        policy_output, value_output = network(board_tensor)

        # Compute the policy loss (Cross-Entropy Loss)
        policy_loss = F.cross_entropy(policy_output[0], target_policy_tensor.flatten())

        # Compute the value loss (Mean Squared Error)
        value_loss = F.mse_loss(value_output[0][0], target_value_tensor)

        # Combined loss
        combined_loss = policy_loss + value_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()

        # Accumulate total loss for monitoring
        total_loss += combined_loss.item()

    # Print average loss every 100 epochs
    if epoch % 5 == 0:
        avg_loss = total_loss / len(training_data)
        print(f"Epoch {epoch}, Combined Loss: {avg_loss}")
