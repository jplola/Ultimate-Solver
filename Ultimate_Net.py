import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from Final_Net import from_numpy_to_onehot

class UltimatePolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.value_1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=2, padding=1, padding_mode='zeros', stride=1)
        self.value_2 = nn.BatchNorm2d(num_features=16)
        self.value_3 = nn.ReLU()
        self.value_4 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=2, padding=0, padding_mode='zeros')
        self.value_5 = nn.BatchNorm2d(num_features=2)
        self.value_6 = nn.Flatten()
        self.value_linear =  nn.Linear(in_features=162, out_features=256)
        self.value_linear_1 = nn.ReLU()
        self.value_linear_2 = nn.Linear(in_features=256, out_features=162)
        self.value_linear_3 = nn.ReLU()
        self.value_linear_4 = nn.Linear(in_features=162, out_features=1)

        self.policy_1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=2, padding=1, padding_mode='zeros', stride=1)
        self.policy_2 = nn.BatchNorm2d(num_features=32)
        self.policy_3 = nn.ReLU()
        self.policy_4 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, padding=0, padding_mode='zeros')
        self.policy_5 = nn.BatchNorm2d(num_features=8)
        self.policy_6 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, padding=0, padding_mode='zeros')
        self.policy_7 = nn.Flatten()

        self.policy_linear_1 = nn.Linear(in_features=784,out_features=256)
        self.policy_linear_2 = nn.ReLU()
        self.policy_linear_3 = nn.Linear(in_features=256,out_features=162)
        self.policy_linear_4 = nn.ReLU()
        self.policy_linear_5 = nn.Linear(in_features=162,out_features=81)


    def forward(self,input_tensor):
        board = input_tensor[:, :2, :, :]
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

        # Mask the probabilities where the board is occupied
        # Assume input_tensor has shape (batch_size, 3, 9, 9)
        not_legal_moves = input_tensor[:, 2, :, :]  # Extract the third channel (shape: batch_size, 9, 9)

        # Flatten if needed to match the shape of `prob`
        not_legal_moves = not_legal_moves.flatten(start_dim=1)  # Flatten across each sample if needed

        # Apply the mask to `prob` with torch.where
        prob = torch.where(not_legal_moves != 0, torch.tensor(-10000.0, device=prob.device), prob)

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





def train_model(model,tensordataset,lr,batch_size,num_epochs):

    # Now you can use it with a DataLoader
    data_loader = DataLoader(tensordataset, batch_size=batch_size, shuffle=True)

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
            value_loss = value_loss_fn(pred_value.view((pred_value.shape[0],)), target_value)

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

    save_model_weights(model,path='/Users/pietropezzoli/Desktop/Thesis Pietro Pezzoli/tesi/pythonProject/Ultimate-Solver/checkpoints/Ult_model_5.pth')

    return model

from create_training_data import create_dataset_ultimate_toe

board = np.zeros((9,9))

board[0,0] = 1
board[1,0] = -1

board_tensor = from_numpy_to_onehot(board)
not_legal = np.ones((9,9))
not_legal[6:9,6:9] = np.zeros((3,3))

not_legal_tensor = torch.tensor(not_legal,dtype=torch.float32).unsqueeze(0).unsqueeze(0)

input_tensor = torch.cat((board_tensor,not_legal_tensor),dim=1)

model = UltimatePolicyValueNet()

dataset = create_dataset_ultimate_toe(games_number = 1,deepness = 10)


model = train_model(model,dataset,lr=0.001,batch_size=10,num_epochs=2)


