import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from TrainModel import train_model,save_model_weights,load_model_weights
from CreateTrainingData import from_numpy_to_onehot





class SimplePolicyValueNet(nn.Module):
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

        self.policy_linear_1 = nn.Linear(in_features=16,out_features=45)
        self.policy_linear_2 = nn.ReLU()
        self.policy_linear_3 = nn.Linear(in_features=45,out_features=18)
        self.policy_linear_4 = nn.ReLU()
        self.policy_linear_5 = nn.Linear(in_features=18,out_features=9)




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

    def get_value(self,board_np):
        self.eval()
        board_tensor = from_numpy_to_onehot(board_np)
        value = self.value_1(board_tensor)
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
        return value[0].item()

    def get_move_probability(self,board_np,move,not_legal_np=None):
        self.eval()
        board_tensor = from_numpy_to_onehot(board_np)
        prob = self.policy_1(board_tensor)
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
        board_occupied = board_tensor.sum(dim=1)
        board_occupied_flat = board_occupied.view(board_tensor.size(0), -1)  # Flatten to (batch_size, 9)

        # Mask the probabilities where the board is occupied
        prob = torch.where(board_occupied_flat != 0, torch.tensor(-10000, device=prob.device), prob)

        # Apply softmax to masked probabilities
        prob = F.softmax(prob, dim=1)
        return prob[0][move].item()






class SimpleNetworkModel:
    def __init__(self,path=None):
        if path is not None:
            self.model = load_model_weights(path,SimplePolicyValueNet)
        else:
            self.model = SimplePolicyValueNet()

    def next_move(self,state):
        self.model.eval()
        board = state.board * state.current_player
        model_in = from_numpy_to_onehot(board)
        model_in = model_in.view((1,2,3,3))
        val,probs = self.model(model_in)
        argmax = torch.argmax(probs).item()
        state.step_forward(argmax)

if __name__ == '__main__':

    from SimpleToeFile import SimpleTicTacToe
    from RandomAgent import RandomModel
    from MonteCarloSearch_deprecated import MonteSearchModel
    from CreateTrainingData import create_dataset_simple_toe
    sim_class = SimpleTicTacToe

    UpgradedModel = SimpleNetworkModel()
    tensordataset = create_dataset_simple_toe(games_number=1000,deepness=100)
    UpgradedModel.model = train_model(UpgradedModel.model,tensordataset,lr=0.001,batch_size=32,num_epochs=25)
    OldModel = RandomModel()
    """
    upgraded_model_player = -1
    tot_games = 10000

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

        if i%100 == 0:
            print(f'game numba = {i}: UpgradedModel = {UpgradedModel_score}, OldModel = {OldModel_score}')

    print(f'perc = {UpgradedModel_score / tot_games}')"""

