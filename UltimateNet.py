
import numpy as np
from SimpleToeFile import SimpleTicTacToe
from Utils import from_numpy_to_onehot, from_tuple_to_int, from_int_to_tuple, board_legal_moves

import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax(x, axis=None):
    x_max = np.max(x, axis=axis, keepdims=True)  # for numerical stability
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class UltimatePolicyValueNet(nn.Module):
    def __init__(self, board_side_size=3, channels=3, intermediate_channels=256, temperature=20.0):
        super().__init__()
        self.channels = channels
        self.board_side_size = board_side_size
        self.intermediate_channels = intermediate_channels
        self.temperature = temperature  # Temperature parameter for softmax

        self.relu = nn.ReLU()

        # Shared ResNet Backbone
        # 1st Conv Block
        self.conv1 = nn.Conv2d(channels, intermediate_channels, kernel_size=(3,3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)

        # 2nd Conv Block
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=(3,3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)

        # Value Net

        self.conv3 = nn.Conv2d(in_channels=intermediate_channels,out_channels=intermediate_channels,kernel_size=(3,3),stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(intermediate_channels)

        self.conv4 = nn.Conv2d(in_channels=intermediate_channels,out_channels=1,kernel_size=(9,9))
        self.tanh = nn.Tanh()
        self.bn7 = nn.BatchNorm2d(1)
        # Policy Net
        self.conv5 = nn.Conv2d(in_channels=intermediate_channels,out_channels=intermediate_channels,kernel_size=(3,3),stride=1,padding=1)
        self.bn5 = nn.BatchNorm2d(intermediate_channels)
        self.conv6 = nn.Conv2d(in_channels=intermediate_channels,out_channels=1,kernel_size=(3,3),stride=1,padding=1)


    def forward(self, input_tensor):
        input_tensor = input_tensor.to(next(self.parameters()).device)
        board = input_tensor[:, :self.channels, :, :]

        x = self.conv1(board)
        x = self.bn1(x)
        x = self.relu(x)
        identity = x
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)

        val = self.conv3(x)
        val = self.bn3(val)
        val = self.relu(val)
        val = self.conv4(val)
        val = self.bn7(val)
        val = val.squeeze([1,2])

        val = self.tanh(val)

        prob = self.conv5(x)
        prob = self.bn5(prob)
        prob = self.relu(prob)
        prob = self.conv6(prob)
        prob = prob.squeeze(1)
        prob = prob.view((-1, 81))

        return val, prob


    def get_value(self,board_np):
        board_tensor = torch.tensor(board_np,dtype=torch.float32).view((1,2,self.board_side_size,self.board_side_size))
        value, prob = self.forward(board_tensor)
        return value

    def get_move_probability(self, board_np, move, not_legal_np=None):
        board_tensor = from_numpy_to_onehot(board_np)
        value, prob = self.forward(board_tensor)

        numba = (move[0] // 3) * 27 + (move[1] // 3) * 9 + (move[0] % 3) * 3 + (move[1] % 3)

        return prob[0][numba]
    def save_model(self,save_path):


        torch.save(self.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self,path):
        self.load_state_dict(torch.load(path))
        print('MODEL LOADED')





class UltimateNetworkModel:
    def __init__(self, model=None, sim_class=SimpleTicTacToe):
        self.model = model
        self.game = sim_class

    def next_move(self,state):
        if self.game==SimpleTicTacToe:
            board = state.board * state.current_player
            model_in = from_numpy_to_onehot(board)
            model_in = model_in.view((1,2,state.board.shape[0],state.board.shape[0]))
            val,probs = self.model(model_in)
            probs = probs.detach().cpu().numpy()[0]
            probs = [p if i in state.legal_moves else 0 for i, p in enumerate(probs)]
            argmax = np.argmax(probs)
            state.step_forward(argmax)
        else:
            board = state.get_board() * state.current_player
            model_in = from_numpy_to_onehot(board)

            if len(state.moves) > 0:
                last = state.moves[-1]
                last_val_on_board = board_legal_moves(board,last)
            else:
                last = (-1,-1)
                last_val_on_board = board_legal_moves(board,last)

            last_val_tensor = torch.tensor(last_val_on_board, dtype=model_in.dtype)

            model_in = torch.cat([model_in, last_val_tensor.unsqueeze(0).unsqueeze(0)], dim=1)


            val, probs = self.model(model_in)
            probs = probs.detach().cpu().numpy()[0]
            probs = softmax(probs)
            probs_ = probs.reshape((9,9))
            move = np.random.choice(np.arange(0,81),p=probs.flatten())
            state.step_forward(from_int_to_tuple(move))






