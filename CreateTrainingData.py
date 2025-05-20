from copy import deepcopy

import numpy as np
import pandas as pd

from SimpleToeFile import SimpleTicTacToe
from RandomAgent import RandomModel


from UltimateToeFile import UltimateToe
from MonteCarloSearch_deprecated import MonteSearchModel
from torch.utils.data import TensorDataset
import torch

from Utils import from_tuple_to_int, from_numpy_to_onehot, from_int_to_tuple, board_legal_moves


def create_dataset_ultimate_toe(games_number = 100,deepness = 100,model=None):

    sim_class = UltimateToe

    if model is None:
        UpgradedModel = MonteSearchModel(sim_class=sim_class, deepness=deepness, simulations=100, eval=False)
    else:
        UpgradedModel = model
    OldModel = RandomModel()

    input_tensors = torch.empty((1,3,9,9))
    prob_tensors = torch.empty((1,81))
    val_tensors = torch.empty((1,))

    upgraded_model_player = 1
    tot_games = games_number

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
                if not game.is_terminal():
                    board,not_legal,prob,val = UpgradedModel.tree.get_state_probabilities(UpgradedModel.tree)
                    board_tensor = from_numpy_to_onehot(board * game.current_player)
                    not_legal_tensor = torch.tensor(not_legal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    input_tensor = torch.cat((board_tensor,not_legal_tensor),dim=1)
                    prob_tensor = torch.tensor(prob.flatten(), dtype=torch.float32).unsqueeze(0)
                    val_tensor = torch.tensor(val, dtype=torch.float32).unsqueeze(0)

                input_tensors = torch.cat((input_tensors, input_tensor), dim=0)
                prob_tensors = torch.cat((prob_tensors, prob_tensor), dim=0)
                val_tensors = torch.cat((val_tensors, val_tensor), dim=0)
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

    print(f'perc = {UpgradedModel_score / tot_games}')

    dataset = TensorDataset(input_tensors, prob_tensors, val_tensors)

    return dataset

def create_dataset_csv(games_number = 100,deepness=100, model=None,game_class=SimpleTicTacToe):
    sim_class = game_class
    if model is None:
        UpgradedModel = MonteSearchModel(sim_class=sim_class, deepness=deepness, simulations=10, eval=False)
    else:
        UpgradedModel = model

    game = sim_class()
    OldModel = RandomModel()

    upgraded_model_player = 1

    rows_numpy = np.zeros((2,int(len(game.board.flatten())*2+1)))

    for i in range(games_number):
        game = sim_class()

        if i % 2 == 0:
            game.current_player *= -1
            upgraded_model_player *= -1

        while not game.is_terminal():
            if int(game.current_player) == upgraded_model_player:
                move = UpgradedModel.next_move(game)

                board,prob,val = UpgradedModel.tree.get_state_probabilities_simple(UpgradedModel.tree)
                board_adj = board * upgraded_model_player
                val_adj = val * upgraded_model_player

                row = np.concatenate([board_adj.flatten(), prob.flatten(), [val_adj]])

                rows_numpy = np.concatenate([rows_numpy,row.reshape((1,len(row)))],axis=0)

            else:
                move = OldModel.next_move(game)

            a = game.board

            game.current_player *= -1

    columns = [f'board_{i}' for i in range(len(game.board.flatten()))] + \
              [f'policy_{i}' for i in range(len(game.board.flatten()))] + \
              ['value']

    df = pd.DataFrame(rows_numpy[2:,:],columns=columns)
    return df

def create_dataset_simple_toe(games_number = 10,deepness = 100,model = None):

    sim_class = SimpleTicTacToe
    if model is None:
        UpgradedModel = MonteSearchModel(sim_class=sim_class, deepness=deepness, simulations=10, eval=False)
    else:
        UpgradedModel = model
    OldModel = RandomModel()

    input_tensors = torch.empty((1,2,3,3))
    prob_tensors = torch.empty((1,9))
    val_tensors = torch.empty((1,))

    upgraded_model_player = 1
    tot_games = games_number

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
                board,prob,val = UpgradedModel.tree.get_state_probabilities_simple(UpgradedModel.tree)
                board_adj = board * upgraded_model_player
                val_adj = val * upgraded_model_player
                input_tensor = from_numpy_to_onehot(board * upgraded_model_player)
                prob_tensor = torch.tensor(prob.flatten(), dtype=torch.float32).unsqueeze(0)
                val_tensor = torch.tensor(val * upgraded_model_player, dtype=torch.float32).unsqueeze(0)

                input_tensors = torch.cat((input_tensors, input_tensor), dim=0)
                prob_tensors = torch.cat((prob_tensors, prob_tensor), dim=0)
                val_tensors = torch.cat((val_tensors, val_tensor), dim=0)
            else:
                move = OldModel.next_move(game)

            a = game.board

            game.current_player *= -1

        # Evaluate results
        if game.winner == -upgraded_model_player:
            OldModel_score += 1
        elif game.winner == upgraded_model_player:
            UpgradedModel_score += 1

        print(f'game numba = {i}: UpgradedModel = {UpgradedModel_score}, OldModel = {OldModel_score}')

    print(f'perc = {UpgradedModel_score / tot_games}')

    dataset = TensorDataset(input_tensors, prob_tensors, val_tensors)

    return dataset

def generate_ultimate_ttt_symmetries(board):
    """
    Generate all 8 symmetric versions of a 9x9 Ultimate Tic Tac Toe board.

    Parameters:
    board (np.ndarray): A 9x9 numpy array representing the board.

    Returns:
    np.ndarray: A numpy array of shape (8, 9, 9) with all symmetries.
    """
    if board.shape != (9, 9):
        raise ValueError("Input board must be a 9x9 numpy array")

    symmetries = []

    # Identity
    symmetries.append(board.copy())

    # Rotations
    symmetries.append(np.rot90(board, 1))
    symmetries.append(np.rot90(board, 2))
    symmetries.append(np.rot90(board, 3))

    # Reflections
    symmetries.append(np.fliplr(board))  # Horizontal flip
    symmetries.append(np.flipud(board))  # Vertical flip
    symmetries.append(np.transpose(board))  # Main diagonal
    symmetries.append(np.fliplr(np.transpose(board)))  # Anti-diagonal

    return np.stack(symmetries, axis=0)

def from_flat_board_prob_to_tensor(board,prob):
    pass
def parse_dataset(df,out_channels=2):

    col_list = df.columns.to_list()
    board_cols = [col for col in col_list if 'board' in col]

    last_val_np = df['last_move'].to_numpy()
    prob_cols = [col for col in col_list if 'policy' in col]
    value_cols = ['value']
    input_np = df[board_cols].to_numpy()
    prob_np = df[prob_cols].to_numpy()
    val_np = df[value_cols].to_numpy()

    input_tensors = []
    prob_tensors = []
    val_tensors = []
    counter = 0
    for i in range(len(df)):
        flat_board = input_np[i,:]
        flat_prob = prob_np[i,:]
        flat_val = val_np[i]



        flat_last_val = last_val_np[i]

        board_normal = flat_board.reshape((9, 9)).astype(int)

        if flat_last_val < 0:
            counter = 0
        else:
            player = flat_board[int(flat_last_val)]
            counter += 1


            print(f'val {flat_val.item():.4f} last player {player}')
            print(board_normal)
        last_val_on_board = board_legal_moves(board_normal,int(flat_last_val))

        legal_moves_symmetries = generate_ultimate_ttt_symmetries(last_val_on_board)
        flat_board_symmetries = generate_ultimate_ttt_symmetries(flat_board.reshape((9, 9)))
        probabilities_symmetries = generate_ultimate_ttt_symmetries(flat_prob.reshape((9, 9)))
        for i in range(8):
            last_val_on_board = legal_moves_symmetries[i,:,:]
            flat_board = flat_board_symmetries[i, :, :]
            flat_prob = probabilities_symmetries[i, :, :]

            input_tensor = from_numpy_to_onehot(flat_board).squeeze(0)


            # Convert numpy to torch tensor
            last_val_tensor = torch.tensor(last_val_on_board, dtype=input_tensor.dtype)

            # Stack to get [3, 9, 9]
            input_tensor = torch.cat([input_tensor, last_val_tensor.unsqueeze(0)], dim=0)




            prob_tensor = torch.tensor(flat_prob.flatten(), dtype=torch.float32)
            val_tensor = torch.tensor(flat_val[0], dtype=torch.float32)  # grab scalar from array


            input_tensors.append(input_tensor)
            prob_tensors.append(prob_tensor)
            val_tensors.append(val_tensor)

        # Stack once
    input_tensors = torch.stack(input_tensors)
    prob_tensors = torch.stack(prob_tensors)
    val_tensors = torch.stack(val_tensors)

    dataset = TensorDataset(input_tensors, prob_tensors, val_tensors)

    return dataset


def create_dataset_csv_ultimate(games_number = 100,deepness=100, model=None,game_class=UltimateToe):
    sim_class = game_class
    if model is None:
        UpgradedModel = MonteSearchModel(sim_class=sim_class, deepness=deepness, simulations=10, eval=False)
    else:
        UpgradedModel = model

    game = sim_class()
    OldModel = RandomModel()

    upgraded_model_player = 1

    # two boards, last move, value
    rows_numpy = np.zeros((2,int(81*2 +1+1)))

    for i in range(games_number):
        game = sim_class()

        if i % 2 == 0:
            game.current_player *= -1
            upgraded_model_player *= -1

        while not game.is_terminal():
            if int(game.current_player) == upgraded_model_player:
                move = UpgradedModel.next_move(game)

                board,prob,val = UpgradedModel.tree.get_state_probabilities(UpgradedModel.tree)
                board_adj = board * upgraded_model_player
                val_adj = val * upgraded_model_player

                if len(game.moves) > 2:
                    last_move = game.moves[-2]

                    last_move = from_tuple_to_int(last_move[0],last_move[1])

                else:
                    last_move = -1

                row = np.concatenate([board_adj.flatten(),[last_move], prob.flatten(), [val_adj]])

                rows_numpy = np.concatenate([rows_numpy,row.reshape((1,len(row)))],axis=0)

            else:
                move = OldModel.next_move(game)

            a = game.board

            game.current_player *= -1

    columns = [f'board_{i}' for i in range(81)] +['last_move']+\
              [f'policy_{i}' for i in range(81)] + \
              ['value']

    df = pd.DataFrame(rows_numpy[2:,:],columns=columns)
    return df



if __name__ == '__main__':
    df = create_dataset_csv_ultimate(games_number = 1000,deepness=200, model=None,game_class=UltimateToe)
    df.to_csv('/Users/pietropezzoli/Desktop/Thesis Pietro Pezzoli/tesi/pythonProject/Ultimate-Solver/Data/ultimate_toe_4.csv')