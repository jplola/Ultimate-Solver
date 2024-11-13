from copy import deepcopy

import numpy as np
from Simple_Simulator import SimpleTicTacToe
from monte_carlo_tree_search import RandomModel
from UltimateToeFile import UltimateToe
from Augmentations import rotate_ninety_counter_data, final_form_data, rotate_ninety_counter,rotate_ninety_simple_toe_data
from MonteCarloSearch import MonteSearchModel
from torch.utils.data import DataLoader, TensorDataset
import torch

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

def make_MCTS_combat(games_played=10, first_deepness=20, second_deepness=10,
                      first_sim=10, second_sim=10, sim_class=UltimateToe,min_visits=50):
    albero_score = 0
    albero_improved_score = 0
    total_game_numbers = games_played


    albero_model = MonteSearchModel(deepness=first_deepness, simulations=first_sim, sim_class=sim_class)
    albero_model_improved = MonteSearchModel(deepness=second_deepness, simulations=second_sim, sim_class=sim_class)
    data = []
    for i in range(total_game_numbers):
        game = sim_class()
        if i % 2 == 0:
            game.current_player = 1
        else:
            game.current_player = -1

        first_to_go = game.current_player
        while not game.is_terminal():
            if game.current_player == -1:
                if len(game.moves)>0:
                    last_move = game.moves[-1]
                move = albero_model.next_move(game)

                prob = albero_model.tree.get_state_probabilities(albero_model.tree)
                if len(game.moves) > 1:
                    if np.count_nonzero(prob)<= 9:
                        data.append((game.visualise_board(),last_move,prob))
            else:
                if len(game.moves)>0:
                    last_move = game.moves[-1]
                move = albero_model_improved.next_move(game)

                prob = albero_model_improved.tree.get_state_probabilities(albero_model_improved.tree)
                if len(game.moves) > 1:
                    if np.count_nonzero(prob) <=9:
                        data.append((game.visualise_board(),last_move,prob))

        if game.winner == -1:
            albero_score += 1
        elif game.winner == 1:
            albero_improved_score += 1

        print(
            f'game numba = {i}: Random Model = {albero_score}, '
            f'albero_model_improved = {albero_improved_score}, fist = {first_to_go}')

    augmented_data = []

    for elem in data:
        if np.count_nonzero(elem[2]) > 0:

            ninety = rotate_ninety_counter_data(elem)
            augmented_data.append(final_form_data(deepcopy(ninety)))

            oneeighty = rotate_ninety_counter_data(ninety)
            augmented_data.append(final_form_data(deepcopy(oneeighty)))

            twoseventy = rotate_ninety_counter_data(oneeighty)
            augmented_data.append(final_form_data(deepcopy(twoseventy)))

            threesixty = rotate_ninety_counter_data(twoseventy)
            augmented_data.append(final_form_data(deepcopy(threesixty)))

    return augmented_data




def make_MCTS_combat_policy_and_value(games_played=10, first_deepness=36, second_deepness=36,
                      first_sim=10, second_sim=10, sim_class=UltimateToe,min_visits=50):
    albero_score = 0
    albero_improved_score = 0
    total_game_numbers = games_played


    albero_model = MonteSearchModel(deepness=first_deepness, simulations=first_sim, sim_class=sim_class)
    albero_model_improved = MonteSearchModel(deepness=second_deepness, simulations=second_sim, sim_class=sim_class)
    data = []
    for i in range(total_game_numbers):
        game = sim_class()
        if i % 2 == 0:
            game.current_player = 1
        else:
            game.current_player = -1

        first_to_go = game.current_player
        while not game.is_terminal():
            if game.current_player == -1:
                if len(game.moves)>0:
                    last_move = game.moves[-1]
                move = albero_model.next_move(game)

                prob = albero_model.tree.get_state_probabilities_simple(albero_model.tree)
                if len(game.moves) > 1:
                    if np.count_nonzero(prob)<= 9:
                        data.append((game.board,prob[1],albero_model.tree.total_reward/albero_model.tree.visits,game.current_player))
            else:
                if len(game.moves)>0:
                    last_move = game.moves[-1]
                move = albero_model_improved.next_move(game)

                prob = albero_model_improved.tree.get_state_probabilities_simple(albero_model_improved.tree)
                if len(game.moves) > 1:
                    if np.count_nonzero(prob) <=9:
                        data.append((game.board,prob[1],albero_model_improved.tree.total_reward/albero_model_improved.tree.visits,game.current_player))
            game.current_player *= -1
        if game.winner == -1:
            albero_score += 1
        elif game.winner == 1:
            albero_improved_score += 1

        print(
            f'game numba = {i}: Random Model = {albero_score}, '
            f'albero_model_improved = {albero_improved_score}, fist = {first_to_go}')
    return data


def create_dataset_simple_toe(games_number = 100,deepness = 100):

    data_set = []

    sim_class = SimpleTicTacToe

    UpgradedModel = MonteSearchModel(sim_class=sim_class, deepness=deepness, simulations=10, eval=False)
    OldModel = RandomModel()

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
                orig = UpgradedModel.tree.get_state_probabilities_simple(UpgradedModel.tree)
                data_set.append(orig)
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

    return data_set


def create_dataset_ultimate_toe(games_number = 100,deepness = 100):

    sim_class = UltimateToe

    UpgradedModel = MonteSearchModel(sim_class=sim_class, deepness=deepness, simulations=10, eval=False)

    OldModel = RandomModel()

    input_tensors = torch.empty((1,3,9,9))  # Adjust `input_shape` to match your input tensor shape
    prob_tensors = torch.empty((1,81))  # Adjust `prob_shape` to match your prob tensor shape
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
                board,not_legal,prob,val = UpgradedModel.tree.get_state_probabilities(UpgradedModel.tree)
                board *= game.current_player
                board_tensor = from_numpy_to_onehot(board)
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

#create_dataset_ultimate_toe(games_number = 100,deepness = 9)