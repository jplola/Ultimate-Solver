import numpy as np

from monte_carlo_tree_search import MCTSmodel,RandomModel
from UltimateToeFile import UltimateToe
from Augmentations import rotate_ninety_counter_data, final_form_data


def make_MCTS_combat(games_played=10, first_deepness=20, second_deepness=10,
                      first_sim=10, second_sim=10, sim_class=UltimateToe,min_visits=50):
    albero_score = 0
    albero_improved_score = 0
    total_game_numbers = games_played


    albero_model = RandomModel()#MCTSmodel(deepness=first_deepness, simulations=first_sim, sim_class=sim_class)
    albero_model_improved = MCTSmodel(deepness=second_deepness, simulations=second_sim, sim_class=sim_class)
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
                move = albero_model.next_move(game)
                albero_model_improved.make_opponent_move(move, game)
            else:
                if len(game.moves)>0:
                    last_move = game.moves[-1]
                move = albero_model_improved.next_move(game, in_place=False)
                albero_model_improved.return_to_root()
                prob = albero_model_improved.tree.get_state_probabilities(albero_model_improved.tree)
                if len(game.moves) > 1:
                    data.append((game.visualise_board(),last_move,prob))

                #albero_model.make_opponent_move(move, game)


        #albero_model.return_to_root()

        #other = albero_model.get_probabilities_for_visited_nodes_list(min_visits=min_visits)




        if game.winner == -1:
            albero_score += 1
        elif game.winner == 1:
            albero_improved_score += 1

        print(
            f'game numba = {i}: Random Model = {albero_score}, '
            f'albero_model_improved = {albero_improved_score}, fist = {first_to_go}')

    #albero_model_improved.tree.merge_trees(albero_model.tree)



    augmented_data = []

    for elem in data:
        if np.count_nonzero(elem[2]) > 0:
            augmented_data.append(final_form_data(elem))
            """ninety = rotate_ninety_counter_data(elem)
            augmented_data.append(final_form_data(ninety))

            oneeighty = rotate_ninety_counter_data(ninety)
            augmented_data.append(final_form_data(oneeighty))

            twoseventy = rotate_ninety_counter_data(oneeighty)
            augmented_data.append(final_form_data(twoseventy))

            threesixty = rotate_ninety_counter_data(twoseventy)
            augmented_data.append(final_form_data(threesixty))"""

    return augmented_data