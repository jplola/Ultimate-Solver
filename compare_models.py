from monte_carlo_tree_search import MCTSmodel,RandomModel
from convol import CONVOLmodel
from Simple_Simulator import SimpleTicTacToe
import numpy as np
from UltimateToeFile import UltimateToe

albero_score = 0
albero_improved_score = 0
sim_class = UltimateToe
total_game_numbers = 100

albero_model = RandomModel()
albero_model_improved = MCTSmodel(deepness=81,simulations=10,sim_class=sim_class)

for i in range(total_game_numbers):
    game = sim_class()
    if i%2==0:
        game.current_player = 1
    else:
        game.current_player = -1

    first_to_go = game.current_player
    while not game.is_terminal():
        cur = game.current_player
        if game.current_player == -1:
            move = albero_model.next_move(game)
            albero_model_improved.make_opponent_move(move,game)
        else:
            move = albero_model_improved.next_move(game,in_place=True)
            #albero_model.make_opponent_move(move,game)

        a = game.visualise_board()


    #albero_model.return_to_root()
    albero_model_improved.return_to_root()

    if game.winner == -1:
        albero_score +=1
    elif  game.winner == 1:
        albero_improved_score +=1

    print(f'game numba = {i}: Random Model = {albero_score}, albero_model_improved = {albero_improved_score}, fist = {first_to_go}')



