from monte_carlo_tree_search import MCTSmodel,RandomModel
from convol import CONVOLmodel
from Simple_Simulator import SimpleTicTacToe
import numpy as np
from UltimateToeFile import UltimateToe

albero_score = 0
albero_improved_score = 0

total_game_numbers = 100

albero_model = MCTSmodel(deepness=10,simulations=9,sim_class=UltimateToe,c=0.5)
albero_model_improved = MCTSmodel(deepness=10,simulations=9,sim_class=UltimateToe)

for i in range(total_game_numbers):
    game = UltimateToe()
    if i%2==0:
        game.current_player = 1
    else:
        game.current_player = -1

    first_to_go = game.current_player
    while not game.is_terminal():
        if game.current_player == -1:
            albero_model.next_move(game)
        else:
            albero_model_improved.next_move(game)

    a = game.give_board()

    if game.winner == -1:
        albero_score +=1
    elif  game.winner == 1:
        albero_improved_score +=1

    print(f'game numba = {i}: Random Model = {albero_score}, albero_model_improved = {albero_improved_score}, fist = {first_to_go}')



