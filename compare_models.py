from monte_carlo_tree_search import MCTSmodel,RandomModel
from convol import CONVOLmodel
from Simple_Simulator import SimpleTicTacToe
import numpy as np
from UltimateToeFile import UltimateToe

albero_score = 0
albero_improved_score = 0

total_game_numbers = 1000

albero_model = RandomModel()
albero_model_improved = MCTSmodel(deepness=81,simulations=1,sim_class=UltimateToe)#CONVOLmodel(num_games_training=10000)

for i in range(total_game_numbers):
    game = UltimateToe()
    game.current_player *= np.random.choice([-1,1])

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

    print(f'game numba = {i}: Random Model = {albero_score}, albero_model_improved = {albero_improved_score}')



