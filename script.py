
import pandas as pd

from CreateTrainingData import parse_dataset
from MonteCarloSearch_deprecated import MonteSearchModel
from RandomAgent import RandomModel
from SimpleToeFile import SimpleTicTacToe
from TrainModel import train_model
from UltimateNet import UltimatePolicyValueNet, UltimateNetworkModel
from UltimateToeFile import UltimateToe

path_1 = '/Users/pietropezzoli/Desktop/Thesis Pietro Pezzoli/tesi/pythonProject/Ultimate-Solver/Data/ultimate_toe_1.csv'

path_2 = '/Users/pietropezzoli/Desktop/Thesis Pietro Pezzoli/tesi/pythonProject/Ultimate-Solver/Data/ultimate_toe_3.csv'
df_1 = pd.read_csv(path_1)
df = pd.read_csv(path_2)

df = pd.concat([df,df_1],axis=0)



print('starting parsing')
tensordataset = parse_dataset(df,out_channels=3)
print('finished parsing')
model = UltimatePolicyValueNet(board_side_size=9, channels=3, intermediate_channels=64)
model = train_model(model, tensordataset, lr=0.001, batch_size=1024, num_epochs=25)
model.save_model('/Users/pietropezzoli/Desktop/Thesis Pietro Pezzoli/tesi/pythonProject/Ultimate-Solver/checkpoints/AlphaCheckpoints/model_checkpoints/mcts_1.pth')
new_model = UltimateNetworkModel(model=model, sim_class=UltimateToe)



UpgradedModel = new_model


OldModel = RandomModel()

upgraded_model_player = -1
tot_games = 1000

OldModel_score = 0
UpgradedModel_score = 0
for i in range(tot_games):
    game = UltimateToe()

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


    print(f'game numba = {i}: UpgradedModel = {UpgradedModel_score}, OldModel = {OldModel_score}')

print(f'perc = {UpgradedModel_score / tot_games}')