import numpy as np

from AlphaZero import train_alpha_zero, make_models_clash, AlphaZeroPlayer, RandomPlayer
from UltimateToeFile import UltimateToe
from UltimateNet import UltimatePolicyValueNet

small_model_intermediate_channels = 256

small_model = UltimatePolicyValueNet(board_side_size=9, channels=3, intermediate_channels=small_model_intermediate_channels)
# Train model (short training for demonstration)
small_model.load_model('/Users/pietropezzoli/Desktop/Thesis Pietro Pezzoli/tesi/pythonProject/Ultimate-Solver/checkpoints/SmallAlphaCheckPoints/alphazero256_1.pth')
trained_small_model = train_alpha_zero(
    sim_class=UltimateToe,
    iterations=1,
    games_per_iter=5,
    model=small_model,
    depth=81,
    temperature=100.0,
    epochs=30,
    batch_size=256
)

trained_small_model.save_model('/Users/pietropezzoli/Desktop/Thesis Pietro Pezzoli/tesi/pythonProject/Ultimate-Solver/checkpoints/SmallAlphaCheckPoints/alphazero256_1.pth')
randomplayer = RandomPlayer()

smallplayer = AlphaZeroPlayer(trained_small_model,UltimateToe,depth=81,c_puct=1.4)

make_models_clash(smallplayer,randomplayer,'small','random',100)
