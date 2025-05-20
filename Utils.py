import shutil

import numpy as np
import torch
import numba
from typing import Union
def from_tuple_to_int(big, small):
    big_row, big_col = divmod(big, 3)
    small_row, small_col = divmod(small, 3)
    row = big_row * 3 + small_row
    col = big_col * 3 + small_col
    return int(row * 9 + col)


def from_int_to_tuple(idx):
    row, col = divmod(idx, 9)

    big_row, small_row = divmod(row, 3)
    big_col, small_col = divmod(col, 3)

    big = big_row * 3 + big_col
    small = small_row * 3 + small_col

    return int(big), int(small)


def from_numpy_to_onehot(board: np.array):
    player1 = (board == 1).astype(np.float32)
    player_minus1 = (board == -1).astype(np.float32)

    # Stack to create a (2, 3, 3) array and convert to tensor
    one_hot = np.stack([player1, player_minus1], axis=0)
    one_hot_tensor = torch.tensor(one_hot, dtype=torch.float32)

    one_hot_tensor = one_hot_tensor.unsqueeze(0)

    return one_hot_tensor

@numba.jit(nopython=True)
def is_board_full(board):
    return np.all(board != 0)

def board_legal_moves(board,last_move: Union[int,tuple]):

    if type(last_move) == int:
        if last_move < 0:
            return np.ones((9, 9))
        last_move = from_int_to_tuple(last_move)
    else:
        try:
            if last_move<0:
                return np.ones((9, 9))
        except:
            pass

    last_val_on_board = np.zeros((9, 9))
    small_board = last_move[1]
    initial_row, initial_col = divmod(small_board, 3)


    if is_board_full(board[int(initial_row * 3):int(initial_row * 3 + 3), \
                                       int(initial_col * 3):int(initial_col * 3 + 3)]
                                       ):

        last_val_on_board = np.abs(np.abs(board) - np.ones(board.shape))

    else:
        last_val_on_board = np.zeros((9, 9))
        selected_sub_board = np.abs(board)
        selected_sub_board = selected_sub_board[int(initial_row * 3):int(initial_row * 3 + 3), \
                             int(initial_col * 3):int(initial_col * 3 + 3)]

        last_val_on_board[int(initial_row * 3):int(initial_row * 3 + 3), \
        int(initial_col * 3):int(initial_col * 3 + 3)] = abs(selected_sub_board - 1)


    return last_val_on_board


import time
import os
import random

# ANSI color codes for fun
COLORS = ['\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m']
RESET = '\033[0m'

def supports_ansi():
    # Safe check for PyCharm or limited environments
    return os.getenv("TERM") is not None or os.getenv("PYCHARM_HOSTED") == "1"

 # Fallback: simulate clear

def print_colored_message(message):
    art = [
        r"         ____    ____     ____         ____    ________   ",
        r"       /  ___|  / ___|    \   \  ___  /   /   |   __   |  ",
        r"       | |  _  | |  _      \   \/   \/   /    |   _____|  ",
        r"       | |_| | | |_| |      \    / \    /     |  |        ",
        r"        \____|  \____|       \__/   \__/      |__|        ",
        r"",
        f"                              {message}                   ",
    ]

    ansi = supports_ansi()

    for _ in range(1):

        color = random.choice(COLORS) if ansi else ''
        for line in art:
            print(color + line + (RESET if ansi else ''))
        time.sleep(0.2)

def game_result_animation(won=True):
    if won:
        print_colored_message("🎉 YOU WON! 🎉")
    else:
        print_colored_message("💀 YOU LOST! 💀")

# Example usage
# game_result_animation(won=True)
# game_result_animation(won=False)

