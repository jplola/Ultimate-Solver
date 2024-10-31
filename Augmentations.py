import numpy as np


def from_last_move_to_next_board(move: tuple) -> int:
    return move[1]


def rotate_ninety_counter(board: np.array) -> np.array:
    return np.rot90(board)


def rotate_last_move_ninety_counter(move: tuple) -> tuple:
    def rotate_component(mov):
        last = mov
        array = np.array([[0, 1, 2],
                          [3, 4, 5],
                          [6, 7, 8]])
        rotated_array = np.rot90(array, k=3)
        row, col = divmod(last, 3)
        return rotated_array[row, col]

    mo = rotate_component(move[0])
    ve = rotate_component(move[1])
    return mo, ve


def rotate_ninety_counter_data(state: tuple) -> tuple:
    if len(state) == 3:
        rotated_board = rotate_ninety_counter(state[0])
        rotated_probabilities = rotate_ninety_counter(state[2])
        rotated_last_move = rotate_last_move_ninety_counter(state[1])
        return rotated_board, rotated_last_move, rotated_probabilities
    if len(state) == 2:
        rotated_board = rotate_ninety_counter(state[0])
        rotated_probabilities = rotate_ninety_counter(state[2])
        return rotated_board, rotated_probabilities


def from_tuple_to_one_to81(move: tuple) -> int:
    big_row, big_col = divmod(move[0], 3)
    small_row, small_col = divmod(move[1], 3)
    array = np.array([[0, 1, 2],
                      [9, 10, 11],
                      [18, 19, 20]])
    relative = array[small_row, small_col]
    perfect = relative + 27 * big_row + 3 * big_col
    return int(perfect)


def final_form_data(state: tuple) -> tuple:
    last = state[1]
    last = from_tuple_to_one_to81(last)
    return state[0], last, state[2]
