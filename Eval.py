import numpy as np



def evaluate_tic_tac_toe(player, board: np.array((3, 3))):
    def check_v_shape(player):
        # Define V shapes with their coordinates and their potential open endpoints
        v_shapes = [
            ([(0, 0), (0, 1), (1, 1)], [(0, 2), (2, 2)]),
            ([(0, 2), (0, 1), (1, 1)], [(0, 0), (2, 0)]),
            ([(0, 0), (1, 0), (1, 1)], [(2, 0), (2, 2)]),
            ([(2, 0), (1, 0), (1, 1)], [(0, 0), (0, 2)]),
            ([(2, 0), (1, 1), (2, 1)], [(0, 1), (0, 2)]),
            ([(2, 1), (2, 2), (1, 1)], [(0, 0), (0, 1)]),
            ([(2, 2), (1, 1), (1, 2)], [(0, 0), (0, 2)]),
            ([(1, 2), (1, 1), (0, 2)], [(0, 2), (0, 1)]),
        ]

        # Iterate over each V shape pattern
        for shape, open_ends in v_shapes:
            # Check if the cells in the shape match the player's marker
            if all(board[x, y] == player for x, y in shape):
                # Check if either of the open end positions is empty
                if any(board[x, y] == 0 for x, y in open_ends):
                    return True

        # Return False if no matching V shape pattern with an open end is found
        return False

    def check_large_v(player):
        # Define large V shapes with their coordinates and open endpoints
        large_v_shapes = [
            ([(0, 0), (1, 1), (0, 2)], [(2, 0), (2, 2)]),
            ([(0, 0), (1, 1), (2, 0)], [(0, 2), (2, 2)]),
            ([(2, 0), (1, 1), (2, 2)], [(0, 0), (0, 2)]),
            ([(2, 2), (1, 1), (0, 2)], [(0, 0), (2, 0)]),
        ]

        # Iterate over each large V shape pattern
        for shape, open_ends in large_v_shapes:
            # Check if the cells in the shape match the player's marker
            if all(board[x, y] == player for x, y in shape):
                # Check if either of the open end positions is empty
                if any(board[x, y] == 0 for x, y in open_ends):
                    return True

        # Return False if no matching large V shape pattern with an open end is found
        return False

    def check_l_shape(player):
        # Define L shapes with their coordinates and open endpoints
        l_shapes = [
            ([(0, 0), (0, 1), (1, 0)], [(0, 2), (2, 0)]),  # Top-left corner L
            ([(0, 1), (0, 2), (1, 2)], [(0, 0), (2, 2)]),  # Top-right corner L
            ([(1, 0), (2, 0), (2, 1)], [(2, 2), (0, 0)]),  # Bottom-left corner L
            ([(2, 1), (2, 2), (1, 2)], [(2, 0), (0, 2)]),  # Bottom-right corner L
        ]

        # Iterate over each L shape pattern
        for shape, open_ends in l_shapes:
            # Check if the cells in the shape match the player's marker
            if all(board[x, y] == player for x, y in shape):
                # Check if at least one of the open end positions is empty
                if any(board[x, y] == 0 for x, y in open_ends):
                    return True

        # Return False if no matching L shape pattern with an open end is found
        return False

    def has_winning_opportunity(player):
        # Check rows for a potential winning opportunity
        for i in range(3):
            row = board[i, :]
            if ((row == [player, player, 0]).all() or
                    (row == [player, 0, player]).all() or
                    (row == [0, player, player]).all()):
                return True

        # Check columns for a potential winning opportunity
        for i in range(3):
            col = board[:, i]
            if ((col == [player, player, 0]).all() or
                    (col == [player, 0, player]).all() or
                    (col == [0, player, player]).all()):
                return True

        # Check main diagonal for a potential winning opportunity
        main_diag = np.array([board[i, i] for i in range(3)])
        if ((main_diag == [player, player, 0]).all() or
                (main_diag == [player, 0, player]).all() or
                (main_diag == [0, player, player]).all()):
            return True

        # Check anti-diagonal for a potential winning opportunity
        anti_diag = np.array([board[i, 2 - i] for i in range(3)])
        if ((anti_diag == [player, player, 0]).all() or
                (anti_diag == [player, 0, player]).all() or
                (anti_diag == [0, player, player]).all()):
            return True

        # If no winning opportunity is found, return False
        return False

    return check_large_v(player) or check_l_shape(player) or check_v_shape(player) \
            or has_winning_opportunity(player)



import numpy as np


# Example usage with a 3x3 NumPy array
board = np.array([
    [-1, -1, 0],
    [1, 1, 0],
    [1, -1, 0]
])
player = -1  # Check for player X's (1) winning opportunity
has_opportunity = evaluate_tic_tac_toe(player, board)
print(f"Player {player} has a winning opportunity:", has_opportunity)
