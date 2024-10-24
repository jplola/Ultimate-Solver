import pygame
import time
import numpy as np


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

def print_game(state, delay=1):

    def state_to_board(state):
        board = [[[[None for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]  # 9x9 grid

        for i, a in enumerate(state):
            large_row = i // 27
            large_col = (i % 9) // 3
            small_col = i % 3
            small_row = (i // 9) % 3

            if a == 1:
                board[large_row][large_col][small_row][int(small_col)] = 'X'
            elif a == -1:
                board[large_row][large_col][small_row][int(small_col)] = 'O'
            else:
                pass
        return board

    pygame.init()
    clock = pygame.time.Clock()

    # Set up the display
    window_size = (600, 600)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Ultimate Tic-Tac-Toe")

    WHITE = (255, 255, 255)  # Assuming you defined white somewhere



    # Fill the screen with white background
    screen.fill(WHITE)

    # Draw the game grid
    draw_grid(screen)

    # Draw the X and O symbols
    draw_symbols(screen, state_to_board(state), None)

    # Update the display
    pygame.display.flip()

    # Control the speed of updates (delay in seconds)
    time.sleep(delay)


    pygame.quit()


def print_game_history(states, delay=1):
    def state_to_board(state):
        board = [[[[None for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]  # 9x9 grid

        for i, a in enumerate(state):
            large_row = i // 27
            large_col = (i % 9) // 3
            small_col = i % 3
            small_row = (i // 9) % 3

            if a == 1:
                board[large_row][large_col][small_row][int(small_col)] = 'X'
            elif a == -1:
                board[large_row][large_col][small_row][int(small_col)] = 'O'
            else:
                pass
        return board

    pygame.init()
    clock = pygame.time.Clock()

    # Set up the display
    window_size = (600, 600)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Ultimate Tic-Tac-Toe")

    WHITE = (255, 255, 255)  # Assuming you defined white somewhere


    for state in states:
        # Fill the screen with white background
        screen.fill(WHITE)

        # Draw the game grid
        draw_grid(screen)

        # Draw the X and O symbols
        draw_symbols(screen, state_to_board(state), None)

        # Update the display
        pygame.display.flip()

        # Control the speed of updates (delay in seconds)
        time.sleep(delay)


    pygame.quit()


# Helper function to draw the grid lines
def draw_grid(screen):
    # Drawing large 3x3 grid
    for x in range(1, 3):
        pygame.draw.line(screen, BLACK, (x * 200, 0), (x * 200, 600), 5)
        pygame.draw.line(screen, BLACK, (0, x * 200), (600, x * 200), 5)

    # Drawing inner 3x3 grids for each large cell
    for row in range(3):
        for col in range(3):
            start_x = col * 200
            start_y = row * 200
            for i in range(1, 3):
                pygame.draw.line(screen, BLACK, (start_x + i * 66, start_y), (start_x + i * 66, start_y + 200), 2)
                pygame.draw.line(screen, BLACK, (start_x, start_y + i * 66), (start_x + 200, start_y + i * 66), 2)


# Draw X and O symbols
def draw_symbols(screen, board_state,large_board_state):
    for large_row in range(3):
        for large_col in range(3):
            for small_row in range(3):
                for small_col in range(3):
                    value = board_state[large_row][large_col][small_row][small_col]
                    if value is not None:
                        center_x = large_col * 200 + small_col * 66 + 33
                        center_y = large_row * 200 + small_row * 66 + 33
                        if value == 'X':
                            pygame.draw.line(screen, RED, (center_x - 20, center_y - 20), (center_x + 20, center_y + 20), 5)
                            pygame.draw.line(screen, RED, (center_x + 20, center_y - 20), (center_x - 20, center_y + 20), 5)
                        elif value == 'O':
                            pygame.draw.circle(screen, BLUE, (center_x, center_y), 25, 5)

    if large_board_state is not None:
        for large_row in range(3):
            for large_col in range(3):
                center_x = large_col * 200 + 100
                center_y = large_row * 200 + 100
                value = large_board_state[large_row][large_col]
                if value is not None:
                    if value == 'X':
                        pygame.draw.line(screen, RED, (center_x - 100, center_y - 100), (center_x + 100, center_y + 100), 5)
                        pygame.draw.line(screen, RED, (center_x + 100, center_y - 100), (center_x - 100, center_y + 100), 5)
                    elif value == 'O':
                        pygame.draw.circle(screen, BLUE, (center_x, center_y), 100, 5)




def print_gameplay(board: np.array((3, 3))):
    def print_row(row):
        stuff = ''
        if row[0] == 1:
            stuff += 'X |'
        elif row[0] == -1:
            stuff += 'O |'
        else:
            stuff += ' - |'
        if row[1] == 1:
            stuff += 'X '
        elif row[1] == -1:
            stuff += 'O '
        else:
            stuff += ' -  '
        if row[2] == 1:
            stuff += '| X '
        elif row[2] == -1:
            stuff += '| O '
        else:
            stuff += '| - '

        print(stuff)

    def print_hline():
        print('-------------')

    for i in range(3):
        row = board[i, :]
        print_row(row)
        if i < 2:
            print_hline()


