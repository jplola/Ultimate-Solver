import pygame




# Helper function to draw the grid lines
def draw_grid(screen):
    # Drawing large 3x3 grid
    for x in range(1, 3):
        pygame.draw.line(screen, GREEN, (x * 200, 0), (x * 200, 600), 10)
        pygame.draw.line(screen, GREEN, (0, x * 200), (600, x * 200), 10)

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





# Get the cell that was clicked
def get_clicked_cell(mouse_pos):
    x, y = mouse_pos
    large_row = y // 200
    large_col = x // 200
    small_row = (y % 200) // 66
    small_col = (x % 200) // 66
    return (large_row, large_col), (small_row, small_col)


# Check if the move is valid
def is_valid_move(large_pos, small_pos, board, next_allowed_grid):
    if next_allowed_grid is not None:
        return large_pos == next_allowed_grid and board[large_pos[0]][large_pos[1]][small_pos[0]][small_pos[1]] is None
    else:
        return board[large_pos[0]][large_pos[1]][small_pos[0]][small_pos[1]] is None


# Check for small board win (standard Tic-Tac-Toe rules)
def check_small_board_win(small_board):
    # Check rows, columns, and diagonals
    for row in small_board:
        if row[0] == row[1] == row[2] and row[0] is not None:
            return row[0]

    for col in range(3):
        if small_board[0][col] == small_board[1][col] == small_board[2][col] and small_board[0][col] is not None:
            return small_board[0][col]

    if small_board[0][0] == small_board[1][1] == small_board[2][2] and small_board[0][0] is not None:
        return small_board[0][0]

    if small_board[0][2] == small_board[1][1] == small_board[2][0] and small_board[0][2] is not None:
        return small_board[0][2]

    return None


# Check if a small board is full
def is_small_board_full(large_pos, board):
    small_board = board[large_pos[0]][large_pos[1]]
    for row in small_board:
        for cell in row:
            if cell is None:
                return False
    return True


# Check for large board win (winning 3 small boards in a row)
def check_large_board_win(large_board):
    # Check rows, columns, and diagonals
    for row in large_board:
        if row[0] == row[1] == row[2] and row[0] is not None:
            return row[0]

    for col in range(3):
        if large_board[0][col] == large_board[1][col] == large_board[2][col] and large_board[0][col] is not None:
            return large_board[0][col]

    if large_board[0][0] == large_board[1][1] == large_board[2][2] and large_board[0][0] is not None:
        return large_board[0][0]

    if large_board[0][2] == large_board[1][1] == large_board[2][0] and large_board[0][2] is not None:
        return large_board[0][2]

    return None

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


if __name__ == '__main__':
    # Initialize Pygame
    pygame.init()

    # Set up the display
    window_size = (600, 600)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Ultimate Tic-Tac-Toe")

    # Define some basic colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 100, 0)

    # Variables to track game state
    board_state = [[[[None for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]  # 9x9 grid
    player_turn = 'X'
    large_board_state = [[None for _ in range(3)] for _ in range(3)]
    next_allowed_grid = None  # Starts with allowing any grid

    # Game loop control
    running = True

    # Game loop
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                large_pos, small_pos = get_clicked_cell(mouse_pos)

                if is_valid_move(large_pos, small_pos, board_state, next_allowed_grid):
                    # Make the move
                    board_state[large_pos[0]][large_pos[1]][small_pos[0]][small_pos[1]] = player_turn

                    # Check if the small board is won
                    small_board = board_state[large_pos[0]][large_pos[1]]
                    small_winner = check_small_board_win(small_board)
                    if small_winner and large_board_state[large_pos[0]][large_pos[1]] is None:
                        large_board_state[large_pos[0]][large_pos[1]] = small_winner



                    # Check if the large board is won
                    large_winner = check_large_board_win(large_board_state)
                    if large_winner:
                        print(f"{large_winner} wins the game!")
                        running = False

                    # Update next allowed grid
                    if not is_small_board_full(small_pos, board_state):
                        next_allowed_grid = small_pos
                    else:
                        next_allowed_grid = None

                    if large_board_state[small_pos[0]][small_pos[1]] == 'X' or  large_board_state[small_pos[0]][small_pos[1]] == 'O':
                        next_allowed_grid = None
                    else:
                        next_allowed_grid = small_pos

                    # Switch player turn
                    player_turn = 'O' if player_turn == 'X' else 'X'

        # Fill the screen with white background
        screen.fill(WHITE)

        # Draw the game grid
        draw_grid(screen)
        # Draw the X and O symbols
        draw_symbols(screen, board_state,large_board_state)

        # Update the display
        pygame.display.flip()

    # Quit Pygame
    pygame.quit()
