import pygame
import sys

from UltimateNet import UltimatePolicyValueNet
pygame.init()
# Constants
WIDTH, HEIGHT = 600, 600
CELL_SIZE = WIDTH // 9
LINE_COLOR = (0, 0, 0)
X_COLOR = (200, 0, 0)
O_COLOR = (0, 0, 200)
BG_COLOR = (255, 255, 255)
THIN_LINE = 1
THICK_LINE = 4
HIGHLIGHT_COLOR = (158, 207, 141)
font = pygame.font.SysFont(None, CELL_SIZE // 2)
big_font = pygame.font.SysFont(None, 60)
# Sample 9x9 board with 0, 1, -1
board = [[0 for _ in range(9)] for _ in range(9)]  # Replace with your actual board


screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ultimate Tic Tac Toe")
font = pygame.font.SysFont(None, CELL_SIZE // 2)


def draw_board(board,legal_moves):
    screen.fill(BG_COLOR)

    for big_index, small_index in legal_moves:
        big_row, big_col = divmod(big_index, 3)
        small_row, small_col = divmod(small_index, 3)
        row = big_row * 3 + small_row
        col = big_col * 3 + small_col
        rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, HIGHLIGHT_COLOR, rect)

    # Draw thin lines for small cells
    for i in range(1, 9):
        width = THICK_LINE if i % 3 == 0 else THIN_LINE
        pygame.draw.line(screen, LINE_COLOR, (i * CELL_SIZE, 0), (i * CELL_SIZE, HEIGHT), width)
        pygame.draw.line(screen, LINE_COLOR, (0, i * CELL_SIZE), (WIDTH, i * CELL_SIZE), width)

    # Draw Xs and Os
    for row in range(9):
        for col in range(9):
            val = board[row][col]
            if val != 0:
                text = font.render('X' if val == 1 else 'O', True, X_COLOR if val == 1 else O_COLOR)
                rect = text.get_rect(center=(col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2))
                screen.blit(text, rect)

def show_end_animation(result):
    if result == "win":
        message = "You Win!"
        color = (0, 200, 0)
    elif result == "lose":
        message = "You Lose!"
        color = (200, 0, 0)
    else:
        message = "Draw!"
        color = (100, 100, 255)

    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(200)
    overlay.fill((255, 255, 255))
    screen.blit(overlay, (0, 0))

    big_font = pygame.font.SysFont(None, 80)
    text = big_font.render(message, True, color)
    rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text, rect)
    pygame.display.flip()

    # Wait for a bit or until user closes window
    pygame.time.wait(2000)



def get_board_position(pos):
    x, y = pos
    row, col = y // CELL_SIZE, x // CELL_SIZE
    big_row, big_col = row // 3, col // 3
    small_row, small_col = row % 3, col % 3
    big_index = big_row * 3 + big_col
    small_index = small_row * 3 + small_col
    return (big_index, small_index)

from UltimateToeFile import UltimateToe
from AlphaZero import AlphaZeroPlayer

new_model_path = '/Users/pietropezzoli/Desktop/Thesis Pietro Pezzoli/tesi/pythonProject/Ultimate-Solver/checkpoints/SmallAlphaCheckPoints/alphazero128.pth'

model_new = UltimatePolicyValueNet(board_side_size=9, channels=3,intermediate_channels=128)
model_new.load_model(
    new_model_path)

pygame.init()

game = UltimateToe()
bot = AlphaZeroPlayer(
    model=model_new,
    sim_class=UltimateToe,
    depth=1000,
    c_puct=0.4
)

def show_menu():

    screen.fill((240, 240, 240))
    text1 = big_font.render("Play as X", True, (0, 0, 0))
    text2 = big_font.render("Play as O", True, (0, 0, 0))
    rect1 = text1.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
    rect2 = text2.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))

    screen.blit(text1, rect1)
    screen.blit(text2, rect2)
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if rect1.collidepoint(event.pos):
                    return 1
                elif rect2.collidepoint(event.pos):
                    return -1


# Start with menu
player = show_menu()

bot_player = -player
legal_moves = game.legal_moves
# Game loop
running = True
while running:
    draw_board(game.visualise_board(),legal_moves)
    pygame.display.flip()

    if game.current_player == bot_player:
        move = bot.next_move(game)
        game.step_forward(move)
        game.current_player *= -1
    else:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                indices = get_board_position(pos)
                if indices in game.legal_moves:
                    game.step_forward(indices)
                    game.current_player *= -1
                    continue
                print(indices)
    legal_moves = game.legal_moves

    if game.is_terminal():
        if game.winner == bot_player:
            show_end_animation('lose')
        elif game.winner == -bot_player:
            show_end_animation('win')
        else:
            show_end_animation('draw')

pygame.quit()
sys.exit()
