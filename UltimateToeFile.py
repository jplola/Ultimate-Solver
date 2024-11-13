from Simple_Simulator import SimpleTicTacToe
import numpy as np
from copy import deepcopy


class UltimateToe(SimpleTicTacToe):
    def __init__(self, game=None):
        super().__init__(game=None)
        if game is None:
            self.board = SimpleTicTacToe()
            self.small_boards = [SimpleTicTacToe() for _ in range(9)]
            self.current_player = 1
            self.legal_moves = list(
                (i, legal_move) for i in range(9) for legal_move in self.small_boards[i].get_legal_moves())
            self.winner = 0
            self.moves = []
            self.turns = []
        else:
            self.board = SimpleTicTacToe(game.board)
            self.small_boards = [SimpleTicTacToe(board) for board in game.small_boards]
            self.current_player = game.current_player
            self.legal_moves = [move for move in game.legal_moves]
            self.winner = game.winner
            self.moves = [move for move in game.moves]
            self.turns = [turn for turn in game.turns]

    def reset(self, game=None):
        if game is None:
            self.board = SimpleTicTacToe()
            self.small_boards = [SimpleTicTacToe() for _ in range(9)]
            self.current_player = 1
            self.legal_moves = list(
                (i, legal_move) for i in range(9) for legal_move in self.small_boards[i].get_legal_moves())
            self.winner = 0
            self.moves = []
        else:
            self.board = SimpleTicTacToe(game.board)
            self.small_boards = [SimpleTicTacToe(board) for board in game.small_boards]
            self.current_player = game.current_player
            self.legal_moves = [move for move in game.legal_moves]
            self.winner = game.winner
            self.moves = [move for move in game.moves]
            self.turns = [turn for turn in game.turns]

    def is_terminal(self):
        winner = self.board.check_board_win(self.board.board)
        self.winner = winner
        if self.winner != 0:
            return True
        else:
            return False

    def step_forward(self,move : tuple):

        small_board, action = move[0],move[1]
        if self.winner == 0 and (small_board, action) in self.legal_moves:
            # put the correct player in the small board
            self.small_boards[small_board].current_player = self.current_player
            # put the step in the small board
            self.small_boards[small_board].step_forward(action) # this is a SimpleTicTacToe method
            # check if the current board is won
            if self.small_boards[small_board].is_terminal():
                # set the correct player in the meta board
                self.board.current_player = self.current_player
                self.board.step_forward(small_board)
            # check if the next board is won or full or playable
            # if it is playable and not full
            # then it's not a terminal state:
            if not self.small_boards[action].is_terminal():
                # then the legal actions are the legal actions in the board:
                self.legal_moves = list((action, move) for move in self.small_boards[action].legal_moves)
            else:
                # if the next board has reached a terminal state
                # then the legal moves are all the legal moves across all boards
                # (if one of the other boards is terminal the legal moves are set to [] )
                self.legal_moves = [
                    (i, small_move)
                    for i, small_board in enumerate(self.small_boards)
                    for small_move in small_board.legal_moves
                ]


            self.winner = self.board.winner
            self.moves.append((small_board, action))
            self.turns.append(self.current_player)

        return self

    def random_self_play(self):
        """ this game if starts from void position, then has a bias of ??? for the starter """

        terminal = self.is_terminal()
        while not terminal:
            smart_legal_moves = self.legal_moves
            index = np.arange(len(smart_legal_moves))
            index = np.random.choice(index)
            action = smart_legal_moves[index]
            self.step_forward(action)
            self.turns.append(self.current_player)
            self.current_player *= -1
            terminal = self.is_terminal()
            #visual = self.visualise_board()
            #print(visual)

    def simulate(self):
        self.random_self_play()
        if self.winner == -5:
            return 0#np.random.choice([-1,1])
        return self.winner

    def determine_bias_random_game(self,nsim = 1000):
        tot = 0
        for i in range(nsim):
            self.reset()
            self.random_self_play()
            winner = self.winner
            if winner == -5:
                winner = 0
            tot += winner

        tot = tot / nsim
        return tot / nsim

    def visualise_board(self):
        visual_board = np.zeros((9,9))
        for i,small_board in enumerate(self.small_boards):
            initial_row,initial_col = divmod(i,3)
            visual_board[int(initial_row * 3):int(initial_row * 3 +3),\
                    int(initial_col * 3):int(initial_col * 3 +3)] = small_board.board
        return visual_board

    def give_board(self):
        board = np.zeros((9,9))
        for i,miniboard in enumerate(self.small_boards):
            big_row,big_col = divmod(i,3)
            board[int(big_row*3):int(big_row*3 + 3),int(big_col*3):int(big_col*3 +3)] = miniboard.get_board()
        return board


game_ = UltimateToe()

game_.random_self_play()
#bias = game_.determine_bias_random_game(nsim = 10000)
