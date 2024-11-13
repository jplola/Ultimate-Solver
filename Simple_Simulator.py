import numpy as np
import numba

class SimpleTicTacToe():
    def __init__(self,game = None):
        if game is None:
            self.current_player = 1
            self.board = np.zeros((3, 3))
            self.winner = 0
            self.legal_moves = [i for i in range(9)]
            self.moves = []
            self.turns = []
        else:
            self.current_player = game.current_player
            self.board = game.board.copy()
            self.winner = game.winner
            self.legal_moves = [move for move in game.legal_moves]
            self.moves = [move for move in game.moves]
            self.turns = [turn for turn in game.turns]

    def reset(self,game=None):
        if game is None:
            self.current_player = 1
            self.board = np.zeros((3,3))
            self.winner = 0
            self.legal_moves = [i for i in range(9)]
            self.moves = []
            self.turns = []

        else:
            self.current_player = game.current_player
            self.board = game.board.copy()
            self.winner = game.winner
            self.legal_moves = [move for move in game.legal_moves]
            self.moves = [move for move in game.moves]
            self.turns = [turn for turn in game.turns]


    def get_current_player(self):
        return self.current_player


    def get_legal_moves(self):
        return self.legal_moves

    @staticmethod
    @numba.jit(nopython=True)
    def check_board_win(board : np.array((3,3))):
        # Check rows, columns, and diagonals
        for row in board:
            if row[0] == row[1] == row[2] and row[0] != 0 and row[0] != -5:
                return row[0]

        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] and board[0][col] != 0 and board[0][col] != -5:
                return board[0][col]

        if board[0][0] == board[1][1] == board[2][2] and board[0][0] != 0 and board[0][0] != -5:
            return board[0][0]

        if board[0][2] == board[1][1] == board[2][0] and board[0][2] != 0 and board[0][2] != -5:
            return board[0][2]

        if np.all(board != 0):
            return -5

        return 0

    @staticmethod
    @numba.jit(nopython=True)
    def is_board_full(board):
        return np.all(board != 0)

    def is_terminal(self):
        winner = self.check_board_win(self.board)
        self.winner = winner
        if winner != 0:
            self.legal_moves = []
            return True
        else:
            return False


    def step_forward(self,action):
        if self.winner == 0 and action in self.legal_moves:
            row,col = divmod(action,3)
            self.board[row,col] = self.current_player
            self.legal_moves.remove(action)
            self.winner = self.check_board_win(self.board)
            self.moves.append(action)
            self.turns.append(self.current_player)
            #self.current_player *= -1

        return self

    @staticmethod
    def smart_combo(board : np.array((3,3)),player) -> set:
        result = -1 * np.ones(9)
        for i in range(3):
            row = np.equal(board[i,:],np.full(3,player))
            col = np.equal(board[:,i],np.full(3,player))
            if np.count_nonzero(row) == 2:
                rel_pos = np.nonzero(~row)[0][0]
                result[int(rel_pos + 3*i)] = int(rel_pos + 3*i)
            if np.count_nonzero(col) == 2:
                rel_pos = np.nonzero(~col)[0][0]
                result[int(3 * rel_pos + i)] = int(3 * rel_pos +  i)

        diag = np.equal(board.diagonal(),np.full(3,player))
        if np.count_nonzero(diag) == 2:
            rel_pos = np.nonzero(~diag)[0][0]
            result[int(3 * rel_pos + rel_pos)] = int(3 * rel_pos + rel_pos)

        opposite_diag = (board @ np.array([[0,0,1],[0,1,0],[1,0,0]])).diagonal()
        opposite_diag = np.equal(opposite_diag,np.full(3,player))
        if np.count_nonzero(opposite_diag) == 2:
            rel_pos = np.nonzero(~opposite_diag)[0]
            result[int(2 + 2 * rel_pos)] = int(2 + 2 * rel_pos)

        result = np.array(result,dtype= int)
        result = set(result)
        return result.difference(set([-1]))



    def see_smart_move(self):
        if len(self.moves) > 2:
            moves_current_player = np.equal(self.turns,self.current_player)
            moves_current_player_count = np.count_nonzero(moves_current_player)
            moves_opponent_count = np.count_nonzero(~moves_current_player)
            if moves_opponent_count > 1:
                aux_losing = np.where(self.board == -self.current_player, self.board, 0)
                anti_loss_moves = self.smart_combo(aux_losing, -self.current_player)
                legal_moves = set(self.legal_moves)
                legal_moves = legal_moves.intersection(anti_loss_moves)
                if len(legal_moves) > 0:
                    return [elem for elem in legal_moves]
                else:
                    pass
            else:
                if moves_current_player_count > 1:
                    aux = np.where(self.board == self.current_player, self.board, 0)
                    winning_moves = self.smart_combo(aux, self.current_player)
                    legal_moves = set(self.legal_moves)
                    legal_moves = legal_moves.intersection(winning_moves)

                    if len(legal_moves) > 0:
                        return [elem for elem in legal_moves]
                    else:
                        pass

        return self.legal_moves


# to see smart move:
    # first check that the opponent is not winnning in the next move

    # if this is the case then that is indeed the smartest move, to avoid is winning.

    # else, if this is not the case there might be a possibility that the current player has a winning move


    def simulate(self):
        self.random_self_play()
        if self.winner == -5:
            return 0#np.random.choice([-1,1])
        return self.winner

    def random_self_play(self):
        """ this game if starts from void position, then has a bias of 30% for the starter """

        terminal = self.is_terminal()
        while not terminal:
            smart_legal_moves = self.legal_moves

            action = np.random.choice(smart_legal_moves)
            self.step_forward(action)
            self.current_player *= -1
            terminal = self.is_terminal()

    def guided_random_self_play(self):

        terminal = self.is_terminal()
        while not terminal:
            smart_legal_moves = self.see_smart_move()

            action = np.random.choice(smart_legal_moves)
            self.step_forward(action)
            self.current_player *= -1
            terminal = self.is_terminal()

    def get_board(self):
        return self.board


