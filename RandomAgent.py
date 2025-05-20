from numpy.random import choice
from numpy import arange
class RandomModel:
    def next_move(self, state):
        if not state.is_terminal():
            index = choice(arange(len(state.legal_moves)))
            my_move = state.legal_moves[index]
            state.step_forward(my_move)
            state.current_player *=-1
            return my_move
        else:
            return False
