from abc import ABC, abstractmethod

class Game(ABC):
    @abstractmethod
    def get_current_player(self):
        pass

    @abstractmethod
    def get_legal_moves(self):
        pass

    @abstractmethod
    def is_terminal(self):
        pass

    @abstractmethod
    def step_forward(self,action):
        pass

    @abstractmethod
    def simulate(self):
        pass

    @abstractmethod
    def random_self_play(self):
        pass

    @abstractmethod
    def get_board(self):
        pass