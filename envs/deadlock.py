import gymnasium as gym
from gymnasium import Env


class Deadlock():
    def __init__(self):
        self.n_actions = 2
        self.payoff_matrix = [
            [(-10, -10), (5, -1)],
            [(5, -1), (-2, -2)]
        ]
