import gymnasium as gym
from gymnasium import Env


class ClimbingGame():
    def __init__(self):
        self.n_actions = 3
        self.payoff_matrix = [
            [(11, 11), (-30, -30), (0, 0)],
            [(-30, -30), (7, 7), (6, 6)],
            [(0, 0), (0, 0), (5, 5)]
        ]
