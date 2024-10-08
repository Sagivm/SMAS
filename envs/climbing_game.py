import gymnasium as gym
from gymnasium import Env

from envs.game import CoordinationGame


class ClimbingGame(CoordinationGame):
    def __init__(self,joint_action: bool):
        super().__init__(joint_action)
        self.action_space: int = 3
        self.observation_space = 3
        self.payoff_matrix = [
            [(20, 20), (-10, -10), (0, 0)],
            [(-10, -10), (7, 7), (6, 6)],
            [(0, 0), (6, 6), (5, 5)]
        ]
