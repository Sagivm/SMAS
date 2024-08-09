import gymnasium as gym
from gymnasium import Env

from envs.game import CoordinationGame


class GuessNumber(CoordinationGame):
    def __init__(self, joint_action: bool):
        super().__init__(joint_action)
        self.action_space: int = 3
        self.observation_space = 3
        self.payoff_matrix = [
            [(1, 1), (0, 0), (0, 0)],
            [(0, 0), (1, 1), (0, 0)],
            [(0, 0), (0, 0), (1, 1)]
        ]
