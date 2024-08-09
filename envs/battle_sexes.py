import gymnasium as gym
from gymnasium import Env

from envs.game import CoordinationGame


class BattleSexes(CoordinationGame):
    def __init__(self, joint_action: bool):
        super().__init__(joint_action)
        self.action_space: int = 2
        self.observation_space = 2
        self.payoff_matrix = [
            [(2, 1), (0, 0)],
            [(0, 0), (1, 2)]
        ]