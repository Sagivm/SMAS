import gymnasium as gym
from gymnasium import Env
from envs.game import CoordinationGame


class SimpleGame(CoordinationGame):
    def __init__(self, joint_action: bool):
        super().__init__(joint_action)
        self.action_space: int = 2
        self.observation_space = 2
        self.payoff_matrix = [
            [(3, 3), (5, 5)],
            [(0, 0), (10, 10)]
        ]
