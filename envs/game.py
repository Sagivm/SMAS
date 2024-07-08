import numpy as np
from random import randrange


class ActionReport:
    view = None

    def __init__(self, state, action: int, reward: int):
        self.state = state
        self.action = action
        self.reward = reward


class CoordinationGame:
    def __init__(self, joint_action):
        self.joint_action = joint_action
        self.action_space: int = 2
        self.observation_space = 2
        self.payoff_matrix = [
            [(3, 3), (5, 5)],
            [(0, 0), (10, 10)]
        ]

    def sample(self):
        return randrange(0, self.action_space)

    def step(self, a_action: int, b_action: int, a_state, b_state):
        actionReportA = ActionReport(
            a_state,
            a_action,
            self.payoff_matrix[a_action][b_action][0]
        )
        actionReportB = ActionReport(
            b_state,
            b_action,
            self.payoff_matrix[a_action][b_action][1]
        )
        if self.joint_action:
            actionReportA.view = actionReportB
            actionReportB.view = actionReportA

        return [actionReportA, actionReportB]

    def base_state(self):
        return np.zeros(self.observation_space)
