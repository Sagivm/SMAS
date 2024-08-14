import random
from datetime import datetime

from envs.game import CoordinationGame, ActionReport
from experience_buffer import ExperienceBuffer
import numpy as np
from keras.layers import Input, Dense, Softmax
from keras.optimizers import Adam
from keras.initializers.initializers import HeUniform
from keras.models import Sequential
import tensorflow as tf

class DQNAgent:

    def __init__(self,name,log_dir, env: CoordinationGame):
        self.name = name
        self.log_dir = f"{log_dir}/{name}"
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.env = env
        self.learning_rate = 0.9
        self.discount_factor = 0.95
        self.decay_rate = 0.95
        self.model = self.getModel(env.observation_space, env.action_space)
        self.target_model = self.getModel(env.observation_space, env.action_space)
        self.epsilon = 1
        self.action_count = [1] * env.action_space
        self.experience_buffer = ExperienceBuffer(10, 512)

    def getModel(self, env_space: tuple, action_space: int):
        model = Sequential([
            Input(shape=env_space),
            Dense(16, activation='relu', kernel_initializer=HeUniform(42)),
            Dense(16, activation='relu', kernel_initializer=HeUniform(42)),
            Dense(8, activation='relu', kernel_initializer=HeUniform(42)),
            Dense(action_space, activation='linear', kernel_initializer=HeUniform(42)),
        ])

        optimizer = Adam(learning_rate=self.learning_rate)

        model.compile(optimizer, loss='mse', metrics=['mse'])

        return model

    def updateBuffer(self, report: ActionReport):

        self.experience_buffer.add_experience(
            report.state, np.reshape(np.eye(self.env.action_space)[report.view.action], (1, self.env.action_space)),
            report.assumption, report.action, report.reward
        )

    # def sampleAction(self, state: np.ndarray):
    #     if random.uniform(0, 1) < self.epsilon:
    #         action = self.env.sample()
    #         return
    #     else:
    #         q_values = self.model.predict(state, verbose=0)
    #         action = np.argmax(q_values)
    #     self.action_count[action]+=1
    #     return action

    def sampleAction(self, state: np.ndarray):
        if random.uniform(0, 1) < self.epsilon:
            action = self.env.sample()
        else:
            q_values = self.model.predict(state, verbose=0)
            # ucb
            action = np.argmax(q_values + 4 * np.sqrt(np.log(self.action_count) / np.array(self.action_count)))

        self.action_count[action] += 1
        return state,action

    def train(self) -> float:
        v_state,_,_, v_action, v_reward = self.experience_buffer.sample_batch()

        qsa = self.model.predict(v_state, verbose=0)
        qsa_target = self.target_model.predict(v_state, verbose=0)

        y_j = np.copy(qsa)
        y_j[np.arange(y_j.shape[0]), v_action.T] = y_j[np.arange(y_j.shape[0]), v_action.T] * (
                1 - self.learning_rate) + self.learning_rate * (v_reward.T + self.discount_factor * np.max(qsa_target,
                                                                                                           axis=1))
        return self.model.train_on_batch(v_state, y_j)[0]

    def decay_epsilon(self):
        self.epsilon = 0 if self.epsilon * self.decay_rate < 0.05 else self.epsilon * self.decay_rate

    def setTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())
