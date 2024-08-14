import random
from datetime import datetime
from math import sqrt, log

from agents.dqn import DQNAgent
from agents.know_dqn import KnowDQNAgent
from envs.climbing_game import ClimbingGame
from envs.game import CoordinationGame, ActionReport
from envs.battle_sexes import BattleSexes
from envs.guess_number import GuessNumber
from envs.simple_game import SimpleGame
from experience_buffer import ExperienceBuffer
import gymnasium as gym
import numpy as np
from keras.layers import Input, Dense, Softmax
from keras.optimizers import Adam
from keras.initializers.initializers import HeUniform
from keras.models import Sequential
import tensorflow as tf
from gymnasium import Env


def write_logs(step, agent: DQNAgent, reward):
    with agent.writer.as_default():
        tf.summary.scalar("reward", reward, step)


def simple_moving_average(data, window_size):
  """Calculates the simple moving average of a given data set.

  Args:
    data: The input data.
    window_size: The size of the moving average window.

  Returns:
    A numpy array containing the calculated moving averages.
  """

  weights = np.repeat(1.0, window_size) / window_size
  sma = np.convolve(data, weights, 'valid')
  return sma

def write_summary_logs(agent, rewards,agent_assumptions):
    with agent.writer.as_default():
        max_reward = np.max(agent.env.payoff_matrix)
        did_cordinate = []
        for reward in rewards:
            agent1_reward, agent2_reward = reward
            did_cordinate.append(int(agent1_reward == max_reward or agent2_reward == max_reward))

        for i,sample in enumerate(simple_moving_average(did_cordinate, 20)):
            tf.summary.scalar("Coordination Moving Avg(20)", sample, i)

        for i,sample in enumerate(simple_moving_average(agent_assumptions, 20)):
            tf.summary.scalar("Assumption Moving Avg(20)", sample, i)




def train(agents: [DQNAgent], env, max_episodes: int, max_steps: int):
    step_counter = 0
    q_iteraion = 10

    global_step = 0
    total_actions =[]
    total_rewards = []
    total_assumptions = []
    for episode in range(max_episodes):

        for step in range(max_steps):
            # Calculate Action

            actions = []
            assumptions = []
            states = []

            for agent in agents:
                state = env.base_state()
                state = np.reshape(state, [1, env.observation_space])

                assumption, action = agent.sampleAction(state)
                # state = assumption
                assumptions.append(assumption)
                actions.append(action)
                states.append(state)

            reports = env.step(*actions, *states, *assumptions)
            rewards = list(map(lambda report: report.reward, reports))
            total_actions.append(actions)
            total_rewards.append(rewards)
            total_assumptions.append(assumptions)
            for agent, report in zip(agents, reports):
                write_logs(global_step, agent, report.reward)
                agent.updateBuffer(report)
                agent.train()
                agent.decay_epsilon()

            step_counter += 1

            if step_counter > q_iteraion:
                for agent in agents:
                    agent.setTargetModel()
                step_counter = 0
            print(f"Episode {episode} : Number of steps {step}, reports: {list(rewards)} ")
            global_step += 1

    for i,agent in enumerate(agents):
        agent_assumptions = []
        for action,assumption in zip(total_actions,total_assumptions):
            agent_assumptions.append(int(action[len(agents)-i-1] == np.argmax(assumption[i])))
        write_summary_logs(agent, total_rewards,agent_assumptions)


if __name__ == '__main__':
    env = ClimbingGame(joint_action=True)
    log_dir = f"logs/MAS/climb/know-dqn/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    train(
        [KnowDQNAgent("agent1", log_dir, env), KnowDQNAgent("agent2", log_dir, env)],
        env,
        4,
        50
    )

x = 0
