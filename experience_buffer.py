import numpy as np


class ExperienceBuffer:
    """
    Experience buffer class
    Operates as an experience buffer storing (state,action,reward,newstate,done) vectors to be replayed
    """

    def __init__(self, batch_size: int, max_size: int):
        self.size = 0
        self.current_index = 0
        self.v_state = None
        self.v_assumption = None
        self.v_view = None
        self.v_action = None
        self.v_reward = None
        self.batch_size = batch_size
        self.max_size = max_size

    def add_experience(self, state, view, assumption, action: int, reward: float) -> None:
        """
        Add experience to the buffer, if the addition exceeds the current size of the buffer remove the oldest instance

        :param action:
        :type action:
        :param reward:
        :type reward:
        :return:
        :rtype:
        """

        if self.size > self.max_size:
            self.v_state[self.current_index % self.max_size, :] = state
            self.v_view[self.current_index % self.max_size, :] = view
            self.v_assumption[self.current_index % self.max_size, :] = assumption
            self.v_action[self.current_index % self.max_size, :] = action
            self.v_reward[self.current_index % self.max_size, :] = reward
        else:
            if self.size != 0:
                self.v_state = np.vstack((self.v_state, state))
                self.v_view = np.vstack((self.v_view, view))
                self.v_assumption = np.vstack((self.v_assumption, assumption))
                self.v_action = np.vstack((self.v_action, action))
                self.v_reward = np.vstack((self.v_reward, reward))
            else:
                if self.size < self.max_size:
                    self.v_state = state
                    self.v_view = view
                    self.v_assumption = assumption
                    self.v_action = np.array([action])
                    self.v_reward = np.array([reward])
            self.size += 1
        self.current_index = self.current_index % self.max_size + 1

    def sample_batch(self) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray, np.ndarray]:
        """
        Random sample a batch of at least size batch_size out of the experience buffer,
        :return:
        :rtype:
        """
        batch_size = self.batch_size if self.batch_size <= self.v_action.shape[0] else self.size
        # idx = np.random.randint(self.v_action.shape[0], size=batch_size)
        idx =list(range(-batch_size,0))
        # return self.v_state[-batch_size], self.v_view[-batch_size], self.v_assumption[-batch_size], self.v_action[-batch_size], self.v_reward[-batch_size]
        return self.v_state[idx],self.v_view[idx],self.v_assumption[idx],self.v_action[idx], self.v_reward[idx]
