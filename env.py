import gym
import numpy as np
import random


class TicTacEnv(gym.Env):

  def __init__(self, as_x = True):
    self.action_space = gym.spaces.Discrete(9)
    self.observation_space = gym.spaces.Box(low=-1, high = 1, shape = (9,), dtype = int)
    self.as_x = as_x

  def step(self, action):

    state = self.get_curr_state()
    avail = self.get_available(state)

    if action in avail:

      self.update_state(action, 1)
      self.random_agent()
      state = self.get_curr_state()
      reward, done = self.reward_func(state)

    else:
      done = True
      reward = -10

    info = {}

    return state, reward, done, info
      

  def render(self):
    ...

  def reset(self):
    self._state = np.zeros(9)
    return self._state

  def get_curr_state(self):
    return self._state

  def random_agent(self):
    state = self.get_curr_state()
    avail = self.get_available(state)
    if len(avail) > 0:
      move = random.choice(avail)
      self.update_state(move, -1)

  def update_state(self, index, player: int):
    self._state[index] = player

  def get_available(self, state):
    available = []
    for i in range(9):
      if state[i] == 0:
        available.append(i)
    return available

  def reward_func(self, nstate):
    '''Checks board/state for a winner

    Returns:    1 if the winner is X,
    -1 if the winner is O or game ends draw
    '''
    total_reward = 0
    available =  self.get_available(nstate)
    done = False

    if (nstate[0] == nstate[1] == nstate[2]) and not nstate[0] == 0: # row 1 winner
        total_reward = nstate[0]	
    elif (nstate[3] == nstate[4] == nstate[5]) and not nstate[3] == 0: # row 2 winner
        total_reward = nstate[3]	
    elif (nstate[6] == nstate[7] == nstate[8]) and not nstate[6] == 0: # row 3 winner
        total_reward = nstate[6]	
    elif (nstate[0] == nstate[3] == nstate[6]) and not nstate[0] == 0: # column 1 winner
        total_reward = nstate[0]	
    elif (nstate[1] == nstate[4] == nstate[7]) and not nstate[1] == 0: # column 2 winner
        total_reward = nstate[1]	
    elif (nstate[2] == nstate[5] == nstate[8]) and not nstate[2] == 0: # column 3 winner
        total_reward = nstate[2]	
    elif (nstate[0] == nstate[4] == nstate[8]) and not nstate[0] == 0: # lr diagonal winner
        total_reward = nstate[0]	
    elif (nstate[2] == nstate[4] == nstate[6]) and not nstate[2] == 0: # rl diagonal winner
        total_reward = nstate[2]

    if total_reward == 0 and len(available) == 0:
        total_reward = 0
        done = True

    if total_reward != 0:
      done = True

    return total_reward, done

  def get_positions(self):
    x_pos = np.zeros(9)
    o_pos = np.zeros(9)
    state = self.get_curr_state()
    for i in range(9):
      if state[i] == 1:
        x_pos[i] = 1
      if state[i] == -1:
        o_pos[i] = 1

    x_pos = np.expand_dims(x_pos, 0)
    o_pos = np.expand_dims(o_pos, 0)

    return np.concatenate([x_pos, o_pos])