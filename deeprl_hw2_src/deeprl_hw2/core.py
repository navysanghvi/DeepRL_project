"""Core classes."""
from collections import deque, namedtuple
import random
import numpy as np

# Used to store observed experience from an MDP. 
#Represents a standard (s, a, r, s', terminal) tuple.
sample = namedtuple('sample', 'state, action, reward, next_state, is_terminal')
        
class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        return self.data[(self.start + i) % self.maxlen]
    
    def append(self, value):
        if self.length < self.maxlen:
            self.length += 1
        elif self.length == self.maxlen:
            self.start = (self.start + 1) % self.maxlen  # Old values removed once memory fills up
        self.data[(self.start + self.length - 1) % self.maxlen] = value
    
class ReplayMemory(object):
    def __init__(self, window_length,max_size):                 
        self.window_length = window_length                          # Number of frames to stack together as a sequence
        self.recent_observations = deque(maxlen=window_length)      # Stores the most recent frames
        self.recent_terminals = deque(maxlen=window_length)         # If next frame to the current observation is terminal or not
        self.max_size = max_size                                    # Maximum Replay memory size
        #Implemented as ring buffer instead of deque to avoid slow sampling
        self.actions = RingBuffer(max_size)                         
        self.rewards = RingBuffer(max_size)
        self.terminals = RingBuffer(max_size)
        self.observations = RingBuffer(max_size)
        
    def sample(self, batch_size,indexes=None):
        if indexes is None:      
           indexes = random.sample(xrange(1, len(self.observations)), batch_size) # Select random index from the filled replay memory
        sequence_batch = []
        #Check to ensure states don't leak through episodes
        for i in indexes:
            # State i-2 indicates if current state(i-1) is terminal or not 
            check_terminal = self.terminals[i - 2] if i>=2 else False
            while check_terminal: # If current index represents a terminal state, choose new index
                i = random.sample(xrange(1,len(self.observations)), 1)[0]
                check_terminal = self.terminals[i - 2] if i >= 2 else False
            state = [self.observations[i - 1]] # Choose this frame as first state of sequence
            for j in range(0, self.window_length - 1):
                current_i = i-j-2       # Check if any of the subsequent states after the chosen index are terminal
                current_terminal = self.terminals[current_i - 1] if current_i - 1 > 0 else False
                if current_i < 0 or current_terminal:
                    break
                state.insert(0, self.observations[current_i])  # If not terminal, populate sequence
            while len(state) < self.window_length:
                state.insert(0, np.zeros((state[0]).shape))    # Zero observation used if sufficient states not found due to terminal 
            action = self.actions[i - 1]                       # Reward, action for current frame and next sequence observations set
            reward = self.rewards[i - 1]
            is_terminal = self.terminals[i - 1]
            next_state = [np.copy(x) for x in state[1:]]        
            next_state.append(self.observations[i])
            sequence_batch.append(sample(state=state, action=action, reward=reward,            # Experience batch created
                                          next_state=next_state, is_terminal=is_terminal))
        return sequence_batch

    def append(self, observation, action, reward, terminal, training=True):
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)
    
   



