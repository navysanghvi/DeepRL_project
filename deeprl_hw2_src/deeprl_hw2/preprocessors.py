"""Suggested Preprocessors."""

import numpy as np
from PIL import Image
from core import Preprocessor

class HistoryPreprocessor(Preprocessor):
    def __init__(self, history_length=1):
        self.history_length = history_length       # Maximum number of recent frames needed for the sequence

    def process_state_for_network(self, current_frame, recent_frames,recent_end_frames):  # History needed to decide current action
        state = [current_frame]                                                           # Recent_frames and recent_end_frames(deque)
        index = len(recent_frames) - 1
        # Check if the recent frames leak over different episodes
        for checkval in range(0, self.history_length - 1):
            current_index = index - checkval
            current_end_frame = recent_end_frames[current_index - 1] if current_index - 1 >= 0 else False
            if current_index < 0 or current_end_frame:
                break                                          # Don't add the previous frame, it is from a different episode
            state.insert(0, recent_frames[current_index]) 
        while len(state) < self.history_length:
            state.insert(0, np.zeros((state[0]).shape))
        return state

    def reset(self):   # Central reset implemented for the dqn agent and ignored within this class
        pass

    def get_config(self):
        return {'history_length': self.history_length}

        
class AtariPreprocessor(Preprocessor):
    def __init__(self, new_size):
        self.new_size = new_size

    def process_state_for_memory(self, state):     # Scaled, converted to greyscale and stored as uint8
        img = Image.fromarray(state)
        img = img.resize(self.new_size).convert('L')
        processed_state = np.array(img)
        return processed_state.astype('uint8')  

    def process_state_for_network(self, state):   # Scaled, converted to greyscale and stored as float32
        img = Image.fromarray(state)
        img = img.resize(self.new_size).convert('L')
        processed_state = np.array(img)
        return processed_state.astype('float32')

    def process_batch(self, samples):              #The batches from replay memory converted to float32
        processed_batch = samples.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):              # Reward not clipped
        #return np.clip(reward, -1., 1.)
        return reward
        

class PreprocessorSequence(Preprocessor):
    def __init__(self, preprocessors):
        pass

