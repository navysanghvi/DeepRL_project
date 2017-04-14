import json
import numpy as np
from keras.callbacks import Callback,CallbackList 

class Callback_base(Callback):
    def _set_env(self, env):
        self.env = env

    def on_episode_begin(self, episode, logs={}):
        pass

    def on_episode_end(self, episode, logs={}):
        pass

    def on_step_begin(self, step, logs={}):
        pass

    def on_step_end(self, step, logs={}):
        pass
        
class Loop_Callbacks(CallbackList):
    def _set_env(self, env):
        for callback in self.callbacks:
                callback._set_env(env)

    def on_episode_begin(self, episode, logs={}):
        for callback in self.callbacks:
                callback.on_episode_begin(episode, logs=logs)
            
    def on_episode_end(self, episode, logs={}):
        for callback in self.callbacks:    
                callback.on_episode_end(episode, logs=logs)
            
    def on_step_begin(self, step, logs={}):
        for callback in self.callbacks:
                callback.on_step_begin(step, logs=logs)
            
    def on_step_end(self, step, logs={}):
        for callback in self.callbacks:
                callback.on_step_end(step, logs=logs)
               
class Print_progress(Callback_base):      # Methods called during fit to display training progress 
    def __init__(self,log_interval):
        self.episode = 0
        self.step=0
        self.log_interval=log_interval

    def on_train_begin(self, logs):
        print('Training for {} steps :'.format(self.params['num_steps']))
    
    def on_train_end(self, logs):
        print('Training Completed or Stopped')

    def on_episode_end(self, episode, logs):
        self.episode+=1
        
    def on_step_begin(self, step, logs):
        self.step +=1
        if(self.step % self.log_interval == 0):
            to_show = 'Episode {0},steps: {1}'
            var = [self.episode,self.step]
            print(to_show.format(*var))

class Save_weights(Callback_base):         # Methods called during fit to periodically save model weights
    def __init__(self, filepath, interval):
        super(Save_weights, self).__init__()
        self.filepath = filepath
        self.interval = interval
        self.total_steps = 0

    def on_step_end(self, step, logs={}):
        self.total_steps += 1
        if self.total_steps % self.interval != 0:
            return
        filepath = self.filepath.format(step=self.total_steps, **logs)
        self.model.save_weights(filepath, overwrite=True)

class Print_test_progress(Callback_base):           # Methods called during evaluate to display test progress
    def on_train_begin(self, logs):
        print('Testing for {} episodes :'.format(self.params['num_episodes']))

    def on_episode_end(self, episode, logs):
        to_show = 'Episode {0}: reward: {1:.3f}'
        var = [episode + 1,logs['episode_reward'],]
        print(to_show.format(*var))
        

class Log_File(Callback_base):
    def __init__(self, filepath, interval=None):
        self.filepath = filepath
        self.interval = interval
        self.metrics = {}
        self.data = {}

    def on_train_begin(self, logs):  # Stores metrics generated from update policy in json file
        self.metrics_names = self.model.metrics_names

    def on_train_end(self, logs):
        self.save_data()

    def on_episode_begin(self, episode, logs):
        self.metrics[episode] = []
        
    def on_episode_end(self, episode, logs):
        metrics = self.metrics[episode]
        if np.isnan(metrics).all():
            mean_metrics = np.array([np.nan for _ in self.metrics_names])
        else:
            mean_metrics = np.nanmean(metrics, axis=0)
        data = list(zip(self.metrics_names, mean_metrics))
        data += list(logs.items())
        data += [('episode', episode)]
        for key, value in data:
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)
        if self.interval is not None and episode % self.interval == 0:
            self.save_data()                                        
        del self.metrics[episode]

    def on_step_end(self, step, logs):
        self.metrics[logs['episode']].append(logs['metrics'])

    def save_data(self):
        if len(self.data.keys()) == 0:
            return
        sorted_indexes = np.argsort(self.data['episode'])
        sorted_data = {}
        for key, values in self.data.items():
            assert len(self.data[key]) == len(sorted_indexes)
            sorted_data[key] = np.array([self.data[key][i] for i in sorted_indexes]).tolist()
        with open(self.filepath, 'w') as f:
            json.dump(sorted_data, f)



