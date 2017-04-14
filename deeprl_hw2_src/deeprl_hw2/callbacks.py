import json
import numpy as np
from keras.callbacks import Callback,CallbackList 

class Save_weights(Callback):         # Methods called during fit to periodically save model weights
    def __init__(self, filepath, interval):
        super(Save_weights, self).__init__()
        self.filepath = filepath
        self.interval = interval
        self.episodes = 0

    def on_episode_end(self, episode, logs={}):
        self.episodes += 1
        if self.episodes % self.interval != 0:
            return
        filepath = self.filepath.format(episode=self.episodes, **logs)
        self.model.save_weights(filepath, overwrite=True)

class Log_File(Callback):
    def __init__(self, filepath, interval=None):
        self.filepath = filepath
        self.interval = interval
        self.metrics = {}
        self.data = {}
        
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



