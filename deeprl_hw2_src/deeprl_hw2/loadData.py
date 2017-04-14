import json
from random import shuffle

class DataSet():
    def __init__(self, path):
        self.path = path

    def load(self, trainSplit):
        fp = open(self.path, "r")
        data = json.load(fp)
        self.list_data = []
        for key, value in data.iteritems():
            self.list_data.append((value['file_name'].rsplit('_',1)[0] + '.jpg', value['sent'], value['bbox'], value['sent_repr']))

        shuffle(self.list_data)
        train_len = int(trainSplit*len(self.list_data))
        self.train_data = self.list_data[:train_len]
        self.test_data = self.list_data[train_len:]

    def permute(self):
        shuffle(self.train_data)




