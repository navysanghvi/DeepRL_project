import json
from random import shuffle

class DataSet():
    def __init__(self, path):
        self.path = path

    def load(self):
        fp = open(self.path, "r")
        data = json.load(fp)
        list_data = []
        for key, value in data.iteritems():
            list_data.append((value['file_name'].rsplit('_',1)[0] + '.jpg', value['sent'], value['bbox']))
        self.list_data = list_data

    def permute(self):
        shuffle(self.list_data)


