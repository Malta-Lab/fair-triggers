from pathlib import Path
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pandas as pd

class BiasInBiosDataset(Dataset):
    def __init__(self, path, tokenizer, max_len : int=256, mode : str='train'):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self.read_data(self.path)
        self.labels = self.count_labels(self.data)
        self.data = train_test_split(self.data, test_size=0.2, random_state=42)
        self.data = self.data[0] if mode == 'train' else self.data[1]
    
    def read_data(self, path):
        files = [f for f in os.listdir(path) if 'pkl' in f]
        data = []
        for f in files:
            file_data = pd.read_pickle(self.path/f)
            for instance in file_data:
                new_instance = {'bio': instance['raw'], 
                                'title': instance['title'], 
                                'gender': instance['gender'], 
                                'name': instance['name']}
                data.append(new_instance)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx].copy()
        instance['bio'] = self.tokenizer(str(instance['bio']),
                                        max_length=self.max_len, 
                                        return_tensors='pt',
                                        padding='max_length', 
                                        truncation=True,)
        instance['title'] = self.get_label_class(instance['title'])
        return instance

    def count_labels(self, data):
        """Count number of labels in the dataset"""
        labels = []
        for instance in data:
            if instance['title'] not in labels:
                labels.append(instance['title'])
        return labels

    def get_label_class(self, label):
        """Get label number from label name"""
        return self.labels.index(label)