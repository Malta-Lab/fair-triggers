from pathlib import Path
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pandas as pd

class DatasetWrapper():
    def __init__(self, dataset_name, path, tokenizer, max_len : int=256):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.path = Path(path)
        
    def _get_dataset(self):
        if self.dataset_name == 'bias-in-bios':
            data = self._read_bias_in_bios()
            train, valid = train_test_split(data, test_size=0.2, random_state=42)
            return BiasInBios(train, self.tokenizer, self.max_len), BiasInBios(valid, self.tokenizer, self.max_len)
        else:
            raise ValueError(f'{self.dataset_name} is not a valid dataset name')

    def _read_bias_in_bios(self):
        files = [f for f in os.listdir(self.path) if 'pkl' in f]
        data = []
        for f in files:
            file_data = pd.read_pickle(self.path/f)
            for instance in file_data:
                new_instance = {'bio': instance['raw'][instance['start_pos']:], 
                                'title': instance['title'], 
                                'gender': instance['gender'], 
                                'name': instance['name']}
                data.append(new_instance)
        return data


class BiasInBios(Dataset):
    def __init__(self, data, tokenizer, max_len : int=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data
        self.labels = self.count_labels(self.data)

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


class BiasInBiosDataset(Dataset):
    def __init__(self, path, tokenizer, max_len : int=256, mode : str='train'):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self.read_data()
        self.labels = self.count_labels(self.data)
        self.data = train_test_split(self.data, test_size=0.2, random_state=42)
        self.data = self.data[0] if mode == 'train' else self.data[1]
    
    def read_data(self):
        files = [f for f in os.listdir(self.path) if 'pkl' in f]
        data = []
        for f in files:
            file_data = pd.read_pickle(self.path/f)
            for instance in file_data:
                new_instance = {'bio': instance['raw'][instance['start_pos']:], 
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