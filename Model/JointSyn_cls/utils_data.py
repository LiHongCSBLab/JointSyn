import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def split(data, repeat):
    kf = KFold(n_splits=5, random_state=repeat, shuffle=True)
    seed = 42
    i = 1
    for train_index , test_index in kf.split(data):
        data_train_val = data.iloc[train_index]
        data_test = data.iloc[test_index]
        data_train, data_val = train_test_split(data_train_val, test_size=0.25, random_state=seed)
        all_index = np.concatenate((data_train.index.values, data_val.index.values, data_test.index.values), axis=0)
        all_index = np.unique(all_index)

        data_train = data_train.reset_index(drop=True)
        data_val = data_val.reset_index(drop=True)
        data_test = data_test.reset_index(drop=True)
        path_train = './data/repeat'+str(repeat)+'_fold'+str(i)+'_train.csv'
        path_val = './data/repeat'+str(repeat)+'_fold'+str(i)+'_val.csv'
        path_test = './data/repeat'+str(repeat)+'_fold'+str(i)+'_test.csv'
        i += 1
        print(path_train)
        data_train.to_csv(path_train)
        data_val.to_csv(path_val)
        data_test.to_csv(path_test)

        
class load_data(Dataset):
    def __init__(self, csv_path, transforms=None):
        data = pd.read_csv(csv_path)
        score_name = 'label'
        score_name_drop = 'synergy_loewe'
        self.labels = data[score_name]
        data = data.drop(columns=score_name)
        data = data.drop(columns=score_name_drop)
        data = data.drop(columns='Unnamed: 0')
        data = np.array(data)
        self.inputs = data
        self.transforms = transforms

    def __getitem__(self, index):
        labels = self.labels[index]
        inputs_np = self.inputs[index]
        inputs_tensor = torch.from_numpy(inputs_np).float()
        return (inputs_tensor, labels)
    
    def __len__(self):
        return len(self.labels.index)