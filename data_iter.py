import numpy as np
import torch
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
import logging 

class Real_Dataset(Dataset):
    def __init__(self, filepath, temperature = 0.1):
        data = np.load(filepath)
        dlen = len(data)
        filename = '.'.join(filepath.split('.')[:-1])
        score = np.array([1.0] * dlen)
        try: score = np.load(f"{filename}_score.npy", allow_pickle = True)
        except: logging.error(f'[TM] No score file in {filename}_score.npy')
        score = F.softmax(torch.tensor(score/temperature, dtype = torch.float32), dim = 0)
        new_data = []
        for _ in range(dlen):
            alt_indx = np.random.choice(dlen, p = score.numpy())
            new_data.append(data[alt_indx])
        self.data = np.array(new_data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).long()

class Dis_Dataset(Dataset):
    def __init__(self, positive_filepath, negative_filepath, temperature = 1.0):
        pos_data = np.load(positive_filepath, allow_pickle = True)
        neg_data = np.load(negative_filepath, allow_pickle = True)
        filename = '.'.join(positive_filepath.split('.')[:-1])
        plen = len(pos_data)
        score = np.array([1.0] * plen)
        try: score = np.load(f"{filename}_score.npy", allow_pickle = True)
        except: logging.error(f'[TM] No score file in {filename}_score.npy')
        score = F.softmax(torch.tensor(score/temperature, dtype = torch.float32), dim = 0)
        pos_label = np.array([1 for _ in pos_data])
        neg_label = np.array([0 for _ in neg_data])
        new_pos_data = []
        for _ in range(plen):
            alt_indx = np.random.choice(plen, p = score.numpy())
            new_pos_data.append(pos_data[alt_indx])
        pos_data = np.array(new_pos_data)
        self.data = np.concatenate([pos_data, neg_data])
        self.label = np.concatenate([pos_label, neg_label])
        self.ps, self.ns = pos_data.shape[0], neg_data.shape[0]

    def pos_neg_length(self):
        return self.ps, self.nsWe

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx]).long()
        label = torch.nn.init.constant_(torch.zeros(1), int(self.label[idx])).long()
        return {"data": data, "label": label}


def real_data_loader(filepath, batch_size, shuffle, num_workers, pin_memory):
    dataset = Real_Dataset(filepath)
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = pin_memory)

def dis_data_loader(positive_filepath, negative_filepath, batch_size, shuffle, num_workers, pin_memory):
    dataset = Dis_Dataset(positive_filepath, negative_filepath)
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = pin_memory)