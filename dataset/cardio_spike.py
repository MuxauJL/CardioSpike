import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate, DataLoader


class CardioSpikeDataset(Dataset):
    def __init__(self, path_to_csv, transforms=None, id_column='id', time_column='time', x_column='x', target_column='y', min_len=None):
        self.transforms = transforms
        self.target_column = target_column
        self.x_column = x_column
        self.time_column = time_column
        self.id_column = id_column

        self.path_to_csv = path_to_csv
        self.data = pd.read_csv(path_to_csv)

        self.ids = self.get_ids(self.data)
        self.min_len = min_len

    # Extract pandas rows by same id and sort it by time
    def get_series_from_data(self, data, idx):
        return data[data[self.id_column] == idx].sort_values(by=self.time_column).reset_index().drop("index", axis=1)

    def get_ids(self, data):
        return list(data[self.id_column].unique())

    def get_dataframe(self):
        return self.data.copy()

    # Convert pandas series to dict
    def conver_to_dict(self, series_from_data):
        id = series_from_data[self.id_column]
        if self.target_column:
            target = np.expand_dims(series_from_data[self.target_column].to_numpy(), 1)
        else:
            target = np.zeros_like(np.expand_dims(series_from_data[self.x_column].to_numpy(), 1))
        assert np.all(id == list(id)[0]) or self.min_len
        id = int(id[0])
        time = np.expand_dims(series_from_data[self.time_column].to_numpy(), 1)
        ampl = np.expand_dims(series_from_data[self.x_column].to_numpy(), 1)

        return {'id': id, 'time': time, 'ampl': ampl, 'target': target, 'time_unormalized': time, 'ampl_unormalized': ampl}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        mapped_index = self.ids[index]
        data_by_id = self.get_series_from_data(self.data, mapped_index)
        if self.min_len:
            while len(data_by_id) < self.min_len:
                additional_data = self.get_series_from_data(self.data, random.randint(0, len(self))).copy()
                additional_data.time += data_by_id.time.max() + 624
                data_by_id = data_by_id.append(additional_data, ignore_index=True)
        data_by_id = self.conver_to_dict(data_by_id)
        if self.transforms is not None:
            data_by_id = self.transforms(data_by_id)
        return data_by_id

    # Collate_fn for output batch, also adds "mask" key
    def collate_fn(self, batched_data):

        len_data = [len(x[self.time_column]) for x in batched_data]
        keys = tuple(batched_data[0].keys())
        group_by_key = defaultdict(list)
        for el in batched_data:
            for key in keys:
                group_by_key[key].append(torch.tensor(el[key]).float())
        result = {}
        mask = [torch.ones(len_el) for len_el in len_data]
        group_by_key['mask'] = mask
        for key, batch in group_by_key.items():
            if hasattr(batch[0], '__len__') and len(batch[0].size()) > 0:
                result[key] = pad_sequence(batch, batch_first=True)
            else:
                result[key] = default_collate(batch)
        result['mask_bool'] = result['mask'] > 0.5
        # id - B, time - BxTx1, ampl - BxTx1,  target - BxTx1, mask - BxT, mask_bool - BxT
        return result

if __name__ == '__main__':
    dataset = CardioSpikeDataset("/home/malchul/work/CadrioSpike/data/train.csv")
    print(dataset[12])
    loader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=2, collate_fn=dataset.collate_fn)
    for data in loader:
        for k,v in data.items():
            print(k, v.shape)
        print(data['time'] * (1 - data['mask']))
