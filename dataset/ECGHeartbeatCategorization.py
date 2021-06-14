from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate, DataLoader

from transforms.transform import O_75, O_25, MEDIAN, TIME_NORMALIZER


class ECGHeartbeatCategorization(Dataset):
    def __init__(self, path_to_csv, transforms=None):
        self.transforms = transforms

        self.path_to_csv = path_to_csv
        numpy_data = pd.read_csv(self.path_to_csv).to_numpy()
        self.data, self.target = numpy_data[:, :-1], numpy_data[:, -1]
        self.target = self.target.astype(np.int)
        self.num_classes = self.target.max() + 1
        self.min_len = 15

    def get_series_from_data(self, data, target, idx):
        return data[idx, :], target[idx], idx

    # Convert pandas series to dict
    def conver_to_dict(self, series_from_data, eps= 1e-6):
        ampl, target, id = series_from_data
        ampl = (ampl[ampl > eps] * 2 - 1) * (O_75 - O_25) + MEDIAN
        if len(ampl) < self.min_len:
            print('value was with small len')
            return self.__getitem__(np.random.randint(0, self.__len__(), 1)[0])
        target = np.ones(len(ampl), dtype=np.int) * target
        one_hot_target = np.zeros((*ampl.shape, self.num_classes), dtype=np.float32)
        one_hot_target[np.arange(target.shape[0]), target] = 1

        assert len(target) == len(ampl)
        time = np.expand_dims(np.arange(0, len(ampl)) * TIME_NORMALIZER, 1)
        ampl = np.expand_dims(ampl, 1)

        return {'id': id, 'time': time, 'ampl': ampl, 'target': one_hot_target, 'time_unormalized': time, 'ampl_unormalized': ampl}

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        data_by_id = self.get_series_from_data(self.data, self.target, index)
        data_by_id = self.conver_to_dict(data_by_id)
        if self.transforms is not None:
            data_by_id = self.transforms(data_by_id)
        return data_by_id

    # Collate_fn for output batch, also adds "mask" key
    def collate_fn(self, batched_data):

        len_data = [len(x['time']) for x in batched_data]
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

