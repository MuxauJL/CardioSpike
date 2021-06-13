from torch.utils.data import Dataset
import pandas as pd
import torch


class CardioSpikeDataset(Dataset):
    def __init__(self, path_to_csv):
        self.path_to_csv = path_to_csv
        self.data = pd.read_csv(path_to_csv)

        counter = 0
        self.identifiers = set()
        self.idx_to_id = {}
        for id in self.data.id:
            if id not in self.identifiers:
                self.identifiers.add(id)
                self.idx_to_id[counter] = id
                counter += 1

    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self, index):
        data_by_id = self.data[self.data.id == self.idx_to_id[index]]
        x = torch.tensor(data_by_id.x.to_numpy())
        y = torch.tensor(data_by_id.y.to_numpy())
        # t = torch.tensor(data_by_id.time.to_numpy())
        # return torch.stack((t, x), 1), y
        t = data_by_id.time.to_numpy()
        time_diff = torch.zeros(len(t))
        for i in range(1, len(t)):
            time_diff[i] = t[i] - t[i - 1]
        return torch.stack((time_diff, x), 1), y
        # return x, y

def collate_fn(batched_data):
    max_len = max([len(y) for x, y in batched_data])
    batch_size = len(batched_data)
    batched_x = torch.zeros(max_len, batch_size, 2, dtype=torch.float32)
    batched_y = torch.full((max_len, batch_size), -1, dtype=torch.float32)
    for batch_idx, (x, y) in enumerate(batched_data):
        assert len(x) == len(y)
        batched_x[:len(x), batch_idx, :] = x
        batched_y[:len(y), batch_idx] = y

    return batched_x, batched_y

def conv_collate_fn(batched_data):
    max_len = max([len(y) for x, y in batched_data])
    batch_size = len(batched_data)
    batched_x = torch.zeros(batch_size, 2, max_len, dtype=torch.float32)
    batched_y = torch.full((max_len, batch_size), -1, dtype=torch.float32)
    for batch_idx, (x, y) in enumerate(batched_data):
        assert len(x) == len(y)
        batched_x[batch_idx, :, :len(x)] = x.T
        batched_y[:len(y), batch_idx] = y

    return batched_x, batched_y