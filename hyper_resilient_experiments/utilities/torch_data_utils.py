from torch.utils.data import Dataset
import torch


class NP_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        selected_x = torch.from_numpy(self.x[index])
        # selected_y = torch.from_numpy(self.y[index])
        selected_y = float(self.y[index])
        return selected_x, selected_y

    def __len__(self):
        return len(self.x)
