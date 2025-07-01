import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_full_dataframe(excel_path):
    return pd.read_excel(excel_path)

class FolderDataset(Dataset):
    def __init__(self, dataframe, folder_path, indices=None, transform=None, max_depth=4000, mode="train"):
        if indices is not None:
            self.database = dataframe.iloc[indices].reset_index(drop=True)
        else:
            self.database = dataframe.reset_index(drop=True)
        self.file_id = self.database['id'].values
        self.folder_path = folder_path
        self.transform = transform
        self.max_depth = max_depth
        self.mode = mode

    def __len__(self):
        return len(self.file_id)

    def __getitem__(self, idx):
        sample_id = self.file_id[idx]
        row = self.database[self.database['id'] == sample_id].iloc[0]

        h_file = glob.glob(os.path.join(self.folder_path, row['Upright_H'] + '*.csv'))[0]
        s_file = glob.glob(os.path.join(self.folder_path, row['Upright_S'] + '*.csv'))[0]

        h_data = pd.read_csv(h_file, header=None).to_numpy()[:self.max_depth, :]  # shape [H, W]
        s_data = pd.read_csv(s_file, header=None).to_numpy()[:self.max_depth, :]  # shape [H, W]

        # H posture: 4 random lines
        h_indices = np.random.choice(h_data.shape[1], 4, replace=False)
        h_lines = h_data[:, h_indices]  # [H, 4]
        h_lines = h_lines.T  # [4, H]

        # S posture: 3 random lines
        s_indices = np.random.choice(s_data.shape[1], 3, replace=False)
        s_lines = s_data[:, s_indices]  # [H, 3]
        s_lines = s_lines.T  # [3, H]

        sample = {
            'horizon': torch.tensor(h_lines, dtype=torch.float32),   # [4, H]
            'sagittal': torch.tensor(s_lines, dtype=torch.float32),  # [3, H]
        }

        if self.mode != "test":
            sample['volume_gt'] = torch.tensor(row['volume'], dtype=torch.float32)
            sample['index'] = idx

        return sample
