import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d

def preprocess_data(data, num_select=6):
    hilberted = hilbert(data, axis=1)
    envelope = np.abs(hilberted)
    smoothed = gaussian_filter1d(envelope, sigma=1, axis=1)
    smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
    smoothed = smoothed[::10, :][:400, :]
    norm_data = (smoothed - smoothed.mean()) / smoothed.std()

    num_columns = norm_data.shape[1]
    group_size = num_columns // num_select
    selected = []

    for i in range(num_select):
        start = i * group_size
        end = start + group_size
        col = np.random.choice(range(start, end))
        selected.append(np.transpose(norm_data[:, col]))

    return np.stack(selected)  # [num_select, 400]

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

        h_data = pd.read_csv(h_file, header=None).to_numpy()[:self.max_depth, :]
        s_data = pd.read_csv(s_file, header=None).to_numpy()[:self.max_depth, :]

        if self.transform:
            h_data = self.transform(h_data, num_select=4)
            s_data = self.transform(s_data, num_select=3)

        sample = {
            'horizon': torch.tensor(h_data, dtype=torch.float32),
            'sagittal': torch.tensor(s_data, dtype=torch.float32)
        }

        if self.mode != "test":
            sample['volume_gt'] = torch.tensor(row['volume'], dtype=torch.float32)
            sample['index'] = idx  # 이미지 인덱스 대응용

        return sample