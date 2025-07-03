import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d

def load_full_dataframe(excel_path):
    return pd.read_excel(excel_path)

class PreprocessTransform:
    """
    RF 데이터 전처리를 위한 클래스.
    과적합 방지를 위한 데이터 증강 기능을 포함합니다.
    """
    def __init__(self, num_select, sigma=1, downsample_rate=10, max_len=400, seed_offset=0,
                 augment=False, noise_level=0.01, max_shift=10):
        self.num_select = num_select
        self.sigma = sigma
        self.downsample_rate = downsample_rate
        self.max_len = max_len
        self.seed_offset = seed_offset
        self.augment = augment
        self.noise_level = noise_level
        self.max_shift = max_shift

    def __call__(self, data, seed):
        rng = np.random.RandomState(seed + self.seed_offset)

        envelope = np.abs(hilbert(data, axis=1))
        smoothed = gaussian_filter1d(envelope, sigma=self.sigma, axis=1)
        smoothed = smoothed[::self.downsample_rate, :][:self.max_len, :]

        mean, std = smoothed.mean(), smoothed.std()
        norm_data = (smoothed - mean) / (std + 1e-8)

        if self.augment:
            noise = rng.randn(*norm_data.shape) * self.noise_level
            norm_data += noise
            shift = rng.randint(-self.max_shift, self.max_shift + 1)
            norm_data = np.roll(norm_data, shift, axis=0)

        num_columns = norm_data.shape[1]
        group_size = max(1, num_columns // self.num_select)
        selected = []
        for i in range(self.num_select):
            start = i * group_size
            end = start + group_size if i < self.num_select - 1 else num_columns
            col_idx = rng.choice(range(start, end))
            selected.append(norm_data[:, col_idx])

        return np.stack(selected)

class FolderDataset(Dataset):
    """
    효율성을 개선한 데이터셋 클래스.
    파일 경로는 생성 시점에 미리 모두 찾아 저장합니다.
    """
    def __init__(self, dataframe, folder_path, indices, transform_h=None, transform_s=None, max_depth=4000, mode="train"):
        self.database = dataframe.iloc[indices].reset_index()  # 원본 인덱스를 'index' 컬럼으로 유지
        self.transform_h = transform_h
        self.transform_s = transform_s
        self.max_depth = max_depth
        self.mode = mode

        h_paths, s_paths = [], []
        for _, row in self.database.iterrows():
            h_path = glob.glob(os.path.join(folder_path, row['Upright_H'] + '*.csv'))[0]
            s_path = glob.glob(os.path.join(folder_path, row['Upright_S'] + '*.csv'))[0]
            h_paths.append(h_path)
            s_paths.append(s_path)

        self.database['h_path'] = h_paths
        self.database['s_path'] = s_paths

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        row = self.database.iloc[idx]
        h_file, s_file = row['h_path'], row['s_path']

        h_data = np.loadtxt(h_file, delimiter=',')[:self.max_depth, :]
        s_data = np.loadtxt(s_file, delimiter=',')[:self.max_depth, :]

        original_index = row['index']

        if self.transform_h:
            h_data = self.transform_h(h_data, seed=original_index)
        if self.transform_s:
            s_data = self.transform_s(s_data, seed=original_index)

        sample = {
            'horizon': torch.tensor(h_data, dtype=torch.float32),
            'sagittal': torch.tensor(s_data, dtype=torch.float32)
        }

        if self.mode != "test":
            sample['volume_gt'] = torch.tensor(row['volume'], dtype=torch.float32)
            sample['index'] = original_index

        return sample