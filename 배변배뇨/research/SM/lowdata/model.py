import torch
import torch.nn as nn
import os

# GPU 설정 (선택 사항)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1D RF 라인 하나 인코딩용 모듈
class RFSingleEncoder(nn.Module):
    def __init__(self, in_channels=1, out_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(10),  # → [B, 64, 10]
            nn.Flatten(),              # → [B, 640]
            nn.Linear(640, out_dim)    # → [B, 512]
        )

    def forward(self, x):  # x: [B, 1, H]
        return self.encoder(x)  # → [B, 512]

# 전체 RF Student 모델
class RFStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.rf_encoder = RFSingleEncoder()

        self.h_projector = nn.Linear(512 * 4, 2560)
        self.s_projector = nn.Linear(512 * 3, 2560)

        self.head = nn.Sequential(
            nn.Linear(5120, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, horizon_tensor, sagittal_tensor):
        """
        Args:
            horizon_tensor: [B, 4, H]
            sagittal_tensor: [B, 3, H]
        Returns:
            volume_pred: [B]
            f_H: [B, 2560]
            f_S: [B, 2560]
        """
        B = horizon_tensor.size(0)

        # [B, 4, H] → 4 tensors: [B, 1, H]
        rf_h_list = [horizon_tensor[:, i, :].unsqueeze(1) for i in range(4)]
        rf_s_list = [sagittal_tensor[:, i, :].unsqueeze(1) for i in range(3)]

        h_feats = [self.rf_encoder(rf) for rf in rf_h_list]  # [B, 512] * 4
        s_feats = [self.rf_encoder(rf) for rf in rf_s_list]  # [B, 512] * 3

        h_concat = torch.cat(h_feats, dim=1)  # [B, 2048]
        s_concat = torch.cat(s_feats, dim=1)  # [B, 1536]

        f_H = self.h_projector(h_concat)  # [B, 2560]
        f_S = self.s_projector(s_concat)  # [B, 2560]

        combined = torch.cat([f_H, f_S], dim=1)  # [B, 5120]
        volume = self.head(combined).squeeze(1)  # [B]

        return volume, f_H, f_S
