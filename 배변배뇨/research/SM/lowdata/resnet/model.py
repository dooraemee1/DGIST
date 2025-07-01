import torch
import torch.nn as nn
import os

# GPU 설정 (선택 사항)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, bottleneck_channels, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(bottleneck_channels)

        self.conv2 = nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(bottleneck_channels)

        self.conv3 = nn.Conv1d(bottleneck_channels, bottleneck_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(bottleneck_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNet1DEncoder(nn.Module):
    def __init__(self, in_channels=1, layers=[3, 4, 6, 3], out_dim=512):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * Bottleneck1D.expansion, out_dim)

    def _make_layer(self, bottleneck_channels, blocks, stride=1):
        downsample = None
        out_channels = bottleneck_channels * Bottleneck1D.expansion
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        layers = [Bottleneck1D(self.inplanes, bottleneck_channels, stride, downsample)]
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(Bottleneck1D(self.inplanes, bottleneck_channels))

        return nn.Sequential(*layers)

    def forward(self, x):  # [B, 1, H]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)  # [B, 256, ~]
        x = self.layer2(x)  # [B, 512, ~]
        x = self.layer3(x)  # [B, 1024, ~]
        x = self.layer4(x)  # [B, 2048, ~]

        x = self.avgpool(x).squeeze(-1)  # [B, 2048]
        x = self.fc(x)  # [B, out_dim]
        return x


class RFStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.rf_encoder = ResNet1DEncoder(in_channels=1, layers=[3, 4, 6, 3], out_dim=512)  # ResNet-50

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
        rf_h_list = [horizon_tensor[:, i, :].unsqueeze(1) for i in range(4)]  # [B, 1, H] × 4
        rf_s_list = [sagittal_tensor[:, i, :].unsqueeze(1) for i in range(3)]  # [B, 1, H] × 3

        h_feats = [self.rf_encoder(rf) for rf in rf_h_list]  # [B, 512] × 4
        s_feats = [self.rf_encoder(rf) for rf in rf_s_list]  # [B, 512] × 3

        h_concat = torch.cat(h_feats, dim=1)  # [B, 2048]
        s_concat = torch.cat(s_feats, dim=1)  # [B, 1536]

        f_H = self.h_projector(h_concat)  # [B, 2560]
        f_S = self.s_projector(s_concat)  # [B, 2560]

        combined = torch.cat([f_H, f_S], dim=1)  # [B, 5120]
        volume = self.head(combined).squeeze(1)  # [B]

        return volume, f_H, f_S