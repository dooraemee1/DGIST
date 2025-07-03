import torch
import torch.nn as nn
import os

class RFSingleEncoder(nn.Module):
    """
    1D RF 신호 하나를 인코딩하는 모듈.
    과적합 방지를 위해 드롭아웃을 추가하고, 두 종류의 풀링을 사용합니다.
    """
    def __init__(self, in_channels=1, out_dim=512, pool_output_size=10, dropout_p=0.25):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),  # Conv 블록 후 드롭아웃

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),  # Conv 블록 후 드롭아웃
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(pool_output_size)
        self.max_pool = nn.AdaptiveMaxPool1d(pool_output_size)

        self.fc = nn.Linear(64 * 2 * pool_output_size, out_dim)

    def forward(self, x):
        x = self.conv_block(x)
        pooled = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        flattened = pooled.flatten(start_dim=1)
        features = self.fc(flattened)
        return features


class RFStudent(nn.Module):
    """
    과적합 방지 기능이 강화된 RF Student 모델.
    - 각 RF 채널별 독립 인코더
    - 드롭아웃 레이어 추가
    - Self-Attention 메커니즘 적용
    """
    def __init__(self, num_h_rf=4, num_s_rf=3, encoder_out_dim=512, projector_out_dim=2560, 
                 num_attn_heads=8, dropout_p=0.5):
        super().__init__()

        # 각 채널별 독립 인코더 (드롭아웃 비율 전달)
        encoder_dropout = dropout_p / 2
        self.h_encoders = nn.ModuleList([
            RFSingleEncoder(out_dim=encoder_out_dim, dropout_p=encoder_dropout) 
            for _ in range(num_h_rf)
        ])
        self.s_encoders = nn.ModuleList([
            RFSingleEncoder(out_dim=encoder_out_dim, dropout_p=encoder_dropout) 
            for _ in range(num_s_rf)
        ])

        self.h_projector = nn.Linear(encoder_out_dim * num_h_rf, projector_out_dim)
        self.s_projector = nn.Linear(encoder_out_dim * num_s_rf, projector_out_dim)

        # 프로젝션 이후 드롭아웃
        self.projection_dropout = nn.Dropout(dropout_p)

        # Self-Attention (내부에도 드롭아웃 포함)
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=projector_out_dim,
            num_heads=num_attn_heads,
            batch_first=True,
            dropout=dropout_p
        )

        # 최종 회귀 헤드
        self.head = nn.Sequential(
            nn.Linear(projector_out_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, 1)
        )

    def forward(self, rf_h_list, rf_s_list):
        h_feats = [self.h_encoders[i](rf_h_list[i]) for i in range(len(rf_h_list))]
        s_feats = [self.s_encoders[i](rf_s_list[i]) for i in range(len(rf_s_list))]

        h_concat = torch.cat(h_feats, dim=1)
        s_concat = torch.cat(s_feats, dim=1)

        f_H = self.h_projector(h_concat)
        f_S = self.s_projector(s_concat)

        f_H = self.projection_dropout(f_H)
        f_S = self.projection_dropout(f_S)

        feature_sequence = torch.cat([f_H.unsqueeze(1), f_S.unsqueeze(1)], dim=1)
        attn_output, _ = self.feature_attention(feature_sequence, feature_sequence, feature_sequence)

        combined_features = attn_output.reshape(attn_output.size(0), -1)
        volume = self.head(combined_features).squeeze(-1)

        return volume, f_H, f_S


if __name__ == '__main__':
    model = RFStudent(dropout_p=0.5)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    bs = 4
    dummy_h = [torch.randn(bs, 1, 400) for _ in range(4)]
    dummy_s = [torch.randn(bs, 1, 400) for _ in range(3)]

    vol, f_h, f_s = model(dummy_h, dummy_s)
    print("\nOutput shapes:")
    print(f"Volume: {vol.shape}")
    print(f"f_H: {f_h.shape}")
    print(f"f_S: {f_s.shape}")