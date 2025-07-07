import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from dataloader import FolderDataset, PreprocessTransform, load_full_dataframe
from model import RFStudent

# ===== 시드 고정 함수 =====
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ===== 환경 및 하이퍼파라미터 설정 =====
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 100
BATCH_SIZE = 16
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
DATA_PATH = "/home/mhlee/Documents/doo_rae_mee/배변배뇨/data/rf/converted_csv"
EXCEL_PATH = "/home/mhlee/Documents/doo_rae_mee/배변배뇨/data/Z_volume_selection.csv"
SAVE_DIR = "./results"
os.makedirs(SAVE_DIR, exist_ok=True)

LR = 1e-3
WEIGHT_DECAY = 1e-2
DROPOUT_P = 0.5

# ===== 데이터 로드 및 유효성 검사 =====
df_initial = load_full_dataframe(EXCEL_PATH)
print(f"로드된 초기 데이터: {len(df_initial)}개")

df_initial['volume'] = pd.to_numeric(df_initial['volume'], errors='coerce')
df_initial.dropna(subset=['volume'], inplace=True)
df_initial['original_index'] = df_initial.index

# 파일 존재 여부 확인
def check_files_exist(row):
    if pd.isna(row['Upright_H']) or pd.isna(row['Upright_S']):
        return False
    h_path = os.path.join(DATA_PATH, str(row['Upright_H']) + '*.csv')
    s_path = os.path.join(DATA_PATH, str(row['Upright_S']) + '*.csv')
    return bool(glob.glob(h_path) and glob.glob(s_path))

valid_mask = df_initial.apply(check_files_exist, axis=1)
df = df_initial[valid_mask].reset_index(drop=True)
print(f"유효성 검사 완료: {len(df)}개의 유효한 데이터 확인.")

# ===== 데이터 분할 =====
indices = list(range(len(df)))
volume_bins = pd.cut(df['volume'], bins=5, labels=False, include_lowest=True)
train_idx, val_test_idx, _, _ = train_test_split(indices, df['volume'], test_size=(VAL_RATIO + TEST_RATIO), random_state=42, stratify=volume_bins)

val_test_df = df.iloc[val_test_idx]
val_test_bins = pd.cut(val_test_df['volume'], bins=5, labels=False, include_lowest=True)
bin_counts = val_test_bins.value_counts()

if 1 in bin_counts.values:
    print("Stratification Error 방지: 최소 클래스 멤버가 1개인 그룹 발견. 해당 데이터를 증강합니다.")
    single_member_bins = bin_counts[bin_counts == 1].index
    rows_to_add = []
    for bin_val in single_member_bins:
        idx_to_duplicate = val_test_bins[val_test_bins == bin_val].index[0]
        row_df = val_test_df.loc[[idx_to_duplicate]].copy().reset_index(drop=True)
        rows_to_add.append(row_df)
        print(f"-> ID {val_test_df.loc[idx_to_duplicate]['original_index']} (volume: {val_test_df.loc[idx_to_duplicate]['volume']})를 복제하여 증강했습니다.")
    val_test_df = pd.concat([val_test_df] + rows_to_add, ignore_index=True)

final_val_test_idx = val_test_df.index.tolist()
final_val_test_y = val_test_df['volume']
final_stratify_key = pd.cut(final_val_test_y, bins=5, labels=False, include_lowest=True)
val_idx, test_idx, _, _ = train_test_split(final_val_test_idx, final_val_test_y, test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)), random_state=42, stratify=final_stratify_key)

print(f"훈련 데이터: {len(train_idx)}개 | 검증 데이터: {len(val_idx)}개 | 테스트 데이터: {len(test_idx)}개")
test_original_indices = df.loc[test_idx]['original_index'].tolist()
torch.save(test_original_indices, os.path.join(SAVE_DIR, "test_idx.pt"))

# ===== 데이터 증강 =====
augmented_train_idx = []
for idx in train_idx:
    volume = df.iloc[idx]['volume']
    augmented_train_idx.extend([idx] * 3)
    if volume <= 150 or volume >= 600:
        augmented_train_idx.extend([idx] * 3)

print(f"원본 훈련 데이터: {len(train_idx)}개 -> 증강 후: {len(augmented_train_idx)}개")

# ===== 데이터셋 및 로더 생성 =====
transform_h = PreprocessTransform(num_select=4)
transform_s = PreprocessTransform(num_select=3)
full_dataset = FolderDataset(df, DATA_PATH, transform_h=transform_h, transform_s=transform_s)
train_dataset = Subset(full_dataset, augmented_train_idx)
val_dataset = Subset(full_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ===== 모델 및 학습 루프 =====
model = RFStudent(dropout_p=DROPOUT_P).to(device)
criterion = nn.HuberLoss(delta=5.0)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best_val_mae = float("inf")
train_losses, val_losses = [], []
val_maes, val_rmses, val_r2s = [], [], []

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        rf_h = [batch['horizon'][:, i, :].unsqueeze(1).to(device) for i in range(4)]
        rf_s = [batch['sagittal'][:, i, :].unsqueeze(1).to(device) for i in range(3)]
        targets = batch['volume_gt'].to(device)

        preds, _, _ = model(rf_h, rf_s)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss, all_preds, all_targets = 0, [], []
    with torch.no_grad():
        for batch in val_loader:
            rf_h = [batch['horizon'][:, i, :].unsqueeze(1).to(device) for i in range(4)]
            rf_s = [batch['sagittal'][:, i, :].unsqueeze(1).to(device) for i in range(3)]
            targets = batch['volume_gt'].to(device)
            preds, _, _ = model(rf_h, rf_s)
            total_val_loss += criterion(preds, targets).item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)

    val_maes.append(mae)
    val_rmses.append(rmse)
    val_r2s.append(r2)

    print(f"[Epoch {epoch+1:03d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")

    if mae < best_val_mae:
        best_val_mae = mae
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
        print(f"-> Best model saved with MAE: {best_val_mae:.2f}")

# ===== 학습 곡선 시각화 =====
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Train/Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(val_maes, label="Val MAE")
plt.plot(val_rmses, label="Val RMSE")
plt.plot(val_r2s, label="Val R2")
plt.title(f"Validation Metrics (Best MAE: {best_val_mae:.2f})")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "metrics_curve.png"))
plt.close()

print("\n훈련이 완료되었습니다.")
print(f"최적 모델은 {SAVE_DIR}/best_model.pt 에 저장되었습니다.")
print(f"테스트 인덱스는 {SAVE_DIR}/test_idx.pt 에 저장되었습니다.")
print(f"학습 곡선은 {SAVE_DIR}/metrics_curve.png 에 저장되었습니다.")