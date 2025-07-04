import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

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
K_FOLDS = 4
TEST_RATIO = 0.2
DATA_PATH = "/home/mhlee/Documents/doo_rae_mee/배변배뇨/data/rf/converted_csv"
EXCEL_PATH = "/home/mhlee/Documents/doo_rae_mee/배변배뇨/data/Z_volume_selection.xlsx"
SAVE_DIR = "./kfold_results_no_augment"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 과적합 방지 관련 하이퍼파라미터 ---
LR = 1e-4
WEIGHT_DECAY = 1e-2
DROPOUT_P = 0.5

# ===== 데이터 로드 및 분할 =====
df = load_full_dataframe(EXCEL_PATH)
indices = list(range(len(df)))
volume_bins_stratify = pd.cut(df['volume'], bins=5, labels=False, include_lowest=True)
trainval_idx, test_idx = train_test_split(indices, test_size=TEST_RATIO, random_state=42, stratify=volume_bins_stratify)
torch.save(test_idx, os.path.join(SAVE_DIR, "test_idx.pt"))

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

for fold, (train_split_idx, val_split_idx) in enumerate(kf.split(trainval_idx)):
    print(f"\n===== Fold {fold + 1} / {K_FOLDS} =====")

    train_real_idx = [trainval_idx[i] for i in train_split_idx]
    val_real_idx = [trainval_idx[i] for i in val_split_idx]

    transform_h = PreprocessTransform(num_select=4)
    transform_s = PreprocessTransform(num_select=3)

    train_dataset = FolderDataset(df, DATA_PATH, indices=train_real_idx, transform_h=transform_h, transform_s=transform_s)

    outlier_idx = [i for i in train_real_idx if df.iloc[i]['volume'] <= 150 or df.iloc[i]['volume'] >= 600]
    if outlier_idx:
        outlier_dataset = FolderDataset(df, DATA_PATH, indices=outlier_idx, transform_h=transform_h, transform_s=transform_s)
        full_train_dataset = ConcatDataset([train_dataset] + [outlier_dataset] * 4)
    else:
        full_train_dataset = train_dataset

    val_dataset = FolderDataset(df, DATA_PATH, indices=val_real_idx, transform_h=transform_h, transform_s=transform_s)

    train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = RFStudent(dropout_p=DROPOUT_P).to(device)
    criterion = nn.HuberLoss(delta=5.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_mae = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        train_iterator = tqdm(train_loader, desc=f"Fold {fold+1} | Train Epoch {epoch+1}/{EPOCHS}", leave=False)
        for i, batch in enumerate(train_iterator):
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
            train_iterator.set_postfix(loss=total_train_loss / (i + 1))

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss, all_preds, all_targets = 0, [], []

        val_iterator = tqdm(val_loader, desc=f"Fold {fold+1} | Validation", leave=False)
        with torch.no_grad():
            for batch in val_iterator:
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

        print(f"[Fold {fold+1} | Epoch {epoch+1:03d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | MAE: {mae:.2f}")

        if mae < best_val_mae:
            best_val_mae = mae
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_model_fold{fold+1}.pt"))
            print(f"-> Best model saved with MAE: {best_val_mae:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title(f"Fold {fold+1} Train/Val Loss (Best MAE: {best_val_mae:.2f})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"loss_curve_fold{fold+1}.png"))
    plt.close()