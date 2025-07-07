import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

from dataloader import FolderDataset, preprocess_data, load_full_dataframe
from model import RFStudent

# GPU 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 설정값
EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-3
K_FOLDS = 4
TEST_RATIO = 0.2
DATA_PATH = "/home/mhlee/Documents/doo_rae_mee/배변배뇨/data/rf/converted_csv"
EXCEL_PATH = "/home/mhlee/Documents/doo_rae_mee/배변배뇨/data/Z_volume_selection.xlsx"
SAVE_DIR = "./kfold_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# 전체 데이터 로드
df = load_full_dataframe(EXCEL_PATH)
dataset = FolderDataset(df, DATA_PATH, transform=preprocess_data)

# 🔹 Train+Val / Test 분할
indices = list(range(len(dataset)))
trainval_idx, test_idx = train_test_split(indices, test_size=TEST_RATIO, random_state=42)
test_dataset = Subset(dataset, test_idx)
torch.save(test_idx, os.path.join(SAVE_DIR, "test_idx.pt"))  # test.py에서 사용하기 위해 저장

# 🔸 K-Fold on trainval set
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_idx)):
    print(f"\n===== Fold {fold + 1} / {K_FOLDS} =====")

    # 인덱스 매핑
    train_real_idx = [trainval_idx[i] for i in train_idx]
    val_real_idx = [trainval_idx[i] for i in val_idx]

    train_dataset = Subset(dataset, train_real_idx)
    val_dataset = Subset(dataset, val_real_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 모델, 손실함수, 옵티마이저
    model = RFStudent().to(device)
    criterion = nn.HuberLoss(delta=5.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val_mae = float("inf")
    train_losses, val_losses = [], []

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

        # Validation
        model.eval()
        total_val_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                rf_h = [batch['horizon'][:, i, :].unsqueeze(1).to(device) for i in range(4)]
                rf_s = [batch['sagittal'][:, i, :].unsqueeze(1).to(device) for i in range(3)]
                targets = batch['volume_gt'].to(device)

                preds, _, _ = model(rf_h, rf_s)
                loss = criterion(preds, targets)
                total_val_loss += loss.item()

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        r2 = r2_score(all_targets, all_preds)

        print(f"[Fold {fold + 1} | Epoch {epoch+1:02d}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

        # MAE 기준으로 best 모델 저장
        if mae < best_val_mae:
            best_val_mae = mae
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_student_fold{fold + 1}.pt"))
            print("Best model saved based on MAE.")

    # 그래프 저장
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Fold {fold + 1} Train/Val Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"loss_curve_fold{fold + 1}.png"))
    plt.close()