import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dataloader import FolderDataset, load_full_dataframe
from model import RFStudent
from torch.utils.data import DataLoader, Subset

# 설정
SAVE_DIR = "./kfold_results"
EXCEL_PATH = "/home/mhlee/Documents/doo_rae_mee/배변배뇨/data/Z_volume_selection.xlsx"
DATA_PATH = "/home/mhlee/Documents/doo_rae_mee/배변배뇨/data/rf/converted_csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
K_FOLDS = 4

# 데이터 로드
df = load_full_dataframe(EXCEL_PATH)
dataset = FolderDataset(df, DATA_PATH)

test_idx = torch.load(os.path.join(SAVE_DIR, "test_idx.pt"))
test_dataset = Subset(dataset, test_idx)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"테스트셋 크기: {len(test_dataset)}개")

# Fold별 테스트
for fold in range(1, K_FOLDS + 1):
    print(f"\n Fold {fold} 테스트 중...")

    model = RFStudent().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"best_student_fold{fold}.pt"), map_location=DEVICE))
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            rf_h = batch['horizon'].to(DEVICE)  # [1, 4, H]
            rf_s = batch['sagittal'].to(DEVICE)  # [1, 3, H]
            target = batch['volume_gt'].to(DEVICE)

            pred, _, _ = model(rf_h, rf_s)

            all_preds.append(pred.item())
            all_targets.append(target.item())

    # 평가 지표 출력
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    within_50 = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)) <= 50) * 100

    print(f"Fold {fold} 결과:")
    print(f" - MAE: {mae:.2f}")
    print(f" - RMSE: {rmse:.2f}")
    print(f" - R²: {r2:.4f}")
    print(f" - ±50 정확도: {within_50:.2f}%")

    # 산점도 저장
    plt.figure(figsize=(6, 6))
    plt.scatter(all_targets, all_preds, alpha=0.6, label='Predictions')
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'k--', label='Ideal (y=x)')
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets)+50, max(all_targets)+50], 'r--', label='±50 Range')
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets)-50, max(all_targets)-50], 'r--')
    plt.xlabel("Actual Volume")
    plt.ylabel("Predicted Volume")
    plt.title(f"Test Prediction Fold {fold}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"test_scatter_fold{fold}.png"))
    plt.close()