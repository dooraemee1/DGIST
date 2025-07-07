import os
import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from dataloader import FolderDataset, PreprocessTransform, load_full_dataframe
from model import RFStudent

# ===== 시드 고정 =====
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ===== 설정 =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
USE_LOG1P = False
SAVE_DIR = "./results"
DATA_PATH = "/home/mhlee/Documents/doo_rae_mee/배변배뇨/data/rf/converted_csv"
EXCEL_PATH = "/home/mhlee/Documents/doo_rae_mee/배변배뇨/data/Z_volume_selection.csv"

# ===== 데이터 로드 =====
df = load_full_dataframe(EXCEL_PATH)
df['original_index'] = df.index  # ⚠️ 꼭 있어야 함

# test index 로드
test_idx_path = os.path.join(SAVE_DIR, "test_idx.pt")
if not os.path.exists(test_idx_path):
    print("test_idx.pt 파일을 먼저 생성해야 합니다. train.py를 실행하세요.")
    exit()

test_idx = torch.load(test_idx_path)

# Transform 정의
transform_h = PreprocessTransform(num_select=4)
transform_s = PreprocessTransform(num_select=3)

# 전체 dataset → Subset으로 테스트셋 구성
full_dataset = FolderDataset(df, DATA_PATH, transform_h, transform_s)
test_dataset = Subset(full_dataset, test_idx)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"테스트셋 크기: {len(test_dataset)}개")

# ===== 모델 로드 =====
model_path = os.path.join(SAVE_DIR, "best_model.pt")
model = RFStudent().to(DEVICE)

try:
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    exit()

model.eval()

# ===== 추론 =====
all_preds, all_targets = [], []

with torch.no_grad():
    for batch in test_loader:
        rf_h = [batch['horizon'][:, i, :].unsqueeze(1).to(DEVICE) for i in range(4)]
        rf_s = [batch['sagittal'][:, i, :].unsqueeze(1).to(DEVICE) for i in range(3)]
        targets = batch['volume_gt'].to(DEVICE)

        preds, _, _ = model(rf_h, rf_s)
        preds = preds.squeeze(-1)

        if USE_LOG1P:
            preds = torch.expm1(preds)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# ===== 회귀 평가 =====
mae = mean_absolute_error(all_targets, all_preds)
rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
r2 = r2_score(all_targets, all_preds)
within_50 = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)) <= 50) * 100

print("\n===== 회귀 기반 평가 =====")
print(f"MAE        : {mae:.2f}")
print(f"RMSE       : {rmse:.2f}")
print(f"R²         : {r2:.4f}")
print(f"±50 정확도 : {within_50:.2f}%")

# ===== 산점도 시각화 =====
plt.figure(figsize=(6, 6))
plt.scatter(all_targets, all_preds, alpha=0.6, label='Predictions')
plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'k--', label='Ideal (y=x)')
plt.plot([min(all_targets), max(all_targets)], [min(all_targets)+50, max(all_targets)+50], 'r--', alpha=0.7, label='±50 Range')
plt.plot([min(all_targets), max(all_targets)], [min(all_targets)-50, max(all_targets)-50], 'r--', alpha=0.7)
plt.xlabel("Actual Volume")
plt.ylabel("Predicted Volume")
plt.title("Prediction vs Actual")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "scatter_testset.png"))
plt.close()

# ===== 다중 클래스 변환 =====
bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, np.inf]
labels = list(range(len(bins) - 1))

true_cls = pd.cut(all_targets, bins=bins, labels=labels).astype(int)
pred_cls = pd.cut(all_preds, bins=bins, labels=labels).astype(int)

# ===== F1-score 계산 =====
f1_macro = f1_score(true_cls, pred_cls, average='macro')
f1_weighted = f1_score(true_cls, pred_cls, average='weighted')

print("\n===== 다중 클래스 분류 평가 (50단위 binning) =====")
print(f"F1-score (macro)    : {f1_macro:.4f}")
print(f"F1-score (weighted) : {f1_weighted:.4f}")

# ===== Confusion Matrix =====
cm = confusion_matrix(true_cls, pred_cls, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"{bins[i]}~{bins[i+1]}" for i in range(len(labels))])
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix (Volume Bins)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix_bins.png"))
plt.close()