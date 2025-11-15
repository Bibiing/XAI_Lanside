import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import json
import os
import data_prepare as dp
from model import LSM_cnn
from sklearn.metrics import accuracy_score
import warnings
import matplotlib
matplotlib.use('Agg')  # Gunakan non-GUI backend untuk simpan plot

# Suppress FutureWarnings
warnings.simplefilter("ignore", category=FutureWarning)

# ================== KONFIGURASI ==================
FEATURE_NAMES = [
    "aspect", "elevasi", "hujan", "jalan", "landuse", "litologi", "NDVI",
    "plan", "prof", "Slope", "SPI", "STI", "sungai", "TWI"
]
FEATURE_FILES = [f"{name}.tif" for name in FEATURE_NAMES]
MODEL_PATH = "hasil kotak 97,95/hasil terbaik.pth"
FEATURE_PATH = "data_kotak/feature/"
LABEL_PATH = "data_kotak/label/label1.tif"
WINDOW_SIZE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "Final_SHAP_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== BACA PARAMETER NORMALISASI ==================
with open("norm_params.json", "r") as f:
    norm_params = json.load(f)

def inverse_normalize(values, min_val, max_val):
    return values * (max_val - min_val) + min_val

# ================== LOAD DATA & MODEL ==================
_, _, n_feature, data = dp.get_feature_data(FEATURE_PATH, WINDOW_SIZE)
label = dp.get_label_data(LABEL_PATH, WINDOW_SIZE)
train_x, train_y, val_x, val_y = dp.get_CNN_data(data, label, WINDOW_SIZE)

X_background = train_x[:100]
X_test = val_x
y_test = val_y[:200].squeeze()

X_bg_tensor = torch.tensor(X_background, dtype=torch.float32).to(DEVICE)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

model = LSM_cnn(n_feature).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ================== SHAP EXPLANATION ==================
explainer = shap.GradientExplainer(model, X_bg_tensor)
shap_values = explainer.shap_values(X_test_tensor)

# SHAP untuk kelas 1 (longsor)
shap_vals_1 = shap_values[1][:, :, :, :].mean(axis=(2, 3))
# SHAP untuk kelas 0 (tidak longsor)
shap_vals_0 = shap_values[0][:, :, :, :].mean(axis=(2, 3))

X_flat = X_test[:shap_vals_1.shape[0]].mean(axis=(2, 3))

# Sinkronisasi kolom jika perlu
if shap_vals_1.shape[1] > X_flat.shape[1]:
    shap_vals_1 = shap_vals_1[:, :X_flat.shape[1]]
if shap_vals_0.shape[1] > X_flat.shape[1]:
    shap_vals_0 = shap_vals_0[:, :X_flat.shape[1]]

# ================== SHAP SUMMARY PLOT ==================
shap.summary_plot(shap_vals_1, X_flat, feature_names=FEATURE_NAMES, show=False, plot_type="violin")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_class1.png"))
plt.close()

shap.summary_plot(shap_vals_0, X_flat, feature_names=FEATURE_NAMES, show=False, plot_type="violin")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_class0.png"))
plt.close()

# ================== MEAN SHAP BAR PLOT ==================
mean_shap_1 = np.abs(shap_vals_1).mean(axis=0)

plt.figure(figsize=(10, 5))
plt.bar(FEATURE_NAMES, mean_shap_1)
plt.xticks(rotation=45)
plt.ylabel("Mean |SHAP value|")
plt.title("Global Feature Importance (Mean SHAP - Class 1 / Longsor)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_mean_class1.png"))
plt.close()

# ================== SHAP DEPENDENCE PLOT ELEVASI ==================
elev_idx = FEATURE_NAMES.index("elevasi")
slope_idx = FEATURE_NAMES.index("Slope")

x_real = inverse_normalize(X_flat[:, elev_idx], norm_params["elevasi.tif"]["min"], norm_params["elevasi.tif"]["max"])
y_shap = shap_vals_1[:, elev_idx]
color_real = inverse_normalize(X_flat[:, slope_idx], norm_params["Slope.tif"]["min"], norm_params["Slope.tif"]["max"])

plt.figure(figsize=(8, 5))
sc = plt.scatter(x_real, y_shap, c=color_real, cmap="coolwarm", s=25)
plt.colorbar(sc, label="Slope (real)")
plt.xlabel("Altitude (elevasi)")
plt.ylabel("SHAP value (elevasi)")
plt.title("SHAP Dependence Plot: Elevasi vs SHAP (colored by Slope)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_dependence_elevasi.png"))
plt.close()

# ================== SHAP DEPENDENCE PLOT SLOPE ==================
x_real = inverse_normalize(X_flat[:, slope_idx], norm_params["Slope.tif"]["min"], norm_params["Slope.tif"]["max"])
y_shap = shap_vals_1[:, slope_idx]
color_real = inverse_normalize(X_flat[:, elev_idx], norm_params["elevasi.tif"]["min"], norm_params["elevasi.tif"]["max"])

plt.figure(figsize=(8, 5))
sc = plt.scatter(x_real, y_shap, c=color_real, cmap="coolwarm", s=25)
plt.colorbar(sc, label="Elevasi (real)")
plt.xlabel("Slope")
plt.ylabel("SHAP value (Slope)")
plt.title("SHAP Dependence Plot: Slope vs SHAP (colored by Elevasi)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_dependence_slope.png"))
plt.close()

# ================== FORCE PLOTS ==================
def plot_force(idx, shap_vals_class, filename):
    real_feats = [
        inverse_normalize(X_flat[idx, i], norm_params[FEATURE_FILES[i]]["min"], norm_params[FEATURE_FILES[i]]["max"])
        for i in range(len(FEATURE_NAMES))
    ]
    shap.force_plot(
        base_value=0.0,
        shap_values=shap_vals_class[idx],
        features=real_feats,
        feature_names=FEATURE_NAMES,
        matplotlib=True,
        show=False
    )
    plt.title(f"SHAP Force Plot - Sample {idx}")
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

plot_force(5, shap_vals_1, "shap_force_longsor.png")
plot_force(10, shap_vals_0, "shap_force_nonlongsor.png")

# ================== PERMUTATION FEATURE IMPORTANCE ==================
pfi_scores = []
acc_orig = accuracy_score(y_test, model(X_test_tensor).argmax(dim=1).cpu().numpy())

for i in range(n_feature):
    X_perm = X_test.copy()
    np.random.shuffle(X_perm[:, i, :, :])
    X_perm_tensor = torch.tensor(X_perm, dtype=torch.float32).to(DEVICE)
    y_pred_perm = model(X_perm_tensor).argmax(dim=1).cpu().numpy()
    acc_shuffled = accuracy_score(y_test, y_pred_perm)
    pfi_scores.append(acc_orig - acc_shuffled)

# ================== SHAP vs PFI COMPARISON ==================
mean_shap_vals = np.abs(shap_vals_1).mean(axis=0)
x = np.arange(n_feature)
bar_width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - bar_width/2, mean_shap_vals, width=bar_width, label='SHAP (Class 1)')
plt.bar(x + bar_width/2, pfi_scores, width=bar_width, label='PFI')
plt.xticks(x, FEATURE_NAMES, rotation=45)
plt.ylabel("Feature Importance")
plt.title("SHAP (Longsor) vs Permutation Feature Importance")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_vs_pfi.png"))
plt.close()