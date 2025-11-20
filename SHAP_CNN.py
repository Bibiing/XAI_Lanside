import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import json
import os

from pipeline import reader, preprocessor, dataset
from model import LSM_cnn
from sklearn.metrics import accuracy_score
import warnings
import matplotlib
matplotlib.use('Agg')  # Gunakan non-GUI backend untuk simpan plot

# Suppress FutureWarnings
warnings.simplefilter("ignore", category=FutureWarning)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="SHAP Explanation for CNN LSM Model")
    parser.add_argument("--feature_path", type=str, default="Data/mojokerto/feature11/", help="Path to feature data")
    parser.add_argument("--label_path", type=str, default="Data/mojokerto/label/label1.tif", help="Path to label data")
    parser.add_argument("--model_path", type=str, default="Hasil/mojokerto/feature11/best.pth", help="Path to trained model")
    parser.add_argument("--norm_params", type=str, default="Hasil/mojokerto/feature11/norm_params.json", help="Path to normalization parameters JSON file")
    parser.add_argument("--window_size", type=int, default=15, help="Window size for CNN input")
    parser.add_argument("--output_dir", type=str, default="Hasil/mojokerto/feature11/SHAP", help="Directory to save SHAP results")
    return parser.parse_args()

args = parse_args()
n = args.window_size // 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.output_dir, exist_ok=True)

with open(args.norm_params, "r") as f:
    norm_params = json.load(f)

def inverse_normalize(values, min_val, max_val):
    return values * (max_val - min_val) + min_val

feature_files = sorted([f for f in os.listdir(args.feature_path) if f.lower().endswith('.tif')])
print(f"Feature files: {feature_files}")

# load data and model
padded_features = []
for feature_name in feature_files:
        img = reader.read_data_from_tif(os.path.join(args.feature_path, feature_name))
        norm_img, _ = preprocessor.normalize_min_max(img)
        padded_img = preprocessor.apply_padding(norm_img, n, pad_value=0)
        padded_features.append(padded_img)
feature_block = np.array(padded_features)
label_img = reader.read_data_from_tif(args.label_path)
padded_label = preprocessor.apply_padding(label_img, n, pad_value=0.1)
    
# create CNN dataset
train_x, train_y, val_x, val_y = dataset.get_CNN_data(
    feature_block, padded_label, args.window_size
)
print(f"Dataset created: {train_x.shape[0]} train data, {val_x.shape[0]} val data.")
X_background = train_x[:100]
X_test = val_x
y_test = val_y[:200].squeeze()

X_bg_tensor = torch.tensor(X_background, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

model = LSM_cnn(in_chanel=feature_block.shape[0]).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

#  SHAP EXPLANATION 
explainer = shap.GradientExplainer(model, X_bg_tensor)
shap_values = explainer.shap_values(X_test_tensor)

# SHAP untuk kelas 0 (longsor)
shap_vals_0 = shap_values[0][:, :, :, :].mean(axis=(2, 3))
# SHAP untuk kelas 1 (tidak longsor)
shap_vals_1 = shap_values[1][:, :, :, :].mean(axis=(2, 3))

X_flat = X_test[:shap_vals_1.shape[0]].mean(axis=(2, 3))

# Sinkronisasi kolom jika perlu
if shap_vals_1.shape[1] > X_flat.shape[1]:
    shap_vals_1 = shap_vals_1[:, :X_flat.shape[1]]
if shap_vals_0.shape[1] > X_flat.shape[1]:
    shap_vals_0 = shap_vals_0[:, :X_flat.shape[1]]

#  SHAP SUMMARY PLOT 
shap.summary_plot(shap_vals_1, X_flat, feature_names=feature_files, show=False, plot_type="violin")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "shap_summary_nls.png"))
plt.close()
print("summary plot nls done")

shap.summary_plot(shap_vals_0, X_flat, feature_names=feature_files, show=False, plot_type="violin")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "shap_summary_ls.png"))
plt.close()
print("summary plot ls done")

# important features
mean_shap_0 = np.abs(shap_vals_0).mean(axis=0)
#  MEAN SHAP BAR PLOT 
plt.figure(figsize=(10, 5))
plt.bar(feature_files, mean_shap_0)
plt.xticks(rotation=45)
plt.ylabel("Mean |SHAP value|")
plt.title("Global Feature Importance (Mean SHAP - Longsor)")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "shap_mean_ls.png"))
plt.close()
print("mean shap plot done")

# SHAP dependence plots
def plot_dependence(feat_x_idx, feat_color_idx, filename):
    feat_x_name = feature_files[feat_x_idx]
    feat_color_name = feature_files[feat_color_idx]

    # undo normalization
    x_real = inverse_normalize(
        X_flat[:, feat_x_idx],
        norm_params[feat_x_name]["min"],
        norm_params[feat_x_name]["max"]
    )

    color_real = inverse_normalize(
        X_flat[:, feat_color_idx],
        norm_params[feat_color_name]["min"],
        norm_params[feat_color_name]["max"]
    )

    y_shap = shap_vals_1[:, feat_x_idx]

    # plot
    plt.figure(figsize=(8, 5))
    sc = plt.scatter(x_real, y_shap, c=color_real, cmap="coolwarm", s=25)
    plt.colorbar(sc, label=feat_color_name)
    plt.xlabel(feat_x_name)
    plt.ylabel(f"SHAP value ({feat_x_name})")
    plt.title(f"SHAP Dependence: {feat_x_name} (colored by {feat_color_name})")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, filename))
    plt.close()

top2 = np.argsort(mean_shap_0)[-2:][::-1] # top-2

top1, top2 = top2
feat1_name = feature_files[top1]
feat2_name = feature_files[top2]

# print("features:", feat1_name, ",", feat2_name)
plot_dependence(top1, top2, f"dependence_{feat1_name}_vs_{feat2_name}.png")
print("dependence plots done")

# FORCE PLOTS 
def plot_force(idx, shap_vals_class, filename):
    real_feats = [
        inverse_normalize(X_flat[idx, i], norm_params[feature_files[i]]["min"], norm_params[feature_files[i]]["max"])
        for i in range(len(feature_files))
    ]
    shap.force_plot(
        base_value=0.0,
        shap_values=shap_vals_class[idx],
        features=real_feats,
        feature_names=feature_files,
        matplotlib=True,
        show=False
    )
    plt.title(f"SHAP Force Plot - Sample {idx}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, filename))
    plt.close()

plot_force(10, shap_vals_0, "shap_force_ls.png")
plot_force(10, shap_vals_1, "shap_force_nls.png")
print("force plots done")
print("All SHAP done")