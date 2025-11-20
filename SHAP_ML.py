import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

from pipeline import dataset

def parse_args():
    parser = argparse.ArgumentParser(description="SHAP analysis for trained ML models")
    parser.add_argument("--model_path", default="Hasil/samodra/ModelRF_best.joblib", type=str)
    parser.add_argument("--feature_path", default="Data/samodra/A/", type=str)
    parser.add_argument("--label_path", default="Data/samodra/label/A/label.tif", type=str)
    parser.add_argument("--output_dir", default="Hasil/samodra/SHAP", type=str)
    return parser.parse_args()

def unify_shap_output(shap_values):
    """
    Convert SHAP outputs into 2 arrays, each shape = (N, F)
    """
    # shap_values is a list
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_vals_0 = np.array(shap_values[0])
            shap_vals_1 = np.array(shap_values[1])
            return shap_vals_0, shap_vals_1
        else:
            raise ValueError("Unexpected list length for SHAP values.")

    # shap_values shape = (N, F, 2)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_vals_0 = shap_values[:, :, 0]
        shap_vals_1 = shap_values[:, :, 1]
        return shap_vals_0, shap_vals_1
    
    # binary classifier
    else:
        shap_vals_1 = np.array(shap_values)
        shap_vals_0 = -shap_vals_1
        return shap_vals_0, shap_vals_1


args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)

try:
    model = joblib.load(args.model_path)
except Exception as e:
    print("Error loading model:", e)
    exit(1)

train_df, val_df = dataset.get_ML_data(args.feature_path, args.label_path)

x_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1].astype(int)
x_test = val_df.iloc[:, :-1]
y_test = val_df.iloc[:, -1].astype(int)

feature_names = x_train.columns.tolist()
# print(feature_names)
print("load data success")

x_test = x_test = x_test.copy()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_test)
shap_vals_0, shap_vals_1 = unify_shap_output(shap_values)

# SHAP Summary Plots
plt.figure()
shap.summary_plot(shap_vals_1, x_test, feature_names=feature_names, show=False, plot_type="violin")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "shap_summary_nls.png"))
plt.close()
print("violin plot class 1 done")

plt.figure()
shap.summary_plot(shap_vals_0, x_test, feature_names=feature_names, show=False, plot_type="violin")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "shap_summary_ls.png"))
plt.close()
print("violin plot class 0 done")

# Feature Importance Plot
plt.figure()
shap.summary_plot(shap_vals_1, x_test, feature_names=feature_names, show=False, plot_type="bar")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "shap_feature_importance.png"))
plt.close()
print("feature importance plot done")

mean_shap = np.abs(shap_vals_1).mean(axis=0)
top2 = np.argsort(mean_shap)[-2:][::-1] # top-2

top1, top2 = top2
feat1_name = feature_names[top1]
feat2_name = feature_names[top2]
# print("Top feature 1:", top1)
# print("Top feature 2:", top2)

#dependence plots
plt.figure()
shap.dependence_plot(top1, shap_vals_1, x_test, interaction_index=top2, show=False)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, f"dependence_{feat1_name}_vs_{feat2_name}.png"))
plt.close()

def plot_force(idx, shap_vals_class,  filename):
    feats = [""] * len(feature_names)
    shap.force_plot(
        base_value=0.0,
        shap_values=shap_vals_class[idx],
        features=feats,              
        feature_names=feature_names,  
        matplotlib=True,
        show=False
    )

    plt.title(f"SHAP Force Plot - Sample {idx}")
    plt.savefig(os.path.join(args.output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()

# Plot force plots for specific samples
plot_force(5, shap_vals_0, "force_plot_ls.png")
plot_force(5, shap_vals_1, "force_plot_nls.png")

print("SHAP analysis complete!")