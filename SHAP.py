import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

def tree_explainer(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Binary / multiclass handling
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values[1])  # class positif
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        print("1")
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        print("2")
    else:
        shap_values = np.array(shap_values)
        expected_value = explainer.expected_value
        print("3")

    return explainer, shap_values, expected_value


def cnn_explainer(model, X_background, X_test, device):
    X_bg_tensor = torch.tensor(X_background, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    model.eval()
    explainer = shap.GradientExplainer(model, X_bg_tensor)
    shap_values = explainer.shap_values(X_test_tensor)
    return explainer, shap_values

def plot_shap_summary(shap_values, X, feature_names=None, plot_type="violin"):
    # Convert feature_names to list to avoid numpy array issues
    if feature_names is not None:
        feature_names = list(feature_names)
    
    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        plot_type=plot_type,
        show=False
    )
    plt.tight_layout()
    plt.show()

# def plot_feature_importance(shap_values, feature_names, top_n=20):
#   mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
#   idx = np.argsort(mean_abs_shap)[::-1][:top_n]
    
#   plt.figure(figsize=(10, 5))
#   plt.bar([feature_names[i] for i in idx], mean_abs_shap[idx])
#   plt.xticks(rotation=45, ha='right')
#   plt.ylabel("Mean |SHAP value|")
#   plt.title("Global Feature Importance (SHAP)")
#   plt.tight_layout()
#   plt.show()

def plot_feature_importance(shap_values, feature_names, top_n=20):
    shap_values = np.array(shap_values)

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    idx = np.argsort(mean_abs_shap)[::-1][:top_n]

    # Convert to list to avoid numpy array issues
    feature_names_list = list(feature_names)

    plt.figure(figsize=(10, 5))
    plt.bar([feature_names_list[i] for i in idx], mean_abs_shap[idx])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Mean |SHAP value|")
    plt.title("Feature Importance (SHAP)")
    plt.tight_layout()
    plt.show()

def plot_dependence_top2(shap_values, X, feature_names):
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:2]

    f1 = feature_names[top_idx[0]]
    f2 = feature_names[top_idx[1]]
    print(f"Top 2 features: {f1}, {f2}")
    plt.figure()
    shap.dependence_plot(f1, shap_values, X, interaction_index=f2, show=False)
    plt.title(f"SHAP Dependence: {f1} vs {f2}")
    plt.tight_layout()
    plt.show()



def plot_force(explainer, shap_values, X, expected_value, index=0):
    shap.initjs()
    return display(shap.force_plot(expected_value, shap_values[index], X.iloc[index]))




