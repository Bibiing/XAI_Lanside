from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import os
import numpy as np
import argparse
import json
from joblib import dump

from pipeline import dataset
from utils import drawAUC_TwoClass


def parse_args():
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument("--feature_path", default='Data/samodra/A', type=str)
    parser.add_argument("--label_path", default='Data/samodra/label/A/label.tif', type=str)
    parser.add_argument("--output_dir", default='output_train', type=str)
    parser.add_argument("--model", default='ModelRF', type=str, help="ModelRF or ModelGB")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_df, val_df, norm_params = dataset.get_ML_data(args.feature_path, args.label_path)
    x_train = train_df.iloc[:, :-1]
    y_train = np.array(train_df.iloc[:, -1]).ravel().astype(int)
    x_test = val_df.iloc[:, :-1]
    y_test = np.array(val_df.iloc[:, -1]).ravel().astype(int)
    
    print(f"Training features shape: {x_train.shape}")
    print(f"Validation features shape: {x_test.shape}")
    print(f"Training labels: {np.unique(y_train)}")
    print(f"Validation labels: {np.unique(y_test)}")

    ModelRF = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
    )

    ModelGB = GradientBoostingClassifier(
        random_state=42,
    )

    models = {
        "ModelRF": ModelRF,
        "ModelGB": ModelGB,
    }

    model = models[args.model]

    model.fit(x_train, y_train)
    model_path = os.path.join(args.output_dir, f"{args.model}.joblib")
    dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save normalization parameters
    norm_path = os.path.join(args.output_dir, f"{args.model}_norm_params.json")
    with open(norm_path, 'w') as f:
        json.dump(norm_params, f, indent=2)
    print(f"Normalization params saved to: {norm_path}")
    
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)

    # 0=ls, 1=nls 
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[1,0]).reshape(-1)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp / (tp + fp) if tp+fp > 0 else 0
    recall = tp / (tp + fn) if tp+fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision+recall > 0 else 0

    print("\n--- VALIDATION RESULT ---")
    print(f"Accuracy   : {accuracy*100:.2f}%")
    print(f"Precision  : {precision*100:.2f}%")
    print(f"Recall     : {recall*100:.2f}%")
    print(f"F1-Score   : {f1*100:.2f}%")

    # AUC
    auc_path = os.path.join(args.output_dir, f"{args.model}_AUC.png")
    drawAUC_TwoClass(y_test, model.predict_proba(x_test)[:,1], auc_path)
    print(f"AUC saved to: {auc_path}")

    train_pred = model.predict(x_train)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_train, train_pred, labels=[1,0]).reshape(-1)
    train_acc = (tp_t+tn_t)/(tp_t+tn_t+fp_t+fn_t)
    train_prec = tp_t / (tp_t + fp_t) if tp_t+fp_t > 0 else 0
    train_recall = tp_t / (tp_t + fn_t) if tp_t+fn_t > 0 else 0
    train_f1 = 2 * train_prec * train_recall / (train_prec + train_recall) if train_prec+train_recall > 0 else 0
    
    print("\n--- TRAIN RESULT ---")
    print(f"Train Accuracy   : {train_acc*100:.2f}%")
    print(f"Train Precision  : {train_prec*100:.2f}%")
    print(f"Train Recall     : {train_recall*100:.2f}%")
    print(f"Train F1-Score   : {train_f1*100:.2f}%")

if __name__ == "__main__":
    main()
