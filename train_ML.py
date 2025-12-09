import optuna
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, recall_score

import os
import numpy as np
import argparse
from joblib import dump

from pipeline import dataset
from utils import drawAUC_TwoClass

def parse_args():
    parser = argparse.ArgumentParser(description="Train ML models Processes on data")
    parser.add_argument( "--feature_path", default='Data/samodra/A/', type=str, help="path to feature data")
    parser.add_argument( "--label_path", default='Data/samodra/label/A/label.tif', type=str, help="path to label data")
    parser.add_argument( "--output_dir", default='Hasil/samodra', type=str, help="output directory")
    parser.add_argument( "--model", default='ModelRF', type=str, help="ModelRF, ModelGB")
    parser.add_argument("--tune", action='store_true', help="optuna hyperparameter tuning")
    args = parser.parse_args()
    return args

def optimize_rf(trial, x_train, y_train, x_test, y_val):
    n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 15)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    max_samples = trial.suggest_float("max_samples", 0.5, 0.9)  # Bagging for variance reduction
    min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.0, 0.01)  # Pruning
    ccp_alpha = trial.suggest_float("ccp_alpha", 0.0, 0.01)  # Cost-complexity pruning

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_samples=max_samples,
        min_impurity_decrease=min_impurity_decrease,
        ccp_alpha=ccp_alpha,
        bootstrap=True,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return recall_score(y_val, y_pred)

def optimize_gb(trial, x_train, y_train, x_test, y_val):
    n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    max_depth = trial.suggest_int("max_depth", 2, 5)
    subsample = trial.suggest_float("subsample", 0.5, 0.9)  # Stochastic gradient boosting
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 15)
    min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.0, 0.01)
    ccp_alpha = trial.suggest_float("ccp_alpha", 0.0, 0.01)  # Pruning

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=min_impurity_decrease,
        ccp_alpha=ccp_alpha,
        validation_fraction=0.2,
        n_iter_no_change=15,  # patience for early stopping
        random_state=42
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return recall_score(y_val, y_pred)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_df, val_df = dataset.get_ML_data(args.feature_path, args.label_path)
    x_train = train_df.iloc[:, :-1]
    y_train = np.array(train_df.iloc[:, -1]).ravel().astype(int)
    x_test = val_df.iloc[:, :-1]
    y_test = np.array(val_df.iloc[:, -1]).ravel().astype(int)
    
    print(f"Training features shape: {x_train.shape}")
    print(f"Validation features shape: {x_test.shape}")
    print(f"Training labels: {np.unique(y_train)}")
    print(f"Validation labels: {np.unique(y_test)}")

    ModelRF = RandomForestClassifier(
        random_state=42, 
        n_estimators=100, 
        max_depth=5, 
        min_samples_leaf=12, 
        min_samples_split=6, 
        max_features='sqrt',
        max_samples=0.7, 
        min_impurity_decrease=0.002,  
        ccp_alpha=0.003, 
        class_weight='balanced',
        n_jobs=-1
    )
    ModelGB = GradientBoostingClassifier(
        random_state=42, 
        n_estimators=100,  
        learning_rate=0.02, 
        max_depth=3, 
        subsample=0.6, 
        max_features='sqrt',
        min_samples_leaf=10,  
        min_samples_split=6,  
        min_impurity_decrease=0.002,  
        ccp_alpha=0.003, 
        validation_fraction=0.2, 
        n_iter_no_change=15  
    )

    models = {
        "ModelRF": ModelRF,
        "ModelGB": ModelGB,
    }

    model = models.get(args.model)

    if args.tune:
        study = optuna.create_study(direction="maximize")
        if args.model == "ModelRF":
            study.optimize(lambda t: optimize_rf(t, x_train, y_train, x_test, y_test), n_trials=40)
        elif args.model == "ModelGB":
            study.optimize(lambda t: optimize_gb(t, x_train, y_train, x_test, y_test), n_trials=40)

        print("\nBest Hyperparameters:")
        print(study.best_params)
        # print(f"Best CV Recall: {study.best_value:.4f}")

        # best parameters
        model = model.set_params(**study.best_params)
        
    model.fit(x_train, y_train)   

    model_path = os.path.join(args.output_dir, f"{args.model}_best.joblib")
    dump(model, model_path)
    print(f"Model saved to: {model_path}")

    y_pred = model.predict(x_test)

    print(f'Model Name: {model}')
    tp, fn, fp, tn = confusion_matrix(y_test, y_pred, labels=[1,0]).reshape(-1)
    accuracy = (tp+tn)/(tp+fp+tn+fn) if (tp+fp+tn+fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0 # How many predicted landslides were actual landslides
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0    # Of all actual landslides, how many did we find?
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
    print(f'Accuracy   : {accuracy*100:.2f}%')
    print(f'Precision  : {precision*100:.2f}%')
    print(f'Recall     : {recall*100:.2f}%')
    print(f'F1-Score   : {f1*100:.2f}%')
        
    drawAUC_TwoClass(y_test, model.predict_proba(x_test)[:, 1], os.path.join(args.output_dir, f'{args.model}_AUC.png'))

    # check overfitting or not
    # train_pred = model.predict(x_train)
    # tp_t, fn_t, fp_t, tn_t = confusion_matrix(y_train, train_pred, labels=[1,0]).reshape(-1)
    # train_accuracy = (tp_t + tn_t) / (tp_t + fp_t + tn_t + fn_t)
    # train_precision = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    # train_recall = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    # train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0

    # print(f"Train Accuracy : {train_accuracy*100:.2f}%")
    # print(f"Train Precision: {train_precision*100:.2f}%")
    # print(f"Train Recall   : {train_recall*100:.2f}%")
    # print(f"Train F1-Score : {train_f1*100:.2f}%")

if __name__=='__main__':
    main()