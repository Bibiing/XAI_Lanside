from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from pipeline import dataset
from utils import drawAUC_TwoClass

def parse_args():
    parser = argparse.ArgumentParser(description="Train ML models Processes on data")
    parser.add_argument( "--feature_path", default='Data/samodra/A/', type=str)
    parser.add_argument( "--label_path", default='Data/samodra/label/A/label.tif', type=str)
    parser.add_argument( "--output_dir", default='Hasil/', type=str)
    parser.add_argument( "--model", default='ModelRF', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    train_df, val_df = dataset.get_ML_data(args.feature_path, args.label_path)
    x_train = train_df.iloc[:, :-1]
    y_train = np.array(train_df.iloc[:, -1]).ravel().astype(int)
    x_test = val_df.iloc[:, :-1]
    y_test = np.array(val_df.iloc[:, -1]).ravel().astype(int)
    
    print(f"Training features shape: {x_train.shape}")
    print(f"Validation features shape: {x_test.shape}")
    print(f"Training labels: {np.unique(y_train)}")
    print(f"Validation labels: {np.unique(y_test)}")


    ModelRF = RandomForestClassifier()
    ModelET = ExtraTreesClassifier()
    ModelKNN = KNeighborsClassifier(n_neighbors=2)
    ModelSVM = SVC(probability=True)
    ModelDC = DecisionTreeClassifier()

    models = {
        "ModelRF": ModelRF,
        "ModelET": ModelET,
        "ModelKNN": ModelKNN,
        "ModelSVM": ModelSVM,
        "ModelDC": ModelDC
    }

    if args.model in models:
        model = models[args.model]
    
        print("\n5-Fold Cross-Validation")
        cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
        print(f"Scores for each fold: {cv_scores}")
        print(f"CV Mean Accuracy: {cv_scores.mean():.4f}")
        print(f"CV Std. Deviation: {cv_scores.std():.4f}")
        print("--------------------------------------------------\n")
        
        model.fit(x_train, y_train)   
        y_pred = model.predict(x_test)
        
        print(f'Model Name: {model}')
        tp, fn, fp, tn = confusion_matrix(y_test, y_pred, labels=[1,0]).reshape(-1)
        accuracy = (tp+tn)/(tp+fp+tn+fn)
        precision = tp / (tp + fp) # How many predicted landslides were actual landslides
        recall = tp / (tp + fn)    # Of all actual landslides, how many did we find?
        f1 = 2 * (precision * recall) / (precision + recall)
        
        print(f'Accuracy   : {accuracy*100:.2f}%')
        print(f'Precision  : {precision*100:.2f}%')
        print(f'Recall     : {recall*100:.2f}%')
        print(f'F1-Score   : {f1*100:.2f}%')
        
        drawAUC_TwoClass(y_test, model.predict_proba(x_test)[:, 1], os.path.join(args.output_dir, f'{args.model}_AUC.png'))

if __name__=='__main__':
    main()