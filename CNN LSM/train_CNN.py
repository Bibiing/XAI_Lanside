import os
import json
import torch
import argparse
import numpy as np

import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from pipeline import reader, preprocessor, dataset
from model import LSM_cnn
from utils import drawAUC_TwoClass, draw_acc, draw_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN Processes on data")
    parser.add_argument("--feature_path", default='Data/samodra/A/', type=str)
    parser.add_argument("--label_path", default='Data/samodra/label/A/label.tif', type=str)
    parser.add_argument("--output_dir", default='Hasil/', type=str)
    parser.add_argument("--window_size", default=45, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    padded_features = []
    n = args.window_size // 2

    reader.validate_consistency(args.feature_path, args.label_path) # validate data
    feature_files = sorted([f for f in os.listdir(args.feature_path) if f.lower().endswith('.tif')])

    # processing features
    for feature_name in feature_files:
        img = reader.read_data_from_tif(os.path.join(args.feature_path, feature_name))
        norm_img, _, _ = preprocessor.normalize_min_max(img)
        padded_img = preprocessor.apply_padding(norm_img, n, pad_value=0)
        padded_features.append(padded_img)
        
    feature_block = np.array(padded_features)
    print(f"Feature block created successfully: {feature_block.shape}")
    
    # processing label
    label_img = reader.read_data_from_tif(args.label_path)
    padded_label = preprocessor.apply_padding(label_img, n, pad_value=0.1)
    
    # create CNN dataset
    train_x, train_y, val_x, val_y = dataset.get_CNN_data(
        feature_block, padded_label, args.window_size
    )
    print(f"Dataset created: {train_x.shape[0]} train data, {val_x.shape[0]} val data.")

    # dataloader 
    train_dataset = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = TensorDataset(torch.from_numpy(val_x).float(), torch.from_numpy(val_y).float())
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = LSM_cnn(in_chanel=feature_block.shape[0]).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    max_acc = 0.0
    record = {"train": {"acc": [], "loss": []}, "val": {"acc": [], "loss": []}}
    
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        train_outputs_list, train_labels_list = [], []
        
        for images, target in train_loader:
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, target.squeeze().long())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += (preds.squeeze() == target.squeeze()).sum().item()
            train_outputs_list.extend(outputs.detach().cpu().numpy())
            train_labels_list.extend(target.cpu().numpy())
        
        # Evaluasi
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        val_outputs_list, val_labels_list = [], []
        
        with torch.no_grad():
            for images, target in val_loader:
                images, target = images.to(device), target.to(device)
                outputs = model(images)
                loss = criterion(outputs, target.squeeze().long())
                _, preds = torch.max(outputs.data, 1)
                
                val_loss += loss.item()
                val_acc += (preds.squeeze() == target.squeeze()).sum().item()
                val_outputs_list.extend(outputs.detach().cpu().numpy())
                val_labels_list.extend(target.cpu().numpy())
        
        avg_train_acc = train_acc / len(train_dataset)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_acc = val_acc / len(val_dataset)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'[{epoch + 1:03d}/{args.epochs:03d}] Train Acc: {avg_train_acc:.6f} Loss: {avg_train_loss:.6f} | Val Acc: {avg_val_acc:.6f} Loss: {avg_val_loss:.6f}')
        
        # save best model
        if avg_val_acc > max_acc:
            print(f'Val Acc meningkat ({max_acc:.6f} --> {avg_val_acc:.6f}). Menyimpan model...')
            max_acc = avg_val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best.pth'))
            
            score_array_val = np.array(val_outputs_list)
            score_array_train = np.array(train_outputs_list)
            drawAUC_TwoClass(val_labels_list, score_array_val[:, 1], os.path.join(args.output_dir, 'val_AUC.png'))
            drawAUC_TwoClass(train_labels_list, score_array_train[:, 1], os.path.join(args.output_dir, 'train_AUC.png'))

        record["train"]["acc"].append(avg_train_acc)
        record["train"]["loss"].append(avg_train_loss)
        record["val"]["acc"].append(avg_val_acc)
        record["val"]["loss"].append(avg_val_loss)
    
        scheduler.step()

    print("Pelatihan selesai.")
    draw_acc(record["train"]["acc"], record["val"]["acc"], os.path.join(args.output_dir, 'accuracy.png'))
    draw_loss(record["train"]["loss"], record["val"]["loss"], os.path.join(args.output_dir, 'loss.png'))

    with open(os.path.join(args.output_dir, 'record.json'), 'w') as f:
        json.dump(record, f, indent=4)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'latest.pth'))

if __name__ == '__main__':
    main()