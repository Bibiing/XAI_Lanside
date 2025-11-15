import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import data_prepare as dp
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--feature_path", default='data_kotak/feature/', type=str)
parser.add_argument("--window_size", default=15, type=int)
args = parser.parse_args()

# Ambil nama fitur
feature_files = sorted([f for f in os.listdir(args.feature_path) if f.endswith('.tif')])
feature_names = [os.path.splitext(f)[0] for f in feature_files]

# Ambil data fitur
w, h, n_feature, data = dp.get_feature_data(args.feature_path, args.window_size)

# Hilangkan padding
n = args.window_size // 2
data = data[:, n:w+n, n:h+n]  # shape: (n_feature, w, h)

# Reshape: (n_feature, w*h) -> (w*h, n_feature)
data_reshaped = data.reshape(n_feature, -1).T

# Buat DataFrame dengan nama kolom sesuai nama fitur
df = pd.DataFrame(data_reshaped, columns=feature_names)
corr = df.corr()

# Simpan matriks korelasi ke file CSV (bentuk matriks)
corr.to_csv('feature_correlation.csv')

# Simpan tabel pasangan fitur dan korelasinya (bentuk long table)
corr_pairs = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)
corr_pairs.columns = ['Fitur 1', 'Fitur 2', 'Korelasi']
corr_pairs['Korelasi Absolut'] = corr_pairs['Korelasi'].abs()
corr_pairs = corr_pairs.sort_values(by='Korelasi Absolut', ascending=False)
corr_pairs.to_csv('feature_correlation_pairs.csv', index=False)

# Tampilkan matriks korelasi dengan nama fitur
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=feature_names, yticklabels=feature_names)
plt.title("Matriks Korelasi Antar Fitur")
plt.tight_layout()
plt.show()