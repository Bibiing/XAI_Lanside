import os
from osgeo import gdal
import argparse
import sys
import numpy as np

from pipeline import reader, preprocessor, dataset

parser = argparse.ArgumentParser()
parser.add_argument( "--feature_path", default='Data/samodra/B/', type=str)
parser.add_argument( "--label_path", default='Data/samodra/label/B/label.tif', type=str)
parser.add_argument( "--window_size", default=45, type=int)
args = parser.parse_args()

def get_tiff_attributes(tiff_path):
    tiff = gdal.Open(tiff_path)
    width, height = tiff.RasterXSize, tiff.RasterYSize
    transform = tiff.GetGeoTransform()
    return width, height, transform

consistency = {"w": [], "h":[], "transform":[]}
#label 
w, h, transform = get_tiff_attributes(args.label_path)
consistency["w"].append(w)
consistency["h"].append(h)
consistency["transform"].append([int(value) for value in transform])
print(f"Reading label1.tif data. \nWidth: {w}, Height: {h}, Transform: {transform}")
#feature
for tif_data in os.listdir(args.feature_path,):
    w, h, transform = get_tiff_attributes(os.path.join(args.feature_path, tif_data))
    print('Reading '+tif_data+' data.')
    consistency["w"].append(w)
    consistency["h"].append(h)
    consistency["transform"].append([int(value) for value in transform])
    print(f"Width: {w}, Height: {h}, Transform: {transform}")
consistency["w"] = set(consistency["w"])
consistency["h"] = set(consistency["h"])
if(len(consistency["w"])!=1):
    print("The width of the data is different")
    print(consistency["w"])
    sys.exit()
if(len(consistency["h"])!=1):
    print("The height of the data is different")
    print(consistency["h"])
    sys.exit()
flag = consistency["transform"][0]  # Get the first tuple in the list
if(not(all(t == flag for t in consistency["transform"]))):
    print("The transform of the data is different")
    print(consistency["transform"])
    sys.exit()

print("datamu aman lee....")

feature_files = sorted([f for f in os.listdir(args.feature_path) if f.lower().endswith('.tif')])

padded_features = []
n = args.window_size // 2
    
# processing features
for feature_name in feature_files:
    img, _, _ = reader.read_data_from_tif(os.path.join(args.feature_path, feature_name))
    norm_img, _, _ = preprocessor.normalize_min_max(img)
    padded_img = preprocessor.apply_padding(norm_img, n, pad_value=0)
    padded_features.append(padded_img)
        
feature_block = np.array(padded_features)
print(f"Feature block created successfully: {feature_block.shape}")

# processing label
label_img, _, _ = reader.read_data_from_tif(args.label_path)
padded_label = preprocessor.apply_padding(label_img, n, pad_value=0.1)
    
# --- create CNN dataset ---
train_x, train_y, val_x, val_y = dataset.get_CNN_data( feature_block, padded_label, args.window_size    )
print(f"Dataset created: {train_x.shape[0]} train data, {val_x.shape[0]} val data.")

# mojokerto Width: 374, Height: 716, feature block (11, 730, 388), 420 train data, 180 val data. : with window size 15
# with lr 0.0001, batch size 128, epochs 500, window size 15, StepLR constant got > 90% val accuracy

# samodra A Width: 2162, Height: 1463, feature block (11, 1477, 2176), 210 train data, 91 val data : with window size 15
# samodra A Width: 2162, Height: 1463, feature block (11, 1507, 2206), 210 train data, 91 val data : with window size 15
# with lr 0.001, batch size 128, epochs 500, window size 45, StepLR constant got > 80% val accuracy

# samodra B Width: 1821, Height: 3253, feature block (11, 3267, 1835), 495 train data, 201 val data : with window size 15
# samodra B Width: 1821, Height: 3253, feature block (11, 3297, 1865), 495 train data, 201 val data : with window size 45


