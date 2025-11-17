from osgeo import gdal
import numpy as np
import random
import os
import pandas as pd

from pipeline.reader import read_data_from_tif

def shuffle_image_label(images, labels):
    """
    data shuffle
    """
    randnum = random.randint(0, len(images))
    random.seed(randnum)
    random.shuffle(images)
    random.seed(randnum)
    random.shuffle(labels)
    return images, labels

def get_CNN_data(feature_block, label_raster, window_size):
    """
    Creating a CNN dataset
    """
    n = window_size // 2
    train_imgs, train_labels = [], []
    val_imgs, val_labels = [], []

    # train data (label 0 dan 2)
    train_longsor_y, train_longsor_x = np.where(label_raster == 0)
    for y, x in zip(train_longsor_y, train_longsor_x):
        patch = feature_block[:, y-n:y+n+1, x-n:x+n+1]
        train_imgs.append(patch)
        train_labels.append(0) # Label: landslide

    train_aman_y, train_aman_x = np.where(label_raster == 2)
    for y, x in zip(train_aman_y, train_aman_x):
        patch = feature_block[:, y-n:y+n+1, x-n:x+n+1]
        train_imgs.append(patch)
        train_labels.append(1) # Label: no landslides

    # Val data (label 1 dan 3)
    val_longsor_y, val_longsor_x = np.where(label_raster == 1)
    for y, x in zip(val_longsor_y, val_longsor_x):
        patch = feature_block[:, y-n:y+n+1, x-n:x+n+1]
        val_imgs.append(patch)
        val_labels.append(0) # Label: landslide
        
    val_aman_y, val_aman_x = np.where(label_raster == 3)
    for y, x in zip(val_aman_y, val_aman_x):
        patch = feature_block[:, y-n:y+n+1, x-n:x+n+1]
        val_imgs.append(patch)
        val_labels.append(1) # Label: no landslides
        
    # Acak dan kembalikan sebagai NumPy array
    train_imgs, train_labels = shuffle_image_label(train_imgs, train_labels)
    val_imgs, val_labels = shuffle_image_label(val_imgs, val_labels)

    return (np.array(train_imgs), np.array(train_labels).reshape(-1, 1), np.array(val_imgs), np.array(val_labels).reshape(-1, 1))

def get_ML_data(tif_paths, label_path):
    """
    Creating a ML dataset
    """
    data = []
    data_name = []
    tif = gdal.Open(label_path)
    w, h = tif.RasterXSize, tif.RasterYSize
    label = np.array(tif.ReadAsArray(0, 0, w, h).astype(np.float32))
    for tif_data in os.listdir(tif_paths):
        img = read_data_from_tif(os.path.join(tif_paths, tif_data))
        data_name.append(tif_data.split('.')[0])
        data.append(img)
    data_name.append('label')   
    data = np.array(data)

    # 0 longsor; 1 nls
    train_data = []
    mask_0 = label == 0
    i_indices_0, j_indices_0 = np.where(mask_0)
    mask_1 = label == 2
    i_indices_1, j_indices_1 = np.where(mask_1)
    for i, j in zip(i_indices_0, j_indices_0):
        train_data.append((data[:,i,j],0))
    for i, j in zip(i_indices_1, j_indices_1):
        train_data.append((data[:,i,j],1))
    
    val_data = []
    mask_2 = label == 1
    i_indices_2, j_indices_2 = np.where(mask_2)
    mask_3 = label == 3
    i_indices_3, j_indices_3 = np.where(mask_3)
    for i, j in zip(i_indices_2, j_indices_2):
        val_data.append((data[:,i,j],0))
    for i, j in zip(i_indices_3, j_indices_3):
        val_data.append((data[:,i,j],1))
    
    train_imgs = [item[0] for item in train_data]
    train_labels = [item[1] for item in train_data]
    val_imgs = [item[0] for item in val_data]
    val_labels = [item[1] for item in val_data]
    
    train_imgs, train_labels = shuffle_image_label(train_imgs, train_labels)
    # val_imgs, val_labels = shuffle_image_label(val_imgs, val_labels) data validasi tidak perlu diacak
    
    train_imgs, val_imgs = np.array(train_imgs),np.array(val_imgs)
    train_labels, val_labels = np.array(train_labels).reshape((-1,1)), np.array(val_labels).reshape((-1,1))
    print(train_imgs.shape,val_imgs.shape, train_labels.shape,val_labels.shape )
    train_df = pd.DataFrame(np.concatenate((train_imgs, train_labels), axis=1), columns=data_name)
    val_df = pd.DataFrame(np.concatenate((val_imgs, val_labels), axis=1), columns=data_name)
    return train_df, val_df