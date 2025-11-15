from osgeo import gdal
import numpy as np
import os
import sys

def read_data_from_tif(tif_path):
    """
    Read impact factor data and convert to nparray
    """
    tif = gdal.Open(tif_path)
    w, h = tif.RasterXSize, tif.RasterYSize
    img = np.array(tif.ReadAsArray(0, 0, w, h).astype(np.float32))
    return img

def validate_consistency(feature_dir, label_path):
    """Validate all .tif files to have the same geotransform and dimensions"""
    label_tif = gdal.Open(label_path)
    label_transform = label_tif.GetGeoTransform()
    label_dims = (label_tif.RasterXSize, label_tif.RasterYSize)
    
    for tif_name in os.listdir(feature_dir):
        if tif_name.endswith('.tif'):
            feature_path = os.path.join(feature_dir, tif_name)
            feature_tif = gdal.Open(feature_path)
            feature_transform = feature_tif.GetGeoTransform()
            feature_dims = (feature_tif.RasterXSize, feature_tif.RasterYSize)
            
            if label_transform != feature_transform or label_dims != feature_dims:
                print(f"ERROR: File '{tif_name}' inconsistent with the label")
                sys.exit()
    print("Data consistency check passed.")
    return True
    