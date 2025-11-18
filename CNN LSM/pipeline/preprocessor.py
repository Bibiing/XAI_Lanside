import numpy as np

def normalize_min_max(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normed = (data - min_val) / (max_val - min_val)
    # save normalization parameters
    norm_params = {"min": float(min_val), "max": float(max_val)}
    return normed, norm_params

def apply_padding(data, padding_size, pad_value=0):
    return np.pad(data, ((padding_size, padding_size), (padding_size, padding_size)), 'constant', constant_values=(pad_value, pad_value))