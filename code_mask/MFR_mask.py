"""
Generate MFR (Model-Focused Region) masks using Grad-CAM on the trained CNN.
Output: one .attr.npy per sample, used as mask in SHARP_masked.py (--method mfr).
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from scipy.ndimage import zoom
from collections import OrderedDict

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)
from cnn.utils.model import CNN_Model
from captum.attr import LayerGradCam


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM for model evaluation")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], default='cuda:3', 
                        help='Device to run the model on')
    parser.add_argument('--target_class', type=int, default=0, help='Target class for Grad-CAM (default: 0)')
    
    return parser.parse_args()

def read_fits(file):
    data = fits.open(file)[1].data
    return data

def get_data(harpnum, file_name):
    file_path = data_path + f'sharp_{harpnum:05d}/{file_name}'

    bl_data = read_fits(file_path + '.magnetogram.fits')
    br_data = read_fits(file_path + '.Br.fits')
    bp_data = read_fits(file_path + '.Bp.fits')
    bt_data = read_fits(file_path + '.Bt.fits')

    data = np.stack([bl_data, br_data, bp_data, bt_data], axis=0)
    return data
    
def pad_matrix_edge3d(A):
    """Pad 3D array to square (max(h,w)) on spatial dims, centered."""
    c, h, w = A.shape
    n = max(h, w)
    B = np.zeros((c, n, n), dtype=A.dtype)
    start_h = (n - h) // 2
    start_w = (n - w) // 2
    
    B[:, start_h:start_h + h, start_w:start_w + w] = A
    
    return B

def ResizeAndNormalize(fits_data):
    """Pad to square then resize to 512 on longest side (match training)."""
    c, h, w = fits_data.shape
    max_dim = max(h, w)
    padded_sample = pad_matrix_edge3d(fits_data)
    resize_factors = 512 / max_dim
    resized_sample = zoom(padded_sample, (1, resize_factors, resize_factors), order=1)
    normalized_sample = resized_sample
    
    return normalized_sample

def unpad_matrix_edge(B, original_shape):
    orig_h, orig_w = original_shape
    padded_h, padded_w = B.shape
    start_x = (padded_h - orig_h) // 2
    start_y = (padded_w - orig_w) // 2
    unpadded_matrix = B[start_x:start_x + orig_h, start_y:start_y + orig_w]
    
    return unpadded_matrix

def data_restore(resized_sample, original_shape):
    """Upsample and unpad Grad-CAM map back to original (h, w)."""
    max_dim = max(original_shape)
    resize_factors = resized_sample.shape[0] / max_dim
    unresized_sample = zoom(resized_sample, (1 / resize_factors, 1 / resize_factors), order=1)
    unpadded_sample = unpad_matrix_edge(unresized_sample, original_shape)

    return unpadded_sample

args = parse_args()
device = torch.device(args.device)
target_class = args.target_class

# Paths: set to your environment
model_path = "./model/cnn_model.pth"
csv_file = "../data/label.csv"
data_path = "/data_work/data_sharp_cea/"
save_path = "/data_work/mask_data/mfr/"

model = CNN_Model(input_channels=4)
model_state = torch.load(model_path, map_location=device)
# Strip 'module.' prefix when model was saved with DataParallel
new_state_dict = OrderedDict()
for k, v in model_state.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model = model.to(device)
model.eval()

model.to(device)
model.eval()
# Grad-CAM on last residual block's conv for class 0 (positive flare)
layer_gc = LayerGradCam(model, layer=model.res_blocks[-2].conv2)


def main():
    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():
        harpnum = row['HARPNUM']
        file_name = row['file_name']
        fits_data = get_data(harpnum, file_name)
        processed_data = ResizeAndNormalize(fits_data)
        input_data = torch.tensor(processed_data).unsqueeze(0).float().to(device)
        attr_gc = layer_gc.attribute(input_data, target=0)
        attr_gc = attr_gc.cpu().detach().numpy().squeeze().squeeze()
        resized_attr = data_restore(attr_gc, (fits_data.shape[1], fits_data.shape[2]))
        save_file = os.path.join(save_path, f'{file_name}.attr.npy')
        np.save(save_file, resized_attr)
        
        print(f"Processed and saved: {save_file}")

    return None

if __name__ == "__main__":
    main()

