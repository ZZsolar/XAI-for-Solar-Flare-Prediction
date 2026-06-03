import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from astropy.io import fits
from scipy.ndimage import zoom

# Path to pre-built 3D .npy samples (set for your environment)
npy_data3d_path = "/data1/data_zz/data_work1/cnn_data/"


class Npy3dDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):

        file_name = self.data_frame.iloc[idx]['file_name']
        label = self.data_frame.iloc[idx]['label']

        file_path = npy_data3d_path + f'/{file_name}.data.npy'
        data = np.load(file_path)

        if self.transform:
            data = self.transform(data)

        return data, label

class ToTensor_3d:
    def __call__(self, sample):
        tensor_data = torch.tensor(sample, dtype=torch.float32)
        return tensor_data.permute(2, 0, 1)

transform_train_3d = transforms.Compose([
    ToTensor_3d(),
    transforms.RandomRotation(360),
])

transform_test_3d = transforms.Compose([
    ToTensor_3d(),
])


def calculate_tss(confusion_mat):
    """
    True Skill Statistic: TSS = Sensitivity + Specificity - 1.
    Sensitivity = TP/(TP+FN), Specificity = TN/(TN+FP).
    """
    tn, fp, fn, tp = confusion_mat.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    tss = sensitivity + specificity - 1
    return tss

def evaluate_model(model, criterion, loader, device):
    """Compute loss, recall, precision, f1, TSS on the given loader."""
    model.eval()

    def evaluate(loader):
        total_loss = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                preds = (outputs > 0.5) * 1

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        confusion_mat = confusion_matrix(all_labels, all_preds)
        metrics = {
            "loss": total_loss / len(loader),
            "recall": recall_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds),
            "f1_score": f1_score(all_labels, all_preds),
            "confusion_matrix": confusion_mat.tolist(),
            "tss": calculate_tss(confusion_mat)
        }
        return metrics

    return evaluate(loader)


def pad_matrix_edge(A):
    x, y = A.shape
    n = max(x, y)
    B = np.zeros((n, n), dtype=A.dtype)
    start_x = (n - x) // 2
    start_y = (n - y) // 2
    B[start_x:start_x + x, start_y:start_y + y] = A
    
    return B

def Resizefits(fits_data):
    h, w = fits_data.shape
    max_dim = max(h, w)
    padded_sample = pad_matrix_edge(fits_data)
    resize_factors = 512 / max_dim
    resized_sample = zoom(padded_sample, (resize_factors, resize_factors), order=1)
    
    return resized_sample

from captum.attr import LayerGradCam


def get_fits_data(file_path):
    with fits.open(file_path) as hdu:
        data = hdu[1].data.astype(np.float32)
    return data

def unpad_matrix_edge(B, original_shape):
    """Extract centered (orig_h, orig_w) region from padded array."""
    orig_h, orig_w = original_shape
    padded_h, padded_w = B.shape
    start_x = (padded_h - orig_h) // 2
    start_y = (padded_w - orig_w) // 2
    unpadded_matrix = B[start_x:start_x + orig_h, start_y:start_y + orig_w]
    return unpadded_matrix


def data_restore(resized_sample, original_shape):
    """Upsample and unpad to original (h, w)."""
    max_dim = max(original_shape)
    resize_factors = resized_sample.shape[0] / max_dim
    unresized_sample = zoom(resized_sample, (1 / resize_factors, 1 / resize_factors), order=1)
    unpadded_sample = unpad_matrix_edge(unresized_sample, original_shape)
    return unpadded_sample


def input_processing(file_data):
    data = Resizefits(file_data)
    return transform_test_3d(data).unsqueeze(0)

def attr_processing(attributions, file_data):
    attributions = attributions.squeeze().cpu().detach().numpy()  
    attributions /= np.max(np.abs(attributions))  
    return data_restore(attributions, file_data.shape)

