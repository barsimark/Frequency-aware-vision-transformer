import numpy as np
import torch
import pywt
from pytorch_wavelets import DWTForward
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torch.utils.data import Dataset

"""
Dataset for processing OFormer numpy files
"""
class CustomDataset(Dataset):
  def __init__(self, filepath) -> None:
     self.data = np.load(filepath)
     self.data = self.data[:,:,:,:]
     print(self.data.shape)

  def __len__(self):
    return (self.data.shape[0] - 1) * self.data.shape[-1]

  def __getitem__(self, index):
    seq_id = index // (self.data.shape[0] - 1)
    item_id = index % (self.data.shape[0] - 1)
    image = np.expand_dims(self.data[item_id,:,:,seq_id], axis=0)
    target = np.expand_dims(self.data[item_id+1,:,:,seq_id], axis=0)
    return image, target

"""
Original implementation: https://www.geeksforgeeks.org/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/
"""
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
        
"""
Original implementation: https://github.com/gegewen/ufno/blob/main/lploss.py
"""
class LpLoss(object):
    def __init__(self, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    
class FftLpLoss:
    """
    loss function in Fourier space

    June 2022, F.Alesiani
    """

    def __init__(self, p=2, reduction="mean"):
        super().__init__()
        # Dimension and Lp-norm type are positive
        assert p > 0
        self.p = p
        self.reduction = reduction

    def __call__(self, x, y, flow=None, fhigh=None, eps=1e-20):
        num_examples = x.size()[0]
        others_dims = x.shape[1:]
        dims = list(range(1, len(x.shape)))
        xf = torch.fft.fftn(x, dim=dims)
        yf = torch.fft.fftn(y, dim=dims)
        if flow is None:
            flow = 0
        if fhigh is None:
            fhigh = np.max(xf.shape[1:])

        if len(others_dims) == 1:
            xf = xf[:, flow:fhigh]
            yf = yf[:, flow:fhigh]
        if len(others_dims) == 2:
            xf = xf[:, flow:fhigh, flow:fhigh]
            yf = yf[:, flow:fhigh, flow:fhigh]
        if len(others_dims) == 3:
            xf = xf[:, flow:fhigh, flow:fhigh, flow:fhigh]
            yf = yf[:, flow:fhigh, flow:fhigh, flow:fhigh]
        if len(others_dims) == 4:
            xf = xf[:, flow:fhigh, flow:fhigh, flow:fhigh, flow:fhigh]
            yf = yf[:, flow:fhigh, flow:fhigh, flow:fhigh, flow:fhigh]

        _diff = xf - yf.reshape(xf.shape)
        _diff = torch.norm(_diff.reshape(num_examples, -1), self.p, 1)
        _norm = eps + torch.norm(yf.reshape(num_examples, -1), self.p, 1)

        if self.reduction in ["mean"]:
            return torch.mean(_diff / _norm)
        if self.reduction in ["sum"]:
            return torch.sum(_diff / _norm)
        return _diff / _norm
    
"""
Wavelet-based loss function for considering high-frequencies
"""
class HighFrequencyLPLoss(object):
    def __init__(self, size_average=True, reduction=True, p=2, weight_hf=1.0, weight_lp=1.0):
        """
        Parameters:
            size_average (bool): Whether to average the loss over the batch.
            reduction (bool): Whether to reduce the loss to a scalar.
            p (int): The norm degree for the Lp loss.
            weight_hf (float): Weight for the high-frequency loss.
            weight_lp (float): Weight for the Lp loss.
        """
        super(HighFrequencyLPLoss, self).__init__()
        self.size_average = size_average
        self.reduction = reduction
        self.p = p
        self.weight_hf = weight_hf
        self.weight_lp = weight_lp

    def wavelet_high_frequency(self, image):
        xfm = DWTForward(wave='haar')
        aprox, details = xfm(image)
        return details[0][:,:,0,:,:], details[0][:,:,1,:,:], details[0][:,:,2,:,:]

    def lp_loss(self, x, y):
        return torch.mean(torch.abs(x - y) ** self.p)
    
    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms
        
    def compute_loss(self, x, y):
        batch_size = x.shape[0]
        lp_loss = self.lp_loss(x, y)

        LH1, HL1, HH1 = self.wavelet_high_frequency(x)
        LH2, HL2, HH2 = self.wavelet_high_frequency(y)
        lh_ssim = StructuralSimilarityIndexMeasure(data_range=LH2.max() - LH2.min())
        hl_ssim = StructuralSimilarityIndexMeasure(data_range=HL2.max() - HL2.min())
        hh_ssim = StructuralSimilarityIndexMeasure(data_range=HH2.max() - HH2.min())
        lh_loss = 1 - lh_ssim(LH1, LH2)
        hl_loss = 1 - hl_ssim(HL1, HL2)
        hh_loss = 1 - hh_ssim(HH1, HH2)
        hf_loss = torch.mean(lh_loss + hl_loss + hh_loss)
            
        if self.size_average:
            hf_loss /= batch_size
            lp_loss /= batch_size

        combined_loss = self.weight_hf * hf_loss + self.weight_lp * lp_loss
        return combined_loss

    def __call__(self, x, y):
        """
        Callable method for computing loss between x and y.

        Parameters:
            x (np.ndarray): Predicted image or batch of images.
            y (np.ndarray): Target image or batch of images.

        Returns:
            float: Combined loss value.
        """
        return self.compute_loss(x, y)
