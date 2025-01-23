import matplotlib.pyplot as plt
import numpy as np
import pywt
from skimage.metrics import structural_similarity as ssim


def plot_radial_frequency_spectrum(image1, image2, ylimit=None):
    def compute_radial_spectrum(image):
        f_transform = np.fft.fft2(image)
        f_shifted = np.fft.fftshift(f_transform)
        
        magnitude_spectrum = np.abs(f_shifted)
        
        rows, cols = image.shape
        x_center, y_center = cols // 2, rows // 2
        
        radius = np.sqrt((np.indices((rows, cols))[0] - y_center)**2 + (np.indices((rows, cols))[1] - x_center)**2)
        
        radius = radius.flatten()
        magnitude_spectrum = magnitude_spectrum.flatten()
        
        radius_sort_idx = np.argsort(radius)
        radius_sorted = radius[radius_sort_idx]
        magnitude_sorted = magnitude_spectrum[radius_sort_idx]
        
        max_radius = np.max(radius_sorted)
        num_bins = int(max_radius)
        bin_means = np.zeros(num_bins)
        
        for r in range(num_bins):
            mask = (radius_sorted >= r) & (radius_sorted < r + 1)
            if np.sum(mask) > 0:
                bin_means[r] = np.mean(magnitude_sorted[mask])
        
        return bin_means

    spectrum1 = compute_radial_spectrum(image1)
    spectrum2 = compute_radial_spectrum(image2)
    
    plt.figure(figsize=(8, 6))
    
    if ylimit:
        ax = plt.gca()
        ax.set_ylim(ylimit)

    plt.plot(np.arange(len(spectrum1)), spectrum1, color='blue', label='Ground truth')
    
    plt.plot(np.arange(len(spectrum2)), spectrum2, color='red', label='Prediction')

    plt.xlabel('Wavelength (Spatial Frequency)')
    plt.ylabel('Amplitude (Magnitude Spectrum)')
    plt.title('Spatial Frequency Spectrum')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def compare_images(image1: np.ndarray, image2: np.ndarray):
    difference = np.abs(image1.astype(np.float32) - image2.astype(np.float32))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Image 1")
    plt.imshow(image1)
    plt.subplot(1, 3, 2)
    plt.title("Image 2")
    plt.imshow(image2)
    plt.subplot(1, 3, 3)
    plt.title("Difference")
    plt.imshow(difference)
    plt.tight_layout()
    plt.show()
    
def wavelet_high_frequency(image):
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs    
    return LH, HL, HH

def compare_high_frequencies(image1, image2):
    LH1, HL1, HH1 = wavelet_high_frequency(image1)
    LH2, HL2, HH2 = wavelet_high_frequency(image2)

    similarities = {
        "LH_similarity": ssim(LH1, LH2, data_range=LH2.max() - LH2.min()), # horizontal
        "HL_similarity": ssim(HL1, HL2, data_range=HL2.max() - HL2.min()), # vertical
        "HH_similarity": ssim(HH1, HH2, data_range=HH2.max() - HH2.min())  # diagonal
    }

    return similarities

def rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))

def normalized_rmse(pred, true):
    norm_factor = np.max(true) - np.min(true)
    return rmse(pred, true) / norm_factor

def max_error(pred, true):
    return np.max(np.abs(pred - true))

def rmse_boundaries(pred, true, margin=5):
    boundary_mask = np.zeros_like(true, dtype=bool)
    boundary_mask[:margin, :] = True 
    boundary_mask[-margin:, :] = True  
    boundary_mask[:, :margin] = True  
    boundary_mask[:, -margin:] = True  
    
    pred_boundary = pred[boundary_mask]
    true_boundary = true[boundary_mask]
    return rmse(pred_boundary, true_boundary)

def conserved_variables_error(pred, true):
    pred_integral = np.sum(pred)
    true_integral = np.sum(true)
    return np.abs(pred_integral - true_integral)

def rmse_fourier(pred, true):
    pred_fft = np.fft.fft2(pred)
    true_fft = np.fft.fft2(true)
    return rmse(np.abs(pred_fft), np.abs(true_fft))

def temporal_sensitivity_error(pred_series, true_series):
    pred_grad = np.gradient(pred_series, axis=0)
    true_grad = np.gradient(true_series, axis=0)
    return rmse(pred_grad, true_grad)
    
    
if __name__ == "__main__":
    data = np.load('data/fno_ns_Re1000_N1200_T20.npy')
    gt = data[0,:,:,0]
    noise = data[0,:,:,0] + np.random.normal(0.1, 0.2, (64,64))
    LH, HL, HH = wavelet_high_frequency(gt)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.title('Original')
    plt.imshow(gt)
    plt.subplot(2, 2, 2)
    plt.title('Horizontal details')
    plt.imshow(LH)
    plt.subplot(2, 2, 3)
    plt.title('Vertical details')
    plt.imshow(HL)
    plt.subplot(2, 2, 4)
    plt.title('Diagonal details')
    plt.imshow(HH)
    
    plt.show()
    print(compare_high_frequencies(gt, gt))
    print(compare_high_frequencies(gt, noise))