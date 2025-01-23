import numpy as np
import matplotlib.pyplot as plt

def visualize_arrays(ground_truth_file, highlighted_file, other_files):
    ground_truth = np.load(ground_truth_file)
    highlighted = np.load(highlighted_file)
    others = [np.load(file) for file in other_files]
    x = np.arange(len(ground_truth))

    plt.figure(figsize=(10, 6))
    plt.plot(x, ground_truth, color='blue', label='Ground Truth')
    plt.plot(x, highlighted, color='red', label='High frequency loss function')
    
    for other in others:
        plt.plot(x, other, color='gray', alpha=0.5)
    
    plt.ylim(0, 5)
    plt.xlabel('Wavelength')
    plt.ylabel('Amplitude')
    plt.title('Energy spectrum comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

visualize_arrays('image3_losses/original.npy', 
                 'image3_losses/8patch4stride.npy', 
                 ['image3_losses/fftloss.npy', 'image3_losses/l1loss.npy', 'image3_losses/l2loss.npy', ]
                 )