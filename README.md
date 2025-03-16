# Installation Guide

## Prerequisites
Before installing this project, ensure you have the following dependencies installed:
- [Git](https://git-scm.com/)
- [Python](https://www.python.org/) (Recommended: Python 3.8 or later)
- [pip](https://pip.pypa.io/en/stable/)
- [virtualenv](https://virtualenv.pypa.io/en/latest/) (Optional but recommended)
- GPU support (CUDA for PyTorch)

## Clone the Repository
To get started, clone this repository to your local machine:
```sh
git clone https://github.com/barsimark/Frequency-aware-vision-transformer
cd Frequency-aware-vision-transformer
```

## Set Up Virtual Environment (Optional but Recommended)
```sh
python -m venv venv
venv\Scripts\activate
```

## Install Dependencies
```sh
pip install -r requirements.txt
```

# Train model

## Download data

Get fno_ns_Re1000_N1200_T20.npy from https://drive.google.com/drive/folders/1z-0V6NSl2STzrSA6QkzYWOGHSTgiOSYq

## Dependencies

```python
import favit
import utils
import train
```

## Training and test datasets and dataloaders

```python
dataset = utils.CustomDataset('data/fno_ns_Re1000_N1200_T20.npy')

n = len(dataset)
train_size = int(0.75 * n)
test_size = n - train_size
train_set = Subset(dataset, range(0, train_size))
test_set = Subset(dataset, range(train_size, n))

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64)
```

## Train model

```python
model = favit.VisionTransformer(ch=1, img_size=64, patch_size=16, stride=4, emb_dim=512, dropout=0.2, n_layers=6)
train_losses, test_losses = train.train(
    model, 
    train_loader, 
    test_loader, 
    loss_function=utils.HighFrequencyLPLoss(weight_hf=0.2, weight_lp=0.8), 
    num_epochs=150, 
    early_stop=utils.EarlyStopping(patience=150, delta=0),
    lr=0.0005)
torch.save(model.state_dict(), 'best.pt')
```

# Evaluate model

## Import evaluation package

```python
import eval
```

## Load model

```python
model = favit.VisionTransformer(ch=1, img_size=64, patch_size=8, stride=2, emb_dim=512, dropout=0.2, n_layers=6)
model.load_state_dict(torch.load('best.pt', weights_only=True))
model.eval()
model.to('cpu') # if evaluating on cpu
```

## Prediction on the test dataset

```python
input, target = next(iter(test_loader))
input = np.expand_dims(input[0], axis=0)
target = np.array(target[0][0])
pred = model(torch.Tensor(input)).detach().numpy()[0,0,:,:]
```

## Numerical evaluation metrics

```python
print('RMSE', eval.rmse(pred, target))
print('Normalized RMSE', eval.normalized_rmse(pred, target))
print('Max error', eval.max_error(pred, target))
print('Boundary RMSE', eval.rmse_boundaries(pred, target))
print('Conserved variables', eval.conserved_variables_error(pred, target))
print('RMSE in Fourier', eval.rmse_fourier(pred, target))
print('Temporal sensitivity', eval.temporal_sensitivity_error(pred, target))
print('SSIM metrics', eval.compare_high_frequencies(pred, target))
```

## Visual evalution metrics

```python
eval.compare_images(pred, ground) # absolute difference between image

eval.plot_radial_frequency_spectrum(ground, pred) # energy spectrum comparison
```
