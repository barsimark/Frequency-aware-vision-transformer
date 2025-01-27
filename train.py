import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch import optim
import torch.fft as fft

from einops import rearrange

def train(model, train_loader, test_loader, num_epochs=20, lr=0.01, loss_function=nn.MSELoss(), device="cuda", early_stop=None):
  print('Start training...')
  model = model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr)

  training_losses = []
  testing_losses = []
  for epoch in range(num_epochs):
    model.train()
    training_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
      print(f'batch: {batch_idx}')
      inputs = inputs.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = loss_function(outputs, labels)
      loss.backward()
      optimizer.step()
      training_loss += loss.item()
    training_loss /= len(train_loader)
    training_losses.append(training_loss)
    print(f"Epoch {epoch + 1} training loss: {training_loss}")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
      for batch_idx, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
    test_loss /= len(test_loader)
    testing_losses.append(test_loss)
    print(f"Epoch {epoch + 1} testing loss: {test_loss}")
    
    torch.save(model.state_dict(), 'latest.pt')
    
    if early_stop:
      early_stop(test_loss, model)
      if early_stop.early_stop:
          print("Early stopping")
          break

  early_stop.load_best_model(model)
  print('Training finished')
  return training_losses, testing_losses