import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch import optim
import torch.fft as fft

from einops import rearrange

class PatchEncoding(nn.Module):
  """
  Image to patches of size embed dim
  Based on: https://github.com/jhagnberger/vcnef/tree/main
  Input dims: [batch size, input channels, input size, input size]
  Output dims: [batch size, patch number, embedded dims]
  """
  def __init__(self, patch_size, stride, in_chans, embed_dim):
    super().__init__()
    self.conv = nn.Sequential(nn.Conv2d(in_chans, embed_dim // 2, kernel_size=patch_size // 2, stride=stride, bias=True),
                              nn.GELU(),
                              nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=2, stride=2, bias=True),
                              nn.Flatten(start_dim=-2))
    
  def forward(self, x):
    x = self.conv(x)
    x = x.transpose(1, 2)
    return x

class PatchDecoding(nn.Module):
    """
    Patches of size embed dim to image
    Based on: https://github.com/jhagnberger/vcnef/tree/main
    Input size: [batch size, patch number, embedded dims]
    Output size: [batch size, output channel, output size, output size]
    """
    def __init__(self, patch_size, stride, channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.conv = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=patch_size // 2, stride=stride, bias=True),
                                  nn.GELU(),
                                  nn.ConvTranspose2d(embed_dim // 2, channels, kernel_size=2, stride=2, bias=True))

    def forward(self, x, height, width):
      dim_h = (height - self.patch_size) // (self.stride * 2) + 1
      dim_w = (width - self.patch_size) // (self.stride * 2) + 1
      unflat = nn.Unflatten(dim=-1, unflattened_size=(dim_h, dim_w))
      x = x.transpose(1, 2)
      x = unflat(x)
      x = self.conv(x)
      return x
  
class PatchDecodingWithConvolution(nn.Module):
    """
    Patches of size embed dim to image
    With Conv2D, ReLU, Conv2D at the end
    Input size: [batch size, patch number, embedded dims]
    Output size: [batch size, output channel, output size, output size]
    """
    def __init__(self, patch_size, stride, channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.conv = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=patch_size // 2, stride=stride, bias=True),
                                  nn.GELU(),
                                  nn.ConvTranspose2d(embed_dim // 2, channels, kernel_size=2, stride=2, bias=True))
        self.cnn = nn.Sequential(nn.Conv2d(channels, 64, kernel_size=3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(64, channels, kernel_size=1, padding=0))

    def forward(self, x, height, width):
      dim_h = (height - self.patch_size) // (self.stride * 2) + 1
      dim_w = (width - self.patch_size) // (self.stride * 2) + 1
      unflat = nn.Unflatten(dim=-1, unflattened_size=(dim_h, dim_w))
      x = x.transpose(1, 2)
      x = unflat(x)
      x = self.conv(x)
      x = self.cnn(x)
      return x
  
class PatchFFT(nn.Module):
  def __init__(self, embed_size, ch, patch_size):
      super().__init__()
      self.patch_size = patch_size
      self.ch = ch
      
      self.encode_lin = nn.Linear(embed_size, ch * patch_size * patch_size)
      self.filter = nn.Parameter(torch.randn(1, 1, 1, patch_size, patch_size, dtype=torch.cfloat))
      self.decode_lin = nn.Linear(ch * patch_size * patch_size, embed_size)
      
  def forward(self, x):
    x = self.encode_lin(x)
    b,p,chw = x.shape
    x = x.view(b, p, self.ch, self.patch_size, self.patch_size)
      
    x_fft = fft.fft2(x, dim=(-2, -1))
    result_fft = x_fft * self.filter
    result = fft.ifft2(result_fft, dim=(-2, -1))
    x = abs(result)
    
    b,p,c,h,w = x.shape
    x = x.view(b, p, c*h*w)
    return self.decode_lin(x)
  
class TransformerEncoder(nn.Module):
    def __init__(self, ch, patch_size,  emb_dim=128, n_heads=4, hidden_dim=256, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, dropout=dropout)
        self.q = nn.Linear(emb_dim, emb_dim)
        self.k = nn.Linear(emb_dim, emb_dim)
        self.v = nn.Linear(emb_dim, emb_dim)

        self.norm2 = nn.LayerNorm(emb_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )
        
        self.patchfft = PatchFFT(emb_dim, ch, patch_size)

    def forward(self, x, context=None):
        # Cross-Attention block with skip connection
        residual = x
        x = self.norm1(x)
        # If there is no context for cross-attention, use self-attention
        if not context:
            context = x
        q = self.q(x)
        k = self.k(context)
        v = self.v(context)
        attn_output, _ = self.attention(q, k, v)
        x = attn_output + residual

        # Feedforward block with skip connection
        residual = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = x + residual
        
        # FFT block with skip connection
        residual = x
        x = self.patchfft(x)
        x = x + residual

        return x
      
class VisionTransformer(nn.Module):
  def __init__(self, ch=3, img_size=256, patch_size=16, stride=8, emb_dim=128,
                n_layers=10, dropout=0.1, heads=4):
    super(VisionTransformer, self).__init__()

    self.channels = ch
    self.height = img_size
    self.width = img_size
    self.patch_size = patch_size
    self.n_layers = n_layers

    # Patch Encoding
    self.patch_embedding = PatchEncoding(in_chans=ch, patch_size=patch_size, stride=stride, embed_dim=emb_dim)

    # Transformer
    self.layers = nn.ModuleList([])
    for _ in range(n_layers):
        transformer_block = nn.Sequential(
            TransformerEncoder(ch, patch_size, emb_dim, n_heads=heads, dropout=dropout)    
        )
        self.layers.append(transformer_block)

    # Patch Decoding
    self.decoder = PatchDecodingWithConvolution(patch_size=patch_size, stride=stride, channels=ch, embed_dim=emb_dim)

  def forward(self, img):
    _,_,h,w = img.shape
    x = self.patch_embedding(img)

    # Transformer layers
    for i in range(self.n_layers):
        x = self.layers[i](x)

    # Output the decoded image
    x = self.decoder(x, height=h, width=w)
    return x

if __name__ == "__main__":
  data = torch.ones((1,1,64,64))
  model = VisionTransformer(1, 64, 8, 4, 512, 10, 0.15, 4)
  print(model(data).shape)