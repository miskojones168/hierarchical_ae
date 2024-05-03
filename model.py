import torch
import torch.nn as nn
import numpy as np

class Autoencoder(nn.Module):
    '''
        Custom Autoencoder built using CNNs
    '''
    def __init__(self, img_size, img_channels=1, device='cuda') -> None:
        super(Autoencoder, self).__init__()

        self.shape = [64, int(img_size/4), int(img_size/4)]

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=32,
                      kernel_size=3, stride=2, padding=1, device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=2, padding=1, device=device),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=np.prod(self.shape), out_features=10, device=device)
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=10, out_features=np.prod(self.shape), device=device),
            nn.Unflatten(dim=1, unflattened_size=self.shape),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32,
                kernel_size=4, stride=2, padding=1, device=device
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=img_channels,
                kernel_size=4, stride=2, padding=1, device=device
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    pass

    