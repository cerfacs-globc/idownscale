'''
U-NET FOR BRAIN MRI
Adapted to diffusion purposes adding time embeddings by Zoé Garcia
'''

import sys
sys.path.append('.')

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

import numpy as np 
from iriscc.transforms import MinMaxNormalisation, LandSeaMask, Pad, FillMissingValue
from iriscc.plotutils import plot_test
from torchvision.transforms import v2

class TimeProcessing(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(TimeProcessing, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

    def forward(self, t):
        return self.fc(t)


class CUNet(nn.Module):

    def __init__(self, n_steps=200, time_emb_dim = 100, in_channels=3, out_channels=1, init_features=32):
        super(CUNet, self).__init__()

        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = self._make_sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        features = init_features
        self.time_fc = TimeProcessing(time_emb_dim, in_channels)
        self.encoder1 = CUNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.time_fc_enc1 = TimeProcessing(time_emb_dim, features)
        self.encoder2 = CUNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.time_fc_enc2 = TimeProcessing(time_emb_dim, features*2)
        self.encoder3 = CUNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.time_fc_enc3 = TimeProcessing(time_emb_dim, features*4)
        self.encoder4 = CUNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.time_fc_enc4 = TimeProcessing(time_emb_dim, features*8)
        
        #self.bottleneck = CUNet._block(features * 4, features * 8, name="bottleneck") ##minMiniCUNet
        
        self.bottleneck = CUNet._block(features * 8, features * 16, name="bottleneck")
        self.time_fc_bottleneck = TimeProcessing(time_emb_dim, features*16)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.time_fc_dec4 = TimeProcessing(time_emb_dim, (features * 8) * 2)
        self.decoder4 = CUNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.time_fc_dec3 = TimeProcessing(time_emb_dim, (features * 4) * 2)
        self.decoder3 = CUNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.time_fc_dec2 = TimeProcessing(time_emb_dim, (features * 2) * 2)
        self.decoder2 = CUNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.time_fc_dec1 = TimeProcessing(time_emb_dim, features * 2)
        self.decoder1 = CUNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x, t, conditionning_image):
        t = self.time_embed(t)
        x = torch.cat((x, conditionning_image), dim=1)

        x = x + self.time_fc(t).reshape(x.size(0), -1, 1, 1)
        enc1 = self.encoder1(x)
        enc1 = enc1 + self.time_fc_enc1(t).reshape(enc1.size(0), -1, 1, 1)
        enc2 = self.encoder2(self.pool1(enc1))
        enc2 = enc2 + self.time_fc_enc2(t).reshape(enc2.size(0), -1, 1, 1)
        enc3 = self.encoder3(self.pool2(enc2))
        enc3 = enc3 + self.time_fc_enc3(t).reshape(enc3.size(0), -1, 1, 1)
        enc4 = self.encoder4(self.pool3(enc3))
        enc4 = enc4 + self.time_fc_enc4(t).reshape(enc4.size(0), -1, 1, 1)

        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = dec4 + self.time_fc_dec4(t).reshape(dec4.size(0), -1, 1, 1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = dec3 + self.time_fc_dec3(t).reshape(dec3.size(0), -1, 1, 1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = dec2 + self.time_fc_dec2(t).reshape(dec2.size(0), -1, 1, 1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = dec1 + self.time_fc_dec1(t).reshape(dec1.size(0), -1, 1, 1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    def _make_sinusoidal_embedding(self, dim_in, dim_out):
        # Returns the standard positional embedding
        embedding = torch.zeros(dim_in, dim_out)
        wk = torch.tensor([1 / 10_000 ** (2 * j / dim_out) for j in range(dim_out)])
        wk = wk.reshape((1, dim_out))
        t = torch.arange(dim_in).reshape((dim_in, 1))
        embedding[:,::2] = torch.sin(t * wk[:,::2])
        embedding[:,1::2] = torch.cos(t * wk[:,::2])
        return embedding

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

if __name__=='__main__':
    model = CUNet(n_steps=100, 
                  time_emb_dim = 100, 
                  in_channels=3, # 2 conditionning channels and 1 noise channel
                  out_channels=1, 
                  init_features=32)
    model = model.float()

    data = dict(np.load('/scratch/globc/garcia/datasets/dataset_exp3_30y/sample_20040101.npz', allow_pickle=True))
    conditionning_image, y = data['x'], data['y']
    transforms = v2.Compose([
            MinMaxNormalisation(Path('/scratch/globc/garcia/datasets/dataset_exp3_30y')), 
            LandSeaMask('france', 0),
            FillMissingValue(0),
            Pad(0)
            ])
    
    conditionning_image, y = transforms((conditionning_image, y))
    conditionning_image = np.expand_dims(conditionning_image, axis=0)
    conditionning_image = torch.tensor(conditionning_image)
    x = torch.randn(1, 1, 160, 160)
    t = 1
    time_tensor = (torch.ones(1, 1) * t).long()

    print(x.shape, conditionning_image.shape, t)

    y_hat = model(x.float(), time_tensor, conditionning_image.float())
    print(y_hat)
    plot_test(y_hat.detach().numpy()[0, 0,:,:], 'title', '/gpfs-calypso/scratch/globc/garcia/graph/test4.png')
    

