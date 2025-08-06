#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:01:26 2024

@author: elaabar
"""

import torch
import torch.nn as nn

import sys
sys.path.append('.')

from iriscc.models.denoising_unet import CUNet

class CDDPM(nn.Module):
    """
    Implementation of the Conditioned DDPM model.

    Reference:
        Ho, J., Chen, X., Srinivas, A., Duan, Y., Abbeel, P., & Finn, C. (2020).
        Denoising Diffusion Probabilistic Models. arXiv preprint arXiv:2006.11239.
    Methods:
        forward(x0, t, eta): Forward pass of the DDPM model.
        backward(x, t, conditioning_image): Backward pass of the DDPM model.
    """
    def __init__(self, n_steps=200, min_beta=1e-4, max_beta=0.02, encode_conditioning_image=False, in_ch=1):
        """
        Initialize the DDPM model.

        Parameters:
            network (nn.Module): The neural network used for estimation.
            n_steps (int): Number of steps in the DDPM algorithm.
            min_beta (float): First value for beta in the DDPM algorithm.
            max_beta (float): Last value for beta in the DDPM algorithm.
        """
        super(CDDPM, self).__init__()
        #print('  >> <class CDDPM> : __init__  ')

        # Store configuration parameters
        self.n_steps = n_steps

        # Move the neural network to the specified device
        self.network = CUNet(n_steps=self.n_steps, 
                             time_emb_dim = 100, 
                             in_channels=in_ch, 
                             out_channels=1, 
                             init_features=32)

        # Generate beta values between min_beta and max_beta
        self.betas = torch.linspace(min_beta, max_beta, n_steps)

        # Calculate alpha values (1 - beta) and alpha bar values
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(n_steps)])

        self.encode_conditioning_image = encode_conditioning_image
        if encode_conditioning_image:
            self.conditioning_encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=in_ch, out_channels=10, kernel_size=3, padding=1),
                    torch.nn.Tanh(),
                    torch.nn.Conv2d(in_channels=10, out_channels=1, kernel_size=1)
                )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x0: torch.Tensor, t: int, eta: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the DDPM model.

        Parameters:
            x0 (torch.Tensor): Input image tensor.
            t (int): Current timestep in the DDPM algorithm.
            eta (torch.Tensor): Optional noise tensor (default is None).

        Returns:
            torch.Tensor: Noisy image tensor.
        """
        # Calculate alpha bar for the current timestep
        #print('  >> <class CDDPM> : forward  ')
        a_bar = self.alpha_bars[t].reshape(x0.shape[0], 1, 1, 1)

        # If noise is not provided, generate random noise
        if eta is None:
            eta = torch.randn(*x0.shape, device=self.device)

        # Add noise to the input image based on the calculated alpha bar
        noisy = a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * eta
        return noisy

    def backward(self, x: torch.Tensor, t: int, conditioning_image: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the DDPM model.

        Parameters:
            x (torch.Tensor): Input image tensor.
            t (int): Current time step in the DDPM algorithm.
            conditioning_image (torch.Tensor): Conditioning image tensor.


        Returns:
            torch.Tensor: Estimated noise tensor.
        """        
        # Run the input image through the neural network for noise estimation
        if self.encode_conditioning_image:                
            conditioning_image = self.conditioning_encoder(conditioning_image)
        return self.network(x, t, conditioning_image)
    
    def set_device(self, device: str) -> None:
        """
        Set the device for tensor computations.

        Parameters:
            device (str): Device for tensor computations.
        """
        self.device = device
        #print('  >> <class CDDPM> : set_device  ')
        self.network.to(device)
        self.betas = self.betas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)

    def sampling(self, start_t: int, conditioning_image: torch.Tensor, eta: torch.Tensor = None) -> torch.Tensor:
        """
        Sampling from the DDPM model.

        Parameters:
            start_t (int): Start time step in the DDPM algorithm.
            conditioning_image (torch.Tensor): Conditioning image tensor.
            eta (torch.Tensor): Optional noise tensor (default is None).

        Returns:
            torch.Tensor: Sampled image tensor.
        """
        # If noise is not provided, generate random noise
        if eta is None:
            eta = torch.randn((1,1,conditioning_image.shape[2], conditioning_image.shape[3]), device=self.device)
        x = eta
        for idx, t in enumerate(list(range(start_t))[::-1]):
            time_tensor = (t * torch.ones(x.shape[0], 1)).to(self.device).long()
            #time_tensor = (torch.ones(1, 1) * t).long()
            # Estimating noise to be removed
            eta_theta = self.backward(x, time_tensor, conditioning_image)

            alpha_t = self.alphas[t]
            alpha_t_bar = self.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)
            if t > 0:
                z = torch.randn(*x.shape).to(self.device)

                # Option 1: sigma_t squared = beta_t
                beta_t = self.betas[t]
                sigma_t = beta_t.sqrt()

                # Adding some noise to the image
                # this is inspired from langevin dynamics
                x = x + sigma_t * z
        # Returning the final image
        return x
    


if __name__=='__main__':

    cddpm = CDDPM(n_steps=5, 
                  min_beta=1e-4, 
                  max_beta=0.02, 
                  encode_conditioning_image=False, 
                  in_ch=3)

