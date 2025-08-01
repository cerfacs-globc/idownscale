"""
Useful functions for diffusion models.

date : 16/07/2025
Rachid Elmontassir script modified by Zoé Garcia
"""


import sys
sys.path.append('.')

from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np
import tqdm
from torchvision.transforms import v2

from iriscc.models.cddpm import CDDPM
from iriscc.transforms import UnPad, MinMaxNormalisation, FillMissingValue, LandSeaMask, Pad
from iriscc.dataloaders import get_dataloaders
from iriscc.plotutils import plot_test
from iriscc.settings import GRAPHS_DIR

def show_forward(ddpm, loader, device, n_images=4, n_noise_steps=5):
    """
    Show the forward process of a DDPM model.

    Parameters:
        ddpm (DDPM): Instance of DDPM model.
        loader (DataLoader): DataLoader containing batches of images.
        device (str): Device for tensor computations.
        n_images (int): Number of images to show (default to 4).
        n_noise_steps (int): number of timesteps to consider (default to 5).

    Note:
        This function visualizes the forward process, showing original images and noisy images at different steps
        in the DDPM algorithm.
    """
    # Iterate over batches in the DataLoader
    for condi, images in loader:
        imgs = images[:n_images]  # Extract the input images from the batch
        mask = (imgs == 0)

        percentages = np.linspace(0,1,n_noise_steps + 1)[1:]
        noisy_images = torch.ones(imgs.shape + (n_noise_steps + 1,))
        noisy_images[...,0] = imgs

        unpad = UnPad(initial_size=[134,143])

        # Iterate over different percentages of steps in the DDPM algorithm
        for i, percent in enumerate(percentages):
            # Compute the corresponding timesteps for the DDPM algorithm based on the given percentage
            timesteps = [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))]
            noisy_images[...,i+1] = ddpm(imgs, timesteps)
            noisy_images[mask] = torch.nan


        # Show the noisy images generated by the DDPM model for different timesteps
        rows = n_images
        cols = n_noise_steps+1
        fig, axes = plt.subplots(rows, cols, figsize=(cols*1, rows*1))

        # Plot each image
        for i in range(rows):
            for ts in range(cols):
                new_noisy_images = unpad(noisy_images[i, ..., ts])
                axes[i,ts].imshow(np.flipud(new_noisy_images[0,...]),cmap='winter')
                axes[i,ts].axis("off")  # Turn off axis labels
                if i==0 and ts>0:
                    axes[i,ts].set_title(f"{int(percentages[ts-1] * ddpm.n_steps)} steps ")
        axes[0,0].set_title(f"Original")

        # Set the title
        fig.suptitle("DDPM forward steps", fontsize=16)

        # Adjust layout and show the figure
        plt.tight_layout()
        plt.savefig(GRAPHS_DIR/'test.png')
        # Break after the first batch to avoid processing the entire dataset
        break


def generate(cddpm, input_data, n_samples=1, neighbours=False, std=1e-1, start_t: int=None, clamp=None, device='cpu'):
    """
    Generates new images using the CDDPM model.

    Parameters:
        input_data (torch.Tensor): Input images.
        output_shape (tuple): Shape of the output images.
        neighbours (bool): If True, generate samples from neighbouring points.
        std (float): Standard deviation for noise.
    
    Returns:
        torch.Tensor: Generated samples.
    """
    n_steps = cddpm.n_steps
    inputs = input_data.to(device)
    _, _, h, w = inputs.shape
    b, c = n_samples, 1 #only one noise channel
    print(b)
    inputs = inputs.expand(b, -1, -1, -1)
    print(inputs.shape)
    if start_t is None:
        if neighbours:
            center = torch.randn(1, c, h, w).to(device)
            x = center + std * torch.randn(b, c, h, w).to(device)
        else:
            x = torch.randn(b, c, h, w).to(device)
    else:
        # start from noised input
        t_b = torch.ones(b, 1).long().to(device) * start_t
        x = cddpm.forward(inputs.to(device), t_b)
        print(x.shape)

    # Generate images
    n_steps = n_steps if start_t is None else start_t
    intermediate_images = [x.cpu().detach().numpy()]
    for idx, t in tqdm.tqdm(enumerate(reversed(range(n_steps))), total=n_steps, desc="Sampling ..", colour="#ffff00"):
        time_tensor = torch.ones(b, 1).long().to(device) * t
        eta_theta = cddpm.backward(x, time_tensor, inputs)
        # detaching the eta_theta to make sure that the gradient is not propagated through the DDPM backward pass
        eta_theta = eta_theta.detach()

        alpha_t = cddpm.alphas[t]
        alpha_t_bar = cddpm.alpha_bars[t]
        x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

        if t > 0:
            z = torch.randn(b, c, h, w).to(device)
            beta_t = cddpm.betas[t]
            sigma_t = beta_t.sqrt()
            x += sigma_t * z
            # normalise x
            #x = (x - x.mean()) / x.std()
            # clamp x
            #x = torch.clamp(x, -1 - 1/t, 1 + 1/t)
            if clamp is not None:
                x = torch.clamp(x, -clamp(t), clamp(t))
        intermediate_images.append(x.cpu().detach().numpy())

    return intermediate_images

if __name__ == '__main__':

    cddpm = CDDPM(n_steps=100, 
                    min_beta=1e-4, 
                    max_beta=0.1, 
                    encode_conditioning_image=False, 
                    in_ch=3)

    train_dataloader = get_dataloaders('train')
    show_forward(cddpm, train_dataloader, 'cpu', n_images=4, n_noise_steps=8)

    data = dict(np.load('/scratch/globc/garcia/datasets/dataset_exp3_30y/sample_20040101.npz', allow_pickle=True))
    conditioning_image, y = data['x'], data['y']
    transforms = v2.Compose([
            MinMaxNormalisation(Path('/scratch/globc/garcia/datasets/dataset_exp3_30y')), 
            LandSeaMask('france', 0),
            FillMissingValue(0),
            Pad(0)
            ])
    
    conditioning_image, y = transforms((conditioning_image, y))
    conditioning_image = np.expand_dims(conditioning_image, axis=0)
    conditioning_image = torch.tensor(conditioning_image)

    start_t = 8
    x = cddpm.sampling(start_t, conditioning_image, eta = None)
    print(x.shape)
    plot_test(x[0,0,...].detach().numpy(), 
              'generate x from noise', 
              GRAPHS_DIR/'test2.png')