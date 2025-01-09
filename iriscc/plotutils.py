import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import cartopy.feature as cfeature

from iriscc.settings import (PROJ, DOMAIN)



def plot_image(var,
               title,
               save_dir):

    proj = PROJ
    _, ax = plt.subplots(
        figsize=(10, 6),
        subplot_kw={"projection": proj}
    )
    img = ax.imshow(
        var,
        extent=DOMAIN, 
        transform=proj, 
        origin='lower',
        cmap='jet',
        aspect='auto'
    )

    ax.set_extent(DOMAIN, crs=proj)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1, zorder=10)
    ax.add_feature(cfeature.BORDERS, linestyle='--', edgecolor='gray', zorder=10)

    cbar = plt.colorbar(img, ax=ax, pad=0.05)
    plt.title(title, fontsize=12)
    plt.savefig(save_dir)

def plot_test(var, title, save_dir, vmin=None, vmax=None):
    ''' Simple test plot '''
    var = np.flip(var, axis=0)
    _, ax = plt.subplots()
    im = ax.imshow(var, aspect='equal', cmap='OrRd', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, pad=0.05)
    plt.title(title)
    plt.savefig(save_dir)

def plot_contour(var, title, save_dir, levels=None):
    _, ax = plt.subplots()
    cs = ax.contourf(var, cmap='OrRd', levels=levels)
    plt.colorbar(cs, ax=ax, pad=0.05)
    plt.title(title)
    plt.savefig(save_dir)

if __name__=='__main__':
    data = dict(np.load('/gpfs-calypso/scratch/globc/garcia/datasets/dataset_exp1/sample_19640101.npz', allow_pickle=True))
    x = data['x']
    y = data['y']
    plot_image(x[0], '20141229 Topography', '/gpfs-calypso/scratch/globc/garcia/graph/datasets/19640101_x0.png')
    plot_image(x[1], '20141229 CNRM-CM6-1 r10i1p1f2', '/gpfs-calypso/scratch/globc/garcia/graph/datasets/19640101_x1.png')
    plot_image(y[0], '20141229 SAFRAN Target', '/gpfs-calypso/scratch/globc/garcia/graph/datasets/19640101_y.png')