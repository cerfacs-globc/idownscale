''' Useful plot functions '''

import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from iriscc.settings import (CONFIG)

def plot_map_image(var,
                    var_desc: str = None,
                    cmap: str = 'OrRd',
                    vmin: float = None,
                    vmax: float = None,
                    domain: list = None,
                    fig_projection = ccrs.PlateCarree(),
                    data_projection = ccrs.PlateCarree(),
                    title: str = None,
                    save_dir: str = None
                ):
    """
    Plots a 2D map image using the provided data and configurations.
    Args:
        var (Any): The 2D array-like data to be plotted.
        var_desc (str, optional): Description of the variable to be used as the colorbar label. Defaults to None.
        cmap (str, optional): Colormap to be used for the plot. Defaults to 'OrRd'.
        vmin (float, optional): Minimum value for the color scale. Defaults to None.
        vmax (float, optional): Maximum value for the color scale. Defaults to None.
        domain (list, optional): List defining the spatial extent of the plot in the format [min_lon, max_lon, min_lat, max_lat]. Defaults to None.
        fig_projection (Any, optional): Cartopy projection for the figure. Defaults to ccrs.PlateCarree().
        data_projection (Any, optional): Cartopy projection for the data. Defaults to ccrs.PlateCarree().
        title (str, optional): Title of the plot. Defaults to None.
        save_dir (str, optional): Path to save the plot as an image file. If None, the function returns the figure and axis objects. Defaults to None.
    Returns:
        tuple: A tuple containing the figure and axis objects if `save_dir` is None. Otherwise, saves the plot to the specified directory.
    """


    fig, ax = plt.subplots(
        figsize=(6,7),
        subplot_kw={"projection": fig_projection}
    )
    
    img = ax.imshow(
        var,
        extent=domain, 
        transform=data_projection, 
        origin='lower',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_extent(domain, crs=data_projection)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1, zorder=10)
    ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1, edgecolor='gray', zorder=10)

    cbar = plt.colorbar(img, ax=ax, pad=0.05, shrink=0.8)
    cbar.set_label(label=var_desc, size=12)
    plt.tight_layout()
    plt.title(title, fontsize=14)
    if save_dir is None:
        return fig, ax  
    else:
        plt.savefig(save_dir)

def plot_map_contour(var,
                    var_desc: str = None,
                    cmap: str = 'OrRd',
                    fig_projection: ccrs.Projection = ccrs.PlateCarree(),
                    data_projection: ccrs.Projection = ccrs.PlateCarree(),
                    levels: list = None,
                    domain: list = None,
                    title: str = None,
                    save_dir: str = None
                    ):
    """
    Plots a contour map using the provided data and configurations.
    Args:
        var: The data to be plotted, typically a 2D array or similar structure.
        var_desc (str, optional): Description of the variable to be used as the colorbar label.
        cmap (str, optional): Colormap to be used for the plot. Defaults to 'OrRd'.
        fig_projection (ccrs.Projection, optional): The map projection for the figure. Defaults to PlateCarree.
        data_projection (ccrs.Projection, optional): The projection of the input data. Defaults to PlateCarree.
        levels (list, optional): Contour levels for the plot. If None, levels are automatically determined.
        domain (list, optional): The geographical extent of the plot in the format [min_lon, max_lon, min_lat, max_lat].
        title (str, optional): Title of the plot. Defaults to None.
        save_dir (str, optional): File path to save the plot. If None, the function returns the figure and axis objects.
    Returns:
        tuple: A tuple containing the figure and axis objects if `save_dir` is None. Otherwise, saves the plot to the specified directory.
    """
    fig, ax = plt.subplots(
        figsize=(6, 7),
        subplot_kw={"projection": fig_projection}
    )
    
    cs = ax.contourf(var, 
                     cmap=cmap, 
                     levels=levels,
                     extent=domain,
                     transform=data_projection
                    )
    
    ax.set_extent(domain, crs=data_projection)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1, zorder=10)
    ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1, edgecolor='gray', zorder=10)

    cbar = plt.colorbar(cs, ax=ax, pad=0.05, shrink=0.75)
    cbar.set_label(label=var_desc, size=12)
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.title(title, fontsize=14)
    if save_dir is None:
        return fig, ax  
    else:
        plt.savefig(save_dir)

def plot_test(var, title: str, save_dir: str, vmin: float = None, vmax: float = None):

    ''' 
    Simple test plot function.

    Parameters:
    var : array-like
        2D array of data to be plotted.
    title : str
        Title of the plot.
    save_dir : str
        Path to save the generated plot.
    vmin : float, optional
        Minimum value for colormap scaling. Defaults to None.
    vmax : float, optional
        Maximum value for colormap scaling. Defaults to None.
    '''

    var = np.flip(var, axis=0)
    _, ax = plt.subplots()
    im = ax.imshow(var, aspect='equal', cmap='OrRd', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, pad=0.05)
    plt.title(title)
    plt.savefig(save_dir)

def plot_contour(var, title, save_dir, levels=None):
    _, ax = plt.subplots()
    cs = ax.contourf(var, cmap='Paired', levels=levels)
    plt.colorbar(cs, ax=ax, pad=0.05)
    plt.title(title)
    plt.savefig(save_dir)

if __name__=='__main__':

    ds = xr.open_dataset('/gpfs-calypso/scratch/globc/garcia/prediction/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20150101_21001231_unet_cmip6_bc.nc')
    tas = ds.tas.values[-1,:,:]
    fig, ax = plot_map_contour(tas,
                   domain = CONFIG['safran']['domain']['france_xy'],
                    title = f'safran {ds.time.values[-1]}',
                    fig_projection=CONFIG['safran']['fig_projection']['france_xy'],
                    data_projection=CONFIG['safran']['data_projection'],
                    cmap='OrRd',
                    var_desc='T (K)')
    plt.savefig('/gpfs-calypso/scratch/globc/garcia/graph/test1.png')
    