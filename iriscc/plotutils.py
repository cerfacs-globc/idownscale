''' 
Useful plot functions

date : 16/07/2025
author : Zoé GARCIA
'''

import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
        figsize=(6,5),
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
    ax.set_extent([-5., 11., 41., 51.], crs=ccrs.PlateCarree())
    #ax.set_extent(domain, crs=fig_projection)
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=100, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1, zorder=10)
    ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1, edgecolor='gray', zorder=10)

    cbar = plt.colorbar(img, ax=ax, pad=0.05, shrink=0.8)
    cbar.set_label(label=var_desc, size=14, labelpad=10)
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.title(title, fontsize=16, pad=10)
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
        figsize=(6, 5),
        subplot_kw={"projection": fig_projection}
    )
    
    cs = ax.contourf(var, 
                     cmap=cmap, 
                     levels=levels,
                     extent=domain,
                     transform=data_projection
                    )
    ax.set_extent([-5., 11., 41., 51.], crs=ccrs.PlateCarree())
    #ax.set_extent(domain, crs=fig_projection)
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=100, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1, zorder=10)
    ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1, edgecolor='gray', zorder=10)

    cbar = plt.colorbar(cs, ax=ax, pad=0.05, shrink=0.8)
    cbar.set_label(label=var_desc, size=14, labelpad=10)
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.title(title, fontsize=16, pad=10)
    if save_dir is None:
        return fig, ax  
    else:
        plt.savefig(save_dir)

def plot_test(var,  save_dir: str, title: str = None,vmin: float = None, vmax: float = None):
    '''
    Simple test plot function.

    Args:
        var (array-like): 2D array of data to be plotted.
        title (str): Title of the plot.
        save_dir (str): Path to save the generated plot.
        vmin (float, optional): Minimum value for colormap scaling. Defaults to None.
        vmax (float, optional): Maximum value for colormap scaling. Defaults to None.
    Returns:
        None
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

def plot_monthly_var_seasonal_cycle(
    var_temporal: np.ndarray, 
    dates: np.ndarray, 
    title: str, 
    var_desc: str, 
    save_dir: str
    ) -> None:
    """
    Plots the seasonal cycle of a variable, showing the mean monthly values
    and individual yearly trends, and saves the plot to a specified directory.
    The function accepts either daily or monthly data as input.

    Args:
    var_temporal (np.ndarray): Array of temporal variable values.
    dates (np.ndarray): Array of corresponding dates for the variable values.
    title (str): Title of the plot.
    var_desc (str): Description of the variable (e.g., temperature).
    save_dir (str): File path to save the generated plot.

    Returns:
    None
    """
    df_var = pd.DataFrame({'date': dates, 'var_temporal': var_temporal})
    df_var['date'] = pd.to_datetime(df_var['date'])
    
    # Determine if the data is daily or monthly
    if len(df_var['date'].dt.day.unique()) > 1:  # Data is daily
        df_var['month'] = df_var['date'].dt.month
        df_var['year'] = df_var['date'].dt.year
        var_monthly_mean = df_var.groupby('month')['var_temporal'].mean()
        var_per_year = df_var.pivot_table(index='month', columns='year', values='var_temporal')
    else:  # Data is already monthly
        df_var['month'] = df_var['date'].dt.month
        df_var['year'] = df_var['date'].dt.year
        var_monthly_mean = df_var.groupby('month')['var_temporal'].mean()
        var_per_year = df_var.pivot_table(index='month', columns='year', values='var_temporal')

    plt.figure(figsize=(10, 6))
    plt.suptitle(title, fontsize=16)
    ax = plt.gca()
    plt.plot(var_monthly_mean.index, var_monthly_mean.values, label='Mean', color='red', linewidth=2)
    for year in var_per_year.columns:
        plt.plot(var_per_year.index, var_per_year[year], label=str(year), alpha=0.3, linestyle='--')
    plt.xticks(ticks=np.arange(1, 13), labels=[
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=12)
    plt.ylabel(f'{var_desc}')
    plt.xlabel('Month')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=12, ncol=2)
    ax.text(0.02, 0.10, f"Mean temporal {var_desc}: {np.mean(var_temporal):.2f}", transform=ax.transAxes, fontsize=12, 
        verticalalignment='top', horizontalalignment='left', color='red')
    plt.tight_layout()
    plt.savefig(save_dir)


def plot_histogram(data, ax, labels, colors, xlabel):
    for i in range(len(data)):
        ax.hist(data[i], histtype='step', color=colors[i], 
                label=labels[i], density=True, range=(260,310), bins=100, linewidth=2)
        ax.axvline(np.nanmean(data[i]), color=colors[i], linestyle='--', linewidth=2)
    ax.set_ylim(0, 0.07)
    ax.set_xlabel(xlabel)
    ax.legend()

