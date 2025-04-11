import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from iriscc.settings import (SAFRAN_PROJ, CONFIG)

def plot_map_image(var,
                var_desc = None,
                cmap = 'OrRd',
                vmin = None,
                vmax = None,
                domain = None,
                fig_projection = ccrs.PlateCarree(),
                data_projection = ccrs.PlateCarree(),
                title = None,
                save_dir = None):

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
                    var_desc = None,
                    cmap = 'OrRd',
                    fig_projection = ccrs.PlateCarree(),
                    data_projection = ccrs.PlateCarree(),
                    levels=None,
                    domain = None,
                    title = None,
                    save_dir = None):
    #projection = ccrs.PlateCarree()
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
    