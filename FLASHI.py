import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import concurrent.futures
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords,extract_times)
from matplotlib.cm import (get_cmap,ScalarMappable)
import glob
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from netCDF4 import Dataset
from metpy.plots import USCOUNTIES
from matplotlib.colors import Normalize
from PIL import Image
from datetime import datetime, timedelta
import colorcet as cc
import sys
from metpy.plots import ctables
import cartopy.io.shapereader as shpreader
import wrffuncs
# ---- User input for file ----

# User input for file
wrf_date_time = datetime(2022,11,18,1,55,00)
domain = 2
numtimeidx = 4
IOP = 2
ATTEMPT = 1

wrf_date_time = wrffuncs.round_to_nearest_5_minutes(wrf_date_time)
print("Closest WRF time to match input time: ", wrf_date_time)
timeidx = int((wrf_date_time.minute % 20) // (20 // numtimeidx))
wrf_filename = wrf_date_time
wrf_min = (round((wrf_filename.minute)// 20) * 20)
wrf_filename = wrf_filename.replace(minute=wrf_min)
print("Using data from this file: ", wrf_filename)

# dBZ contour line to use
dBZ_contour = [20]
# Variable to use to create many maps

savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{ATTEMPT}/"

# ---- End User input for file ----

# Weird thing to do cause error
# export PROJ_NETWORK=OFF

# Open the NetCDF file
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_{IOP}/ATTEMPT_{ATTEMPT}/"
pattern = f"wrfout_d0{domain}_{wrf_filename.year:04d}-{wrf_filename.month:02d}-{wrf_filename.day:02d}_{wrf_filename.hour:02d}:{wrf_filename.minute:02d}:00"
wrffile = path + pattern 
def generate_frame(wrffile, timeidx, savepath):
    try:
    # Read data from file
        with Dataset(wrffile) as wrfin:
            data = getvar(wrfin, "FLSHI", timeidx=timeidx)
            mdbz = getvar(wrfin, "mdbz", timeidx=timeidx)

    # Create a figure
        fig = plt.figure(figsize=(30,15),facecolor='white')

    # Get the latitude and longitude points
        lats, lons = latlon_coords(data)

    # Get the cartopy mapping object
        cart_proj = get_cartopy(data)
    
    # Set the GeoAxes to the projection used by WRF
        ax = plt.axes(projection=cart_proj)
    
    # Special stuff for counties
        reader = shpreader.Reader('countyl010g.shp')
        counties = list(reader.geometries())
        COUNTIES = cfeature.ShapelyFeature(counties, crs.PlateCarree(),zorder=5)
        ax.add_feature(COUNTIES,facecolor='none', edgecolor='black',linewidth=1)

    # Set the map bounds
        ax.set_xlim(cartopy_xlim(data))
        ax.set_ylim(cartopy_ylim(data))
    

    # Add the gridlines
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
        gl.xlabel_style = {'rotation': 'horizontal','size': 28,'ha':'center'} # Change 14 to your desired font size
        gl.ylabel_style = {'size': 28}  # Change 14 to your desired font size
        gl.xlines = True
        gl.ylines = True
        gl.top_labels = False  # Disable top labels
        gl.right_labels = False  # Disable right labels
        gl.xpadding = 20
    
    
    #data = data[0,:,:]
        print(lats)
        print(lons)
        print(data)
    # Define contour levels
        levels = np.arange(0,50,5)    
        data = to_np(data)
        lats = to_np(lats) 
        lons = to_np(lons) 
        flash_data = np.sum(data, axis=0)
        cmap = ctables.registry.get_colortable('NWSReflectivity')
        qcs = plt.contour(lons, lats,mdbz,transform=crs.PlateCarree(), levels=dBZ_contour,colors="green")

    #Check to see if any data is present
    #for index,matrix in enumerate(flash_data):
       # for j, row in enumerate(matrix):
            #for k, value in enumerate(row):
      #      if row > 0:
     #           print("value of " + str(row) + "at index " + str(j))
    
    #print("coords: ", lats[flash_data == 1.0], lons[flash_data == 1.0])
        ax.scatter(lons[flash_data == 1.0],lats[flash_data == 1.0],s=75,marker="X",c='red',transform=crs.PlateCarree())
    # Add a fixed colorbar
    #cbar = plt.colorbar()
    #cbar.set_label("", fontsize=22)
    #cbar.ax.tick_params(labelsize=14)  
    
        # Adjust format for date to use in figure
        date_format = wrf_date_time.strftime("%Y-%m-%d %H:%M:%S")

        ax.set_title(f"Flash Initiation Points at {date_format}",fontsize=25,fontweight='bold')
        ax.set_extent([-76.965996,-75.090013,43.394115,44.273301],crs=crs.PlateCarree())

        plt.savefig(savepath + f"FLSHI{wrf_date_time.year:04d}{wrf_date_time.month:02d}{wrf_date_time.day:02d}{wrf_date_time.hour:02d}{wrf_date_time.minute:02d}D{domain}T{timeidx}A{ATTEMPT}.png")
        plt.show()
    except IndexError:
        print("Error occured")
if __name__ == "__main__":
    generate_frame(wrffile, timeidx,savepath)

    
