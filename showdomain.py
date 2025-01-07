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
from shapely.geometry import box
from cartopy.feature import ShapelyFeature
from shapely.geometry import Polygon
# ---- User input for file ----

# YEAR MONTH DAY HOUR MIN DOMAIN ATTEMPT
date_time = [2022, 11, 18 , "13", "40",2,1]
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{date_time[6]}/"
wrffile = f"wrfout_d0{date_time[5]}_{date_time[0]}-{date_time[1]}-{date_time[2]}_{date_time[3]}:{date_time[4]}:00"

#timeidx in this domain
timeidx = 2

# ---- End User input for file ----

# Weird thing to do cause error
# Disable PROJ cache
# export PROJ_NETWORK=OFF
#os.environ['PROJ_NETWORK'] = 'OFF'
#os.environ['PROJ_USER_WRITABLE_DIRECTORY'] = '/data2/white/PYTHON_SCRIPTS/PROJ_LEE/tmp/proj'
#np.set_printoptions(threshold=sys.maxsize)
# Open the NetCDF file
path = "/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_1/"
wrffile = path + wrffile
def generate_frame(wrffile, timeidx, savepath):
    try:
    # Read data from file
        with Dataset(wrffile) as wrfin:
            mdbz = getvar(wrfin, "mdbz", timeidx=timeidx)

    # Create a figure
        fig = plt.figure(figsize=(30,15),facecolor='white')

    # Get the latitude and longitude points
        lats, lons = latlon_coords(mdbz)

    # Get the cartopy mapping object
        cart_proj = get_cartopy(mdbz)
    
    # Set the GeoAxes to the projection used by WRF
        ax = plt.axes(projection=cart_proj)
    
    # Special stuff for counties
        reader = shpreader.Reader('countyl010g.shp')
        counties = list(reader.geometries())
        COUNTIES = cfeature.ShapelyFeature(counties, crs.PlateCarree(),zorder=5)
        ax.add_feature(COUNTIES,facecolor='none', edgecolor='black',linewidth=1)

    # Set the map bounds
        ax.set_xlim(cartopy_xlim(mdbz))
        ax.set_ylim(cartopy_ylim(mdbz))
    

    # Add the gridlines
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
        gl.xlabel_style = {'rotation': 'horizontal','size': 22,'ha':'center'} # Change 14 to your desired font size
        gl.ylabel_style = {'size': 22}  # Change 14 to your desired font size
        gl.xlines = True
        gl.ylines = True
        gl.top_labels = False  # Disable top labels
        gl.right_labels = False  # Disable right labels
        gl.xpadding = 20
    
        lat1, lon1 = 43.1386, -77.345  # Bottom-left corner
        lat2, lon2 = 44.2262, -74.468  # Top-right corner
        # Define the coordinates of the square (in order)
        coordinates = [(lon1, lat1), (lon2, lat1), (lon2, lat2), (lon1, lat2), (lon1, lat1)]
        polygon = Polygon(coordinates)
        # Create a feature for the polygon
        square_feature = ShapelyFeature([polygon], crs.PlateCarree(), edgecolor='red', facecolor='none',linewidth=3)

        # Add the square to the plot
        ax.add_feature(square_feature)
        plt.savefig(savepath + f"FLSHI{date_time[0]}{date_time[1]}{date_time[2]}{date_time[3]}{date_time[4]}D{date_time[5]}T{timeidx}A{date_time[6]}")
        plt.show()
    except IndexError:
        print("Error occured")
if __name__ == "__main__":
    generate_frame(wrffile, timeidx,savepath)

    
