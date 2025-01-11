import numpy as np
from netCDF4 import Dataset
import glob
import os
import matplotlib.pyplot as plt
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords,extract_times)
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import cartopy.crs as crs
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
# Define paths to your WRF output directories
path_a1 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_1/"
path_a2 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_2/"
path_a3 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_3/"

# Get list of all WRF output files
wrf_filelist_a1 = glob.glob(os.path.join(path_a1, 'wrfout_d02*'))
wrf_filelist_a2 = glob.glob(os.path.join(path_a2, 'wrfout_d02*'))
wrf_filelist_a3 = glob.glob(os.path.join(path_a3, 'wrfout_d02*'))

# Initialize cumulative arrays
cumulative_data1 = None
cumulative_data2 = None
cumulative_data3 = None

# Function to read and sum data from a file
def read_and_sum(file_path, variable_name, cumulative_data):
    with Dataset(file_path) as wrfin:
        # timeidx at None gives all times, meta at False gives numpy array instead of xarray
        data = getvar(wrfin, variable_name, timeidx=None, meta=False)
        data = np.sum(data, axis=0)
        if cumulative_data is None:
            cumulative_data = np.zeros_like(data)
        
        cumulative_data = cumulative_data + data
        #print(cumulative_data.shape)
        #print(cumulative_data[0][0])
    return cumulative_data

# Loop through all files and sum the data
for file_path1, file_path2, file_path3 in zip(wrf_filelist_a1, wrf_filelist_a2, wrf_filelist_a3):
    cumulative_data1 = read_and_sum(file_path1, 'LIGHTDENS', cumulative_data1)
    cumulative_data2 = read_and_sum(file_path2, 'LIGHTDENS', cumulative_data2)
    cumulative_data3 = read_and_sum(file_path3, 'LIGHTDENS', cumulative_data3)

# Read a file in for the metadata to create the figure
with Dataset(path_a1+f'wrfout_d02_2022-11-17_11:00:00') as wrfin:
    data = getvar(wrfin, 'LIGHTDENS', timeidx=1)

# Create a figure
fig = plt.figure(figsize=(12,9),facecolor='white')
    
# Get the latitude and longitude points
lats, lons = latlon_coords(data)

# Get the cartopy mapping object
cart_proj = get_cartopy(data)
    
# Set the GeoAxes to the projection used by WRF
ax = plt.axes(projection=cart_proj)
    
# Special stuff for counties
reader = shpreader.Reader('countyline_files/countyl010g.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, crs.PlateCarree(),zorder=5)
ax.add_feature(COUNTIES,facecolor='none', edgecolor='black',linewidth=1)

# Set the map bounds
ax.set_extent([-77.3, -74.5,42.5, 44.5])

# Add the gridlines
gl = ax.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
gl.xlabel_style = {'rotation': 'horizontal','size': 18,'ha':'center'} # Change 14 to your desired font size
gl.ylabel_style = {'size': 18}  # Change 14 to your desired font size
gl.xlines = True
gl.ylines = True
gl.top_labels = False  # Disable top labels
gl.right_labels = False  # Disable right labels
gl.xpadding = 20

# Create a custom color map where zero values are white
colors = [(1, 1, 1), (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]  # White to Red
n_bins = 500  # Discretizes the interpolation into bins
cmap_name = 'custom_cmap'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Plot the cumulative flash extent density using pcolormesh
mesh = ax.pcolormesh(to_np(lons), to_np(lats), cumulative_data3, cmap=custom_cmap, norm=LogNorm(1,50), transform=crs.PlateCarree())
extent = [-77.5, -74.0,42.5, 44.5]

# Add a colorbar
cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.1,shrink=.5)
cbar.set_label('Flash Extent Density', fontsize=18)

# Show the plot
plt.title("Flash Extent Density (Sum # of Flashes / Grid Column)", fontsize=28)
plt.show()

