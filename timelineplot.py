import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob
import os
from datetime import datetime
import wrffuncs # Personal library
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords,extract_times)
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import cartopy.crs as crs
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta


# ---- User input for file ----
start_time, end_time  = datetime(2022,11,17,00,00,00), datetime(2022, 11, 19, 12, 15, 00)
domain = 2
numtimeidx = 4
attempt = 1

# Define paths to your WRF output directories
path_a1 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_1/"
path_a2 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_2/"
path_a3 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_3/"

# Get list of all WRF output files
wrf_filelist_a1 = sorted(glob.glob(os.path.join(path_a1, 'wrfout_d02*')))
wrf_filelist_a2 = sorted(glob.glob(os.path.join(path_a2, 'wrfout_d02*')))
wrf_filelist_a3 = sorted(glob.glob(os.path.join(path_a3, 'wrfout_d02*')))
#print(wrf_filelist_a1[0])
# Initialize lists to store time and cumulative data
cumulative_data1 = []
cumulative_data2 = []
cumulative_data3 = []

# Function to read and sum data from a file
def read_and_sum(file_path, variable_name,timeidx,mask):
    with Dataset(file_path) as wrfin:
        try:
            data = getvar(wrfin, variable_name, timeidx=timeidx, meta=False)[mask]
            
        except IndexError:
            print("index exceeds dimension bounds")
            data = None
        return np.mean(data)

time_array = []
current_time = start_time
while current_time <= end_time:
    time_array.append(current_time)
    current_time += timedelta(minutes=5)    

# Create mask to limit data to specific area
with Dataset(path_a1+'wrfout_d02_2022-11-19_00:00:00') as wrfin:
        lat = getvar(wrfin, "lat", timeidx=0)
        lon = getvar(wrfin, "lon", timeidx=0)

lat_min, lat_max = 43.1386, 44.2262
lon_min, lon_max = -77.345, -74.468
lat = to_np(lat)
lon = to_np(lon)
lat_mask = (lat > lat_min) & (lat < lat_max)
lon_mask = (lon > lon_min) & (lon < lon_max)
    
# This mask can be used an any data to ensure we are in are modified region
region_mask = lat_mask & lon_mask
    
# Apply the mask to WRF data
masked_data_a1 = np.where(region_mask, lat, np.nan)

# Use this to remove nan's for statistical operations, can apply to all the data since they have matching domains
final_mask = ~np.isnan(masked_data_a1)

# Loop through all files and each time index to sum the data
for file_path1, file_path2, file_path3 in zip(wrf_filelist_a1, wrf_filelist_a2, wrf_filelist_a3):
    for time_idx in range(4):  # Assuming there are 4 time indices per file
        #print(time_idx)
        cumulative_data1.append(read_and_sum(file_path1, 'PBLH', time_idx,final_mask))
        cumulative_data2.append(read_and_sum(file_path2, 'PBLH', time_idx,final_mask))
        cumulative_data3.append(read_and_sum(file_path3, 'PBLH', time_idx,final_mask))

# Convert lists to numpy arrays for plotting
times = np.array(time_array)
cumulative_data1 = np.array(cumulative_data1)
cumulative_data2 = np.array(cumulative_data2)
cumulative_data3 = np.array(cumulative_data3)
print(times.shape)
print(cumulative_data1.shape)


# Example list of flash times
flash_times_YSU = [
    datetime(2022, 11, 18, 13, 40, 00),
    datetime(2022, 11, 18, 13, 45, 00),
]

flash_times_MYNN2 = [
    datetime(2022, 11, 18, 11, 5, 00),
    datetime(2022, 11, 18, 11, 35, 00),
    datetime(2022, 11, 18, 11, 40, 00),
    datetime(2022, 11, 18, 11, 50, 00),
    datetime(2022, 11, 18, 11, 55, 00),
    datetime(2022, 11, 18, 12, 5, 00),
    datetime(2022, 11, 18, 12, 35, 00),
    datetime(2022, 11, 18, 14, 5, 00),
    datetime(2022, 11, 18, 14, 15, 00),
    datetime(2022, 11, 18, 14, 20, 00),
    datetime(2022, 11, 18, 14, 25, 00),
    datetime(2022, 11, 18, 14, 30, 00),
    datetime(2022, 11, 18, 14, 35, 00),
    datetime(2022, 11, 18, 14, 40, 00),
    datetime(2022, 11, 18, 15, 10, 00),

]

# Create a line plot


fig,ax = plt.subplots(figsize=(12, 6))
ax.plot(times[1:], cumulative_data1[1:], label='YSU', color='red')
ax.plot(times[1:], cumulative_data2[1:], label='UW', color='yellow')
ax.plot(times[1:], cumulative_data3[1:], label='MYNN2', color='blue')

for flash_time in flash_times_YSU:
    ax.axvline(x=flash_time, color='red', linestyle='dashdot', linewidth=.5)

for flash_time in flash_times_MYNN2:
    ax.axvline(x=flash_time, color='blue', linestyle='dashdot', linewidth=.5)
ax.set_xlabel('Time', fontsize=18)
ax.set_ylabel('Height (m)',fontsize=18)
ax.set_title('Planetary Boundary Layer Height',fontsize=24)
ax.legend()
ax.grid(True)
plt.show()
