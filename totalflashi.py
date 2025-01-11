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
from metpy.plots import USCOUNTIES, ctables
from matplotlib.colors import Normalize
from PIL import Image
from datetime import datetime, timedelta
import wrffuncs
import cartopy.io.shapereader as shpreader
import pyart
import multiprocessing as mp

# MAY NEED TO USE IN SHELL
#export PROJ_NETWORK=OFF


# ---- User input for file ----
start_time, end_time  = datetime(2022,11,17,00,00,00), datetime(2022, 11, 19,12, 00, 00)
domain = 2
numtimeidx = 4
ATTEMPT = 1

#If WRF input is not already to nearest 5 minutes, it is rounded here. This should be adjuted based on the file, however this file has 5 minute resolution
wrf_date_time_start = wrffuncs.round_to_nearest_5_minutes(start_time)
wrf_date_time_end = wrffuncs.round_to_nearest_5_minutes(end_time)

# Empty arrays for storing data
timeidxlist = np.array([],dtype=int)
filelist = np.array([])

current_time = wrf_date_time_start

while current_time <= wrf_date_time_end:
    # Calculate time index needed from file
    timeidx = int((current_time.minute % 20) // (20 // numtimeidx))
    print("Using timeidx: ", timeidx)

    # Get WRF minute, since they are in 20 minute increments
    wrf_min = (round((current_time.minute)// 20) * 20)
    wrf_filename = current_time.replace(minute=wrf_min)

    # Display the file using and time index
    print(f"Using data from file: {wrf_filename}, time index: {timeidx}")
    pattern = f"wrfout_d0{domain}_{wrf_filename.year:04d}-{wrf_filename.month:02d}-{wrf_filename.day:02d}_{wrf_filename.hour:02d}:{wrf_filename.minute:02d}:00"
    
    # Add the file and time index to our empty arrays to use later
    timeidxlist = np.append(timeidxlist, timeidx)
    filelist = np.append(filelist, pattern)

    # Increment time by 5 minute
    current_time += timedelta(minutes=5)

# Directory to store saved files
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/"

# Henderson Harbor coordinates
hend_harb = [43.88356, -76.155543]
# ---- End User input for file ----

# Open the NetCDF file
path_a1 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_1/"
path_a2 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_2/"
path_a3 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_3/"


def generate_frame(file_path1,file_path2,file_path3, timeidx):
    print("Starting generate frame")
        

    def parse_filename_datetime_obs(filepath):
    
    # Get the filename 
        filename = filepath.split('/')[-1]
    # Extract the date part (8 characters starting from the 5th character)
        date_str = filename[4:12]
    # Extract the time part (6 characters starting from the 13th character)
        time_str = filename[13:19]
    # Combine the date and time strings
        datetime_str = date_str + time_str
    # Convert the combined string to a datetime object
    #datetime_obj = datetime.datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
    #formatted_datetime_obs = parse_filename_datetime_obs(radar_data)datetime_obj.strftime('%B %d, %Y %H:%M:%S') 
        return datetime.strptime(datetime_str, '%Y%m%d%H%M%S')

    try:
   
    # Read data from file
        with Dataset(file_path1) as wrfin:
            data1 = getvar(wrfin, "FLSHI", timeidx=timeidx)
            print("Read in WRF data from file 1")
        with Dataset(file_path2) as wrfin:
            data2 = getvar(wrfin, "FLSHI", timeidx=timeidx)
            print("Read in WRF data from file 2")
        with Dataset(file_path3) as wrfin:
            data3 = getvar(wrfin, "FLSHI", timeidx=timeidx)
            print("Read in WRF data from file 3")


    # Define the format of the datetime string in your filename
        datetime_format = "wrfout_d02_%Y-%m-%d_%H:%M:%S"
        print("made datetime format")

    # Parse the datetime string into a datetime object
        time_object = datetime.strptime(os.path.basename(file_path), datetime_format)
        print("Made time_object")

    # Add timeidx value to get the exact minute
        add_time = 5 * float(timeidx)
        time_object_adjusted = time_object + timedelta(minutes=add_time)
        print("Adjusted WRF time")
            
    # Get the latitude and longitude points
        lats1, lons1 = latlon_coords(data1)
        lats2, lons2 = latlon_coords(data1)
        lats3, lons3 = latlon_coords(data1)


    # Convert arrays intop numpy arrays, np.sum is used so we only get the x,y coordinates           
        data1 = to_np(data1)
        lats1 = to_np(lats1) 
        lons1 = to_np(lons1) 
        flash_data1 = np.sum(data1, axis=0)
        
        data2 = to_np(data2)
        lats2 = to_np(lats2) 
        lons2 = to_np(lons2) 
        flash_data2 = np.sum(data2, axis=0)

        data3 = to_np(data3)
        lats3 = to_np(lats3) 
        lons3 = to_np(lons3) 
        flash_data3 = np.sum(data3, axis=0)

    # Plot each lon and lat points, based on if there is data (1.0) in the respective flash_data array
        ax.scatter(lons1[flash_data1 == 1.0],lats1[flash_data1 == 1.0],s=75,marker="*",c='yellow',transform=crs.PlateCarree())
        ax.scatter(lons2[flash_data2 == 1.0],lats2[flash_data2 == 1.0],s=75,marker="*",c='red',transform=crs.PlateCarree())
        ax.scatter(lons3[flash_data3 == 1.0],lats3[flash_data3 == 1.0],s=75,marker="*",c='blue',transform=crs.PlateCarree())
        global count
        global coords
        # If we wanted to track the coords and counts of certian flashes, we can use this
        if any(lons1[flash_data1 == 1.0]):
            count += 1
            times.append(time_object_adjusted)
            coords.append((lats1[flash_data1 == 1.0],lons1[flash_data1 == 1.0]))
            
        #print(f"{os.path.basename(file_path)} Processed!")
        #if time_object_adjusted.day == 19 and time_object_adjusted.hour == 0 and time_object_adjusted.minute == 0:
            #plt.show()
        #plt.close()


    except IndexError:
        print("Error processing files")
            
if __name__ == "__main__":
    count = 0
    coords = []
    times = []
    # Read data from any matching WRF file to set up the plot, not for the data
    with Dataset(path_a1+f'wrfout_d0{domain}_2022-11-17_11:00:00') as wrfin:
        data = getvar(wrfin, "FLSHI", timeidx=timeidx)
     
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
    ax.set_extent([-77, -75.5,43.4, 44.4])

    # Add the gridlines
    gl = ax.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {'rotation': 'horizontal','size': 24,'ha':'center'} # Change 14 to your desired font size
    gl.ylabel_style = {'size': 28}  # Change 14 to your desired font size
    gl.xlines = True
    gl.ylines = True
    gl.top_labels = False  # Disable top labels
    gl.right_labels = False  # Disable right labels
    gl.xpadding = 20

    # Set each filelist to contain every WRF file for the loop
    wrf_filelist_a1 = glob.glob(os.path.join(path_a1,f'wrfout_d0{domain}_2022-11-1*'))
    wrf_filelist_a2 = glob.glob(os.path.join(path_a2,f'wrfout_d0{domain}_2022-11-1*'))
    wrf_filelist_a3 = glob.glob(os.path.join(path_a3,f'wrfout_d0{domain}_2022-11-1*'))

   

    
    for idx, file_path in enumerate(wrf_filelist_a1):
        for timeidx in range(numtimeidx):
            generate_frame(wrf_filelist_a1[idx], wrf_filelist_a2[idx], wrf_filelist_a3[idx], timeidx)
    print("Times", times)
    print("Count of YSU", count)
    print("Coords of YSU", coords)
    plt.title("Total Flash Initiation Locations",fontsize=28)
    plt.show()         

       
