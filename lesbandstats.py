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
import cartopy.io.shapereader as shpreader
import pyart
import multiprocessing as mp
import numpy.ma as ma
import wrffuncs

# MAY NEED TO USE IN SHELL
#export PROJ_NETWORK=OFF

# ---- User input for file ----
start_time, end_time  = datetime(2022,11,18,12,37,00), datetime(2022, 11, 18, 13, 1, 00)
domain = 2
numtimeidx = 4


wrf_date_time_start = wrffuncs.round_to_nearest_5_minutes(start_time)
wrf_date_time_end = wrffuncs.round_to_nearest_5_minutes(end_time)
timeidxlist = np.array([])
filelist = np.array([])
current_time = wrf_date_time_start
while current_time <= wrf_date_time_end:
    timeidx = int((current_time.minute % 20) // (20 // numtimeidx))
    print("Using timeidx: ", timeidx)
    wrf_min = (round((current_time.minute)// 20) * 20)
    wrf_filename = current_time.replace(minute=wrf_min)
    print(f"Using data from file: {wrf_filename}, time index: {timeidx}")
    pattern = f"wrfout_d0{domain}_{wrf_filename.year:04d}-{wrf_filename.month:02d}-{wrf_filename.day:02d}_{wrf_filename.hour:02d}:{wrf_filename.minute:02d}:00"
    
    # Add your file processing code here
    timeidxlist = np.append(timeidxlist, timeidx)
    filelist = np.append(filelist, pattern)
    # Increment time by 5 minutes
    current_time += timedelta(minutes=5)



# Directory to store saved files
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/"
# Locate radar data directory
radar_data_dir = "/data2/white/DATA/PROJ_LEE/IOP_2/NEXRADLVL2/HAS012534416/"

# ---- End User input for file ----

# Open the NetCDF file
path_a1 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_1/"
path_a2 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_2/"
path_a3 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_3/"


def generate_frame(args):
    file_path_a1, file_path_a2, file_path_a3, timeidx, final_mask = args
    def find_closest_file(target_datetime, directory):
    

        closest_file = None
        closest_diff = None
    
        for filepath in glob.glob(os.path.join(directory, '*.ar2v')):
            # Extract the filename
            filename = os.path.basename(filepath)
            try:
            # Parse the datetime from the filename
                file_datetime = parse_filename_datetime_obs(filename)
            # Calculate the difference between the file's datetime and the target datetime
                diff = abs((file_datetime - target_datetime).total_seconds())
            # Update the closest file if this file is closer
                if closest_diff is None or diff < closest_diff:
                    closest_file = filepath
                    closest_diff = diff
            except ValueError:
            # If the filename does not match the expected format, skip it
                continue
        return closest_file

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
        with Dataset(file_path_a1) as wrfin:
            mdbz_a1 = getvar(wrfin, "mdbz", timeidx=timeidx)
            mdbz_a1 = to_np(mdbz_a1)
        with Dataset(file_path_a2) as wrfin2:
            mdbz_a2 = getvar(wrfin2, "mdbz", timeidx=timeidx)
            mdbz_a2 = to_np(mdbz_a2)
        with Dataset(file_path_a3) as wrfin3:
            mdbz_a3 = getvar(wrfin3, "mdbz", timeidx=timeidx)
            mdbz_a3 = to_np(mdbz_a3)
    
    # Define the format of the datetime string in your filename
        datetime_format = "wrfout_d02_%Y-%m-%d_%H:%M:%S"
    
    # Parse the datetime string into a datetime object
        time_object = datetime.strptime(os.path.basename(file_path_a2), datetime_format)
    
    # Add timeidx value
        add_time = 5 * timeidx
        time_object_adjusted = time_object + timedelta(minutes=add_time)
    
    # Find the closest radar file
        # Locate radar data directory
        radar_data_dir = "/data2/white/DATA/PROJ_LEE/IOP_2/NEXRADLVL2/HAS012534416/"
        closest_file = find_closest_file(time_object_adjusted, radar_data_dir)

    # Get the observed variables
        
        obs_dbz = pyart.io.read_nexrad_archive(closest_file)
        comp_ref = pyart.retrieve.composite_reflectivity(obs_dbz, field="reflectivity")
        comp_ref_data = comp_ref.fields['composite_reflectivity']['data']
        #comp_ref_data = to_np(comp_ref_data)
    # Filters and flattens
        
        filtered_data_a1 = mdbz_a1[final_mask]
        filtered_data_a2 = mdbz_a2[final_mask]
        filtered_data_a3 = mdbz_a3[final_mask]
       
        # Additional mask to include values only over 10 dBZ
        dbz_mask_a1 = filtered_data_a1 > 10
        dbz_mask_a2 = filtered_data_a2 > 10
        dbz_mask_a3 = filtered_data_a3 > 10
        dbz_mask_obs = (comp_ref_data > 10) & (comp_ref_data < 60)
        
        filtered_data_a1 = filtered_data_a1[dbz_mask_a1]
        filtered_data_a2 = filtered_data_a2[dbz_mask_a2]
        filtered_data_a3 = filtered_data_a3[dbz_mask_a3]
        comp_ref_data = comp_ref_data[dbz_mask_obs]
        
        #print('All masks completed')
        # Convert to Z for stats evalutation

        for idx,value in enumerate(filtered_data_a1):
            filtered_data_a1[idx] = 10**(value/10.) # Use linear Z for interpolation
        for idx,value in enumerate(filtered_data_a2):
            filtered_data_a2[idx] = 10**(value/10.) # Use linear Z for interpolation
        for idx,value in enumerate(filtered_data_a3):
            filtered_data_a3[idx] = 10**(value/10.) # Use linear Z for interpolation
        for idx,value in enumerate(comp_ref_data):
            comp_ref_data[idx] = 10**(value/10.) # Use linear Z for interpolation
        #print('Data Converted')

        max_a1 = np.max(filtered_data_a1)
        max_a2 = np.max(filtered_data_a2)
        max_a3 = np.max(filtered_data_a3)
        max_obs = np.max(comp_ref_data)

        #Back to logrithamic
        max_a1 = 10.0 * np.log10(max_a1)
        max_a2 = 10.0 * np.log10(max_a2)
        max_a3 = 10.0 * np.log10(max_a3)
        max_obs = 10.0 * np.log10(max_obs)
        #print('Max calculated')

        min_a1 = np.min(filtered_data_a1)
        min_a2 = np.min(filtered_data_a2)
        min_a3 = np.min(filtered_data_a3)
        min_obs = np.min(comp_ref_data)
        
        #Back to logrithamic
        min_a1 = 10.0 * np.log10(min_a1)
        min_a2 = 10.0 * np.log10(min_a2)
        min_a3 = 10.0 * np.log10(min_a3)
        min_obs = 10.0 * np.log10(min_obs)
        #print("Min Calculated")
        mean_a1 = np.mean(filtered_data_a1)
        mean_a2 = np.mean(filtered_data_a2)
        mean_a3 = np.mean(filtered_data_a3)
        mean_obs = np.mean(comp_ref_data)
        
        #Back to logrithamic
        mean_a1 = 10.0 * np.log10(mean_a1)
        mean_a2 = 10.0 * np.log10(mean_a2)
        mean_a3 = 10.0 * np.log10(mean_a3)
        mean_obs = 10.0 * np.log10(mean_obs)
        #print("Mean calculated")
        median_a1 = ma.median(filtered_data_a1)
        #print(median_a1)
        median_a2 = ma.median(filtered_data_a2)
        #print(median_a2)
        median_a3 = ma.median(filtered_data_a3)
        #print(median_a3)
        median_obs = ma.median(comp_ref_data)
        #print(median_obs)
        #print("Median calculated one")
        #Back to logrithamic
        median_a1 = 10.0 * np.log10(median_a1)
        median_a2 = 10.0 * np.log10(median_a2)
        median_a3 = 10.0 * np.log10(median_a3)
        median_obs = 10.0 * np.log10(median_obs)
        #print("Median calculated")
        std_a1 = np.std(filtered_data_a1)
        std_a2 = np.std(filtered_data_a2)
        std_a3 = np.std(filtered_data_a3)
        std_obs = np.std(comp_ref_data)
    
        #Back to logrithamic
        std_a1 = 10.0 * np.log10(std_a1)
        std_a2 = 10.0 * np.log10(std_a2)
        std_a3 = 10.0 * np.log10(std_a3)
        std_obs = 10.0 * np.log10(std_obs)

        range_a1 = max_a1 - min_a1
        range_a2 = max_a2 - min_a2
        range_a3 = max_a3 - min_a3
        range_obs = max_obs - min_obs

        datetime_obs = parse_filename_datetime_obs(closest_file)
        formatted_datetime_obs = datetime_obs.strftime('%Y-%m-%d %H:%M:%S')
        
        #print("All stats completed")
        if time_object_adjusted.minute == 0 and time_object_adjusted.second == 0:
            print("Attempt 1 at" + str(time_object_adjusted) + " - Max: " + str(round(max_a1,2)) + " Mean: " + str(round(mean_a1,2)) + " Median: " + str(round(median_a1,2)) + " Standard Dev: " + str(round(std_a1, 2)))
            print("Attempt 2 at" + str(time_object_adjusted) + " - Max: " + str(round(max_a2,2)) + " Mean: " + str(round(mean_a2,2)) + " Median: " + str(round(median_a2,2)) + " Standard Dev: " + str(round(std_a2, 2)))
            print("Attempt 3 at" + str(time_object_adjusted) + " - Max: " + str(round(max_a3,2)) + " Mean: " + str(round(mean_a3,2)) + " Median: " + str(round(median_a3,2)) + " Standard Dev: " + str(round(std_a3, 2))) 
            print("Observation at" + formatted_datetime_obs + " - Max: " + str(round(max_obs,2)) + " Mean: " + str(round(mean_obs,2)) + " Median: " + str(round(median_obs,2)) + " Standard Dev: " + str(round(std_obs, 2)))
            print("------------------------------------------------------------")
    except IndexError:
        print("Error processing files")
        
if __name__ == "__main__":
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

    # Sort our files so our data is in order
    wrf_filelist_a1 = []
    wrf_filelist_a2 = []
    wrf_filelist_a3 = []
    
    # Create full directory paths for each attempt based on our times
    for file_pattern in filelist:
        full_pattern_a1 = os.path.join(path_a1, file_pattern)
        full_pattern_a2 = os.path.join(path_a2, file_pattern)
        full_pattern_a3 = os.path.join(path_a3, file_pattern)
        print(full_pattern_a1)
        matched_files_a1 = glob.glob(full_pattern_a1)
        matched_files_a2 = glob.glob(full_pattern_a2)
        matched_files_a3 = glob.glob(full_pattern_a1)
        wrf_filelist_a1.extend(matched_files_a1)
        wrf_filelist_a2.extend(matched_files_a2)
        wrf_filelist_a3.extend(matched_files_a3)

    # List of file paths to the data files
    # Generate tasks
    tasks = []
    for file_path_a1, file_path_a2, file_path_a3 in zip(sorted(wrf_filelist_a1), sorted(wrf_filelist_a2), sorted(wrf_filelist_a3)):
        for timeidx in range(numtimeidx):
            tasks.append((file_path_a1, file_path_a2, file_path_a3, timeidx,final_mask))

    # List of file paths to the data files
    # Generate tasks
        # Use multiprocessing to generate frames in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor: # Use ProcessPoolExecutor for CPU-bound tasks
        print("Starting multiprocessing")
        frame_filenames_gen = executor.map(generate_frame, tasks)
    
    
    # For Normal Processing
    #frame_filenames = []
    #for file_path, timeidx in tasks:
    #    filename = generate_frame(file_path, timeidx)
    #    if filename:
    #        frame_filenames.append(filename)    
    

      
