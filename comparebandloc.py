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
from datetime import datetime,timedelta
import cartopy.io.shapereader as shpreader
import pyart
import multiprocessing as mp
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import wrffuncs
# ---- User input for file ----

# MUST USE
#export PROJ_NETWORK=OFF
#export PROJ_USER_WRITABLE_DIRECTORY=/data2/white/PYTHON_SCRIPTS/PROJ_LEE/tmp/proj

start_time, end_time  = datetime(2022,11,18,12,37,00), datetime(2022, 11, 18, 15, 46, 00)
domain = 2# Add a shared title at the top with the time label

numtimeidx = 4

wrf_date_time_start = wrffuncs.round_to_nearest_5_minutes(start_time)
wrf_date_time_end = wrffuncs.round_to_nearest_5_minutes(end_time)
timeidxlist = np.array([])
filelist = np.array([])
while wrf_date_time_start <= wrf_date_time_end:
    timeidx = int((wrf_date_time_start.minute % 20) // (20 // numtimeidx))
    print("Using timeidx: ", timeidx)
    wrf_min = (round((wrf_date_time_start.minute)// 20) * 20)
    wrf_filename = wrf_date_time_start.replace(minute=wrf_min)
    print(f"Using data from file: {wrf_filename}, time index: {timeidx}")
    pattern = f"wrfout_d0{domain}_{wrf_filename.year:04d}-{wrf_filename.month:02d}-{wrf_filename.day:02d}_{wrf_filename.hour:02d}:{wrf_filename.minute:02d}:00"
    
    # Add your file processing code here
    timeidxlist = np.append(timeidxlist, timeidx)
    filelist = np.append(filelist, pattern)
    # Increment time by 5 minutes
    wrf_date_time_start += timedelta(minutes=5)

print(timeidxlist)
print(filelist)


# Path to WRF data
path_a1 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_1/"
path_a2 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_2/"
path_a3 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_3/"

# Directory to store saved files
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/"
# Locate radar data directory
radar_data_dir = "/data2/white/DATA/PROJ_LEE/IOP_2/NEXRADLVL2/HAS012534416/"


def generate_frame(args):
    print("Starting generate frame")
    file_path_a1, file_path_a2, file_path_a3, timeidx = args
    print(file_path_a1)
    print(file_path_a2)
    print(file_path_a3)
    print(timeidx)
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
        with Dataset(file_path_a2) as wrfin2:
            mdbz_a2 = getvar(wrfin2, "mdbz", timeidx=timeidx)
        with Dataset(file_path_a3) as wrfin3:
            mdbz_a3 = getvar(wrfin3, "mdbz", timeidx=timeidx)
        print("Read in WRF data")
    # Define the format of the datetime string in your filename
        datetime_format = "wrfout_d02_%Y-%m-%d_%H:%M:%S"
        print("made datetime format")
    # Parse the datetime string into a datetime object
        time_object = datetime.strptime(os.path.basename(file_path_a2), datetime_format)
        print("Made time_object")
    # Add timeidx value
        add_time = 5 * timeidx
        time_object_adjusted = time_object + timedelta(minutes=add_time)
        print("Adjusted WRF time")
    # Find the closest radar file
        # Locate radar data directory
        radar_data_dir = "/data2/white/DATA/PROJ_LEE/IOP_2/NEXRADLVL2/HAS012534416/"
        closest_file = find_closest_file(time_object_adjusted, radar_data_dir)
        print("Found closest radar file")
    # Get the observed variables
        
        obs_dbz = pyart.io.read_nexrad_archive(closest_file)
        display = pyart.graph.RadarMapDisplay(obs_dbz)
        print("Got observed variables")
    # Get the cartopy projection object
        cart_proj = get_cartopy(mdbz_a1)

    # Create a figure
        fig = plt.figure(figsize=(30,15),facecolor='white')
        ax_a1 = fig.add_subplot(2,2,1, projection=cart_proj)
        ax_a2 = fig.add_subplot(2,2,2, projection=cart_proj)
        ax_a3 = fig.add_subplot(2,2,3, projection=cart_proj)
        ax_obs = fig.add_subplot(2,2,4, projection=cart_proj)
        print("Created Figures")

    # Get the latitude and longitude points
        lats, lons = latlon_coords(mdbz_a1)

    # Download and add the states, lakes  and coastlines
        states = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none", name="admin_1_states_provinces")
        ax_a1.add_feature(states, linewidth=.1, edgecolor="black")
        ax_a1.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
        ax_a1.coastlines('50m', linewidth=1)
        ax_a1.add_feature(USCOUNTIES, alpha=0.1)
        print("Made land features")
    # Set the map bounds
        ax_a1.set_xlim(cartopy_xlim(mdbz_a1))
        ax_a1.set_ylim(cartopy_ylim(mdbz_a2))
        print("Set map bounds")
    # Add the gridlines
        gl_a1 = ax_a1.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=True)
        gl_a1.xlabel_style = {'rotation': 'horizontal','size': 10,'ha':'center'} # Change 14 to your desired font size
        gl_a1.ylabel_style = {'size': 10}  # Change 14 to your desired font size
        gl_a1.xlines = True
        gl_a1.ylines = True
        gl_a1.top_labels = False  # Disable top labels
        gl_a1.right_labels = True # Disable right labels
        gl_a1.xpadding = 20
        gl_a1.xlocator = mticker.FixedLocator([-77.5,-76.5,-75.5])  # Set longitude gridlines every 10 degrees
        gl_a1.ylocator = mticker.FixedLocator(np.arange(41, 47, .5))    # Set latitude gridlines every 10 degrees

        print("Made gridlines")
    # Download and add the states, lakes  and coastlines
        ax_a2.add_feature(states, linewidth=.1, edgecolor="black")
        ax_a2.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
        ax_a2.coastlines('50m', linewidth=1)
        ax_a2.add_feature(USCOUNTIES, alpha=0.1)

    # Set the map bounds
        ax_a2.set_xlim(cartopy_xlim(mdbz_a1))
        ax_a2.set_ylim(cartopy_ylim(mdbz_a2))

    # Add the gridlines
        gl_a2 = ax_a2.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=True)
        gl_a2.xlabel_style = {'rotation': 'horizontal','size': 10,'ha':'center'} # Change 14 to your desired font size
        gl_a2.ylabel_style = {'size': 10}  # Change 14 to your desired font size
        gl_a2.xlines = True
        gl_a2.ylines = True
        gl_a2.top_labels = False  # Disable top labels
        gl_a2.right_labels = True  # Disable right labels
        gl_a2.xpadding = 20
        # Increase the number of gridlines by setting locators
        gl_a2.xlocator = mticker.FixedLocator([-77.5,-76.5,-75.5])  # Set longitude gridlines every 10 degrees
        gl_a2.ylocator = mticker.FixedLocator(np.arange(41, 47, .5))    # Set latitude gridlines every 10 degrees

        # Format the gridline labels
        gl_a2.xformatter = LONGITUDE_FORMATTER
        gl_a2.yformatter = LATITUDE_FORMATTER
        

    # Download and add the states, lakes  and coastlines
        ax_a3.add_feature(states, linewidth=.1, edgecolor="black")
        ax_a3.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
        ax_a3.coastlines('50m', linewidth=1)
        ax_a3.add_feature(USCOUNTIES, alpha=0.1)

    # Set the map bounds
        ax_a3.set_xlim(cartopy_xlim(mdbz_a1))
        ax_a3.set_ylim(cartopy_ylim(mdbz_a2))

    # Add the gridlines
        gl_a3 = ax_a3.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=True)
        gl_a3.xlabel_style = {'rotation': 'horizontal','size': 10,'ha':'center'} # Change 14 to your desired font size
        gl_a3.ylabel_style = {'size': 10}  # Change 14 to your desired font size
        gl_a3.xlines = True
        gl_a3.ylines = True
        gl_a3.top_labels = False  # Disable top labels
        gl_a3.right_labels = True  # Disable right labels
        gl_a3.xpadding = 20
        gl_a3.xlocator = mticker.FixedLocator([-77.5,-76.5,-75.5])  # Set longitude gridlines every 10 degrees
        gl_a3.ylocator = mticker.FixedLocator(np.arange(41, 47, .5))    # Set latitude gridlines every 10 degrees

    
    # Download and add the states, lakes  and coastlines
        ax_obs.add_feature(states, linewidth=.1, edgecolor="black")
        ax_obs.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
        ax_obs.coastlines('50m', linewidth=1)
        ax_obs.add_feature(USCOUNTIES, alpha=0.1)

    # Set the map bounds
        ax_obs.set_xlim(cartopy_xlim(mdbz_a1))
        ax_obs.set_ylim(cartopy_ylim(mdbz_a2))

    # Add the gridlines
        gl_obs = ax_obs.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=True)
        gl_obs.xlabel_style = {'rotation': 'horizontal','size': 10,'ha':'center'} # Change 14 to your desired font size
        gl_obs.ylabel_style = {'size': 10}  # Change 14 to your desired font size
        gl_obs.xlines = True
        gl_obs.ylines = True
        gl_obs.top_labels = False  # Disable top labels
        gl_obs.right_labels = True  # Disable right labels
        gl_obs.xpadding = 20
        gl_obs.xlocator = mticker.FixedLocator([-77.5,-76.5,-75.5])  # Set longitude gridlines every 10 degrees
        gl_obs.ylocator = mticker.FixedLocator(np.arange(41, 47, .5))    # Set latitude gridlines every 10 degrees

    # Get composite reflectivity from observed LVL2 data
        nwscmap = ctables.registry.get_colortable('NWSReflectivity')

        comp_ref = pyart.retrieve.composite_reflectivity(obs_dbz, field="reflectivity")
        display = pyart.graph.RadarMapDisplay(comp_ref)
        print("Calculated composite reflectivity")
        obs_contour = display.plot_ppi_map("composite_reflectivity",vmin=10,vmax=60,mask_outside=True,ax=ax_obs, colorbar_flag=False, title_flag=False, add_grid_lines=False, cmap=nwscmap)
        print("Made observation contours")
    # Read in cmap and map contours
        levels = [10, 15, 20, 25, 30, 35, 40, 45,50,55,60]


        mdbz_a1 = np.ma.masked_outside(to_np(mdbz_a1),10,65)
        mdbz_a2 = np.ma.masked_outside(to_np(mdbz_a2),10,65)
        mdbz_a3 = np.ma.masked_outside(to_np(mdbz_a3),10,65)

        mdbz_a1_contour = ax_a1.contour(to_np(lons), to_np(lats), mdbz_a1,levels=[20],vmin=10,vmax=60,cmap=nwscmap, transform=crs.PlateCarree())

        mdbz_a2_contour = ax_a2.contour(to_np(lons), to_np(lats), mdbz_a2,levels=[20],vmin=10,vmax=60,cmap=nwscmap, transform=crs.PlateCarree())

        mdbz_a3_contour = ax_a3.contour(to_np(lons), to_np(lats), mdbz_a3,levels=[20],vmin=10,vmax=60,cmap=nwscmap, transform=crs.PlateCarree())
        print("Made all contours")
    # Add the colorbar for the first plot with respect to the position of the plots
        #cbar_a2 = fig.add_axes([ax_a2.get_position().x1 + 0.01,
        #                ax_a2.get_position().y0,
        #                 0.02,
        #                 ax_a2.get_position().height])

        #cbar1 = fig.colorbar(mdbz_a2, cax=cbar_a2)
        #cbar1.set_label("dBZ", fontsize=12)
        #cbar1.ax.tick_params(labelsize=10)

    # Format the datetime into a more readable format
        datetime_obs = parse_filename_datetime_obs(closest_file)
        formatted_datetime_obs = datetime_obs.strftime('%Y-%m-%d %H:%M:%S')
        print("Made formatted datetime for obs")
        ax_a1.set_title(f"Attempt 1 at " + str(time_object_adjusted),fontsize=12,fontweight='bold')
        ax_a2.set_title(f"Attempt 2 at " + str(time_object_adjusted),fontsize=12,fontweight='bold')
        ax_a3.set_title(f"Attempt 3 at " + str(time_object_adjusted),fontsize=12,fontweight='bold')
        ax_obs.set_title(f"Observation at " + formatted_datetime_obs, fontsize=12,fontweight='bold')

    # Save the figure to a file
        frame_number = os.path.splitext(os.path.basename(file_path_a1))[0]
        filename = f'frame_{frame_number}{timeidx}.png'
        plt.savefig(filename)
        print("Saving Frame")
    #print(f"{os.path.basename(file_path)} Processed!")
        plt.close()

        return filename
    except IndexError:
        print("Error processing files")
        
def create_gif(frame_filenames, output_filename):

    frames = []
    for filename in frame_filenames:
            new_frame = Image.open(filename)
            frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(savepath + output_filename, format='GIF', append_images=frames[1:],save_all=True,duration=75, loop=0)
    
if __name__ == "__main__":
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
    print(wrf_filelist_a1)
    # List of file paths to the data files
    # Generate tasks
    tasks = []
    for file_path_a1, file_path_a2, file_path_a3 in zip(sorted(wrf_filelist_a1), sorted(wrf_filelist_a2), sorted(wrf_filelist_a3)):
        for timeidx in range(numtimeidx):
            tasks.append((file_path_a1, file_path_a2, file_path_a3, timeidx))
    output_gif = f'4panelcomparerefloopD{domain}.gif'
    print("Finished gathering tasks")
    print(tasks[:5])
    # Use multiprocessing to generate frames in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor: # Use ProcessPoolExecutor for CPU-bound tasks
        print("Starting multiprocessing")
        frame_filenames_gen = executor.map(generate_frame, tasks)
        frame_filenames = list(frame_filenames_gen)  # Convert generator to list
       
    # Create the GIF
    filtered_list = [filename for filename in frame_filenames if filename is not None]    
    create_gif(sorted(filtered_list), output_gif)

    # Clean up the frame files
    for filename in filtered_list:
        print("Removing: ", filename)
        os.remove(filename)
    
