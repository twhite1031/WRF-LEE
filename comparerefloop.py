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
from matplotlib.colors import Normalize, from_levels_and_colors
import wrffuncs

# MAY NEED TO USE IN SHELL
#export PROJ_NETWORK=OFF

# ---- User input for file ----
start_time, end_time  = datetime(2022,11,18,12,37,00), datetime(2022, 11, 18, 12, 56, 00)
domain = 3
numtimeidx = 4
attempt = 4


wrf_date_time_start = wrffuncs.round_to_nearest_5_minutes(start_time)
wrf_date_time_end = wrffuncs.round_to_nearest_5_minutes(end_time)
timeidxlist = np.array([],dtype=int)
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
print(filelist)
print(timeidxlist)
# Directory to store saved files
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{attempt}/"
# Locate radar data directory
radar_data_dir = "/data2/white/DATA/PROJ_LEE/IOP_2/NEXRADLVL2/HAS012534416/"

# ---- End User input for file ----

# Open the NetCDF file
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{attempt}/"


def generate_frame(args):
    print("Starting generate frame")
    file_path, timeidx = args
    print(file_path)
    print(timeidx)   
    try:
   
    # Read data from file
        with Dataset(file_path) as wrfin:
            mdbz = getvar(wrfin, "mdbz", timeidx=timeidx)
            print("Read in WRF data")
    # Define the format of the datetime string in your filename
        datetime_format = "wrfout_d03_%Y-%m-%d_%H:%M:%S"
        print("made datetime format")
    # Parse the datetime string into a datetime object
        time_object = datetime.strptime(os.path.basename(file_path), datetime_format)
        print("Made time_object")
    # Add timeidx value
        add_time = 5 * float(timeidx)
        time_object_adjusted = time_object + timedelta(minutes=add_time)
        print("Adjusted WRF time")
    # Find the closest radar file
        # Locate radar data directory
        radar_data_dir = "/data2/white/DATA/PROJ_LEE/IOP_2/NEXRADLVL2/HAS012534416/"
        closest_file = wrffuncs.find_closest_radar_file(time_object_adjusted, radar_data_dir, "KTYX")
        print("Found closest radar file")
    # Get the observed variables
        
        obs_dbz = pyart.io.read_nexrad_archive(closest_file)
        comp_ref = pyart.retrieve.composite_reflectivity(obs_dbz, field="reflectivity")
        display = pyart.graph.RadarMapDisplay(comp_ref)
        print("Got observed variables")
    # Get the cartopy projection object
       # Get the lat/lon points
        lats, lons = latlon_coords(mdbz)

# Get the cartopy projection object
        cart_proj = get_cartopy(mdbz)

# Create a figure that will have 3 subplots
        fig = plt.figure(figsize=(30,15))
        ax_wspd = fig.add_subplot(1,2,1, projection=cart_proj)
        ax_dbz = fig.add_subplot(1,2,2, projection=cart_proj)

# Set the margins to 0
        ax_wspd.margins(x=0,y=0,tight=True)
        ax_dbz.margins(x=0,y=0,tight=True)


# Special stuff for counties
        reader = shpreader.Reader('countyl010g.shp')
        counties = list(reader.geometries())
        COUNTIES = cfeature.ShapelyFeature(counties, crs.PlateCarree(),zorder=7)

# Add County/State borders
        ax_wspd.add_feature(COUNTIES, facecolor='none', edgecolor='black',linewidth=1)
        ax_dbz.add_feature(COUNTIES, facecolor='none', edgecolor='black',linewidth=1)

        ax_dbz.set_xlim(cartopy_xlim(mdbz))
        ax_dbz.set_ylim(cartopy_ylim(mdbz))

        gl = ax_wspd.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
        gl.xlabel_style = {'rotation': 'horizontal','size': 14,'ha':'center'} # Change 14 to your desired font size
        gl.ylabel_style = {'size': 14}  # Change 14 to your desired font size
        gl.xlines = True
        gl.ylines = True

        gl.top_labels = False  # Disable top labels
        gl.right_labels = False  # Disable right labels
        gl.xpadding = 20

        gl2 = ax_dbz.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
        gl2.xlabel_style = {'rotation': 'horizontal','size': 14,'ha':'center'} # Change 14 to your desired font size
        gl2.ylabel_style = {'size': 14}  # Change 14 to your desired font size
        gl2.xlines = True
        gl2.ylines = True
        gl2.top_labels = False  # Disable top labels
        gl2.right_labels = False  # Disable right labels
        gl2.xpadding = 20

        levels = [10, 15, 20, 25, 30, 35, 40, 45,50,55,60]

        nwscmap = ctables.registry.get_colortable('NWSReflectivity')
        mdbz = np.ma.masked_outside(to_np(mdbz),10,65)
        mdbz_contourline = ax_wspd.contourf(to_np(lons), to_np(lats), mdbz,levels=levels,vmin=10,vmax=60,cmap=nwscmap, transform=crs.PlateCarree(),zorder=5)

# CMap for LPI
        dbz_levels = np.arange(5., 75., 5.)
        dbz_rgb = np.array([[255,255,255],[255,255,255],
                    [3,44,244], [3,0,210],
                    [2,253,2], [1,197,1],
                    [0,142,0], [253,248,2],
                    [229,188,0], [253,149,0],
                    [253,0,0], [212,0,0],
                    [188,0,0],[248,0,253]], np.float32) / 255.
        dbz_map, dbz_norm = from_levels_and_colors(dbz_levels, dbz_rgb,extend="max")

# Add LPI to the plot with simulated reflectivity
# Make the filled countours with specified levels and range
#lpi_levels = np.arange(1,6,1)
#lpi_contour = ax_wspd.contourf(to_np(lons), to_np(lats),to_np(lpi),levels=lpi_levels, transform=crs.PlateCarree(),cmap=dbz_map,zorder=2)

        obs_contour = display.plot_ppi_map("composite_reflectivity",vmin=10,vmax=60,mask_outside=True,ax=ax_dbz, colorbar_flag=False, title_flag=False, add_grid_lines=False, cmap=nwscmap,zorder=5)


    # Add the colorbar for the first plot with respect to the position of the plots
        cbar_ax1 = fig.add_axes([ax_dbz.get_position().x1 + 0.01,
                         ax_dbz.get_position().y0,
                         0.02,
                         ax_dbz.get_position().height])
        cbar1 = fig.colorbar(mdbz_contourline, cax=cbar_ax1)
        cbar1.set_label("dBZ", fontsize=12)
        cbar1.ax.tick_params(labelsize=10)

    # Set the view of the plot
        ax_dbz.set_extent([-77.965996,-75.00000,43.000,44.273301],crs=crs.PlateCarree())
        ax_wspd.set_extent([-77.965996,-75.00000,43.0000,44.273301],crs=crs.PlateCarree())

    #Set the x-axis and  y-axis labels
        ax_dbz.set_xlabel("Longitude", fontsize=8)
        ax_wspd.set_ylabel("Lattitude", fontsize=8)
        ax_dbz.set_ylabel("Lattitude", fontsize=8)
 
  
    # Format the datetime into a more readable format
        datetime_obs = wrffuncs.parse_filename_datetime_obs(closest_file)
        formatted_datetime_obs = datetime_obs.strftime('%Y-%m-%d %H:%M:%S')
        print("Made formatted datetime for obs")
        #ax_a1.set_title(f"Attempt 1 at " + str(time_object_adjusted),fontsize=12,fontweight='bold')
        #ax_a2.set_title(f"Attempt 2 at " + str(time_object_adjusted),fontsize=12,fontweight='bold')
        #ax_a3.set_title(f"Attempt 3 at " + str(time_object_adjusted),fontsize=12,fontweight='bold')
        #ax_obs.set_title(f"Observation at" + formatted_datetime_obs, fontsize=12,fontweight='bold')

    # Save the figure to a file
        frame_number = os.path.splitext(os.path.basename(file_path))[0]
        filename = f'frame_{frame_number}{timeidx}.png'
        plt.savefig(filename)
        print("Saving Frame")
    #print(f"{os.path.basename(file_path)} Processed!")
        if time_object_adjusted.day == 19 and time_object_adjusted.hour == 0 and time_object_adjusted.minute == 0:
            plt.show()
        plt.close()

        return filename
    except ValueError:
        print("Error processing files")
        print(os.path.basename(file_path)[0])
        
def create_gif(frame_filenames, output_filename):

    frames = []
    for filename in frame_filenames:
            new_frame = Image.open(filename)
            frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(savepath + output_filename, format='GIF', append_images=frames[1:],save_all=True,duration=75, loop=0)
    print("GIF created")
if __name__ == "__main__":

    # Sort our files so our data is in order
    wrf_filelist = []
    for file_pattern in filelist:
        full_pattern = os.path.join(path, file_pattern)
        matched_files = glob.glob(full_pattern)
        wrf_filelist.extend(matched_files)
    
      
    # List of file paths to the data files
    # Generate tasks
    tasks = list(zip(wrf_filelist, timeidxlist))
    output_gif = f'2panelcomparerefloopD{domain}A{attempt}{wrf_date_time_start.month:02d}{wrf_date_time_start.day:02d}{wrf_date_time_start.hour:02d}{wrf_date_time_start.minute:02d}to{wrf_date_time_end.month:02d}{wrf_date_time_end.day:02d}{wrf_date_time_end.hour:02d}{wrf_date_time_end.minute:02d}.gif'
    print("Finished gathering tasks")
    print(tasks)
    # Use multiprocessing to generate frames in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor: # Use ProcessPoolExecutor for CPU-bound tasks
        print("Starting multiprocessing")
        frame_filenames_gen = executor.map(generate_frame, tasks)
        frame_filenames = list(frame_filenames_gen)  # Convert generator to list
    
    
    # For Normal Processing
    #frame_filenames = []
    #for file_path, timeidx in tasks:
    #    filename = generate_frame(file_path, timeidx)
    #    if filename:
    #        frame_filenames.append(filename)    
    

    # Create the GIF
    filtered_list = [filename for filename in frame_filenames if filename is not None] 
    print(filtered_list)
    create_gif(sorted(filtered_list), output_gif)

    # Clean up the frame files
    for filename in filtered_list:
        print("Removing: ", filename)
        os.remove(filename)
    
