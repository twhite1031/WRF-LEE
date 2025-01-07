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
import cartopy.io.shapereader as shpreader

# ---- User input for file ----

# MUST USE
#export PROJ_NETWORK=OFF
#export PROJ_USER_WRITABLE_DIRECTORY=/data2/white/PYTHON_SCRIPTS/PROJ_LEE/tmp/proj

# Domain to use
domain = 2

# Attempt to use
attempt = 3

# Number of timeidx in this domain
numtimeidx = 4

# Height index if data is 3D
heightidx = 0

# Variable to use to create many maps
var = "ELECMAG"

savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{attempt}/"

# ---- End User input for file ----

# Open the NetCDF file
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{attempt}"
def generate_frame(args):
    try:
    # Seperate these for normal processing and place in args ^
        file_path, timeidx = args
        # Read data from file
        with Dataset(file_path) as wrfin:
            data = getvar(wrfin, var, timeidx=timeidx)
            pos_ic = getvar(wrfin, "FLSHFEDICP", timeidx=timeidx)
            neg_ic = getvar(wrfin, "FLSHFEDICN", timeidx=timeidx)
            pos_cg = getvar(wrfin, "FLSHFEDCGP", timeidx=timeidx)
            neg_cg = getvar(wrfin, "FLSHFEDCGN", timeidx=timeidx)
            light = getvar(wrfin, "LIGHT", timeidx=timeidx)
            if np.any(to_np(light) > 0): 
                print("Lightning! at:" + os.path.basename(file_path) + "Timeindex: " + str(timeidx))
            if np.any(to_np(pos_ic) > 0):
                print("Positive IC Flash!")
            if np.any(to_np(neg_ic) > 0):
                print("Negative IC Flash!")
            if np.any(to_np(pos_cg) > 0):
                print("Positive CG Flash!")
            if np.any(to_np(neg_cg) > 0):
                print("Negative CG Flash!")
   
    except TypeError:
        print("Error processing files")

if __name__ == "__main__":

    # Sort our files so our data is in order
    wrf_filelist = glob.glob(os.path.join(path,f'wrfout_d0{domain}_2022-11-1*'))
    tasks = [(file_path, timeidx) for file_path in wrf_filelist for timeidx in range(numtimeidx)] 
    # List of file paths to the data files
    data_filepaths = wrf_filelist
    output_gif = f'{var}loopD{domain}A{attempt}.gif'

    # Use multiprocessing to generate frames in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor:  # Use ProcessPoolExecutor for CPU-bound tasksI
    
        frame_filenames_gen = executor.map(generate_frame, tasks)
        
        
