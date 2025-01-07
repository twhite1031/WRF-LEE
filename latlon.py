import numpy as np
import matplotlib.pyplot as plt
import os
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords,extract_times, xy_to_ll,ll_to_xy)
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from netCDF4 import Dataset
import cartopy.io.shapereader as shpreader
import pyart


# ---- User input for file ----


# YEAR MONTH DAY HOUR MIN DOMAIN ATTEMPT
date_time = [2022, 11, 18 , "13", "40",2,1]
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{date_time[6]}/"
wrffile = f"wrfout_d0{date_time[5]}_{date_time[0]}-{date_time[1]}-{date_time[2]}_{date_time[3]}:{date_time[4]}:00"

# Locate radar data directory
radar_data_dir = "/data2/white/DATA/PROJ_LEE/IOP_2/NEXRADLVL2/HAS012534416/"

#timeidx in this domain
timeidx = 2

path = "/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_1/"
wrfin = Dataset(path+wrffile)

data = getvar(wrfin, "mdbz", timeidx=timeidx)
lat = getvar(wrfin, "lat", timeidx=timeidx)
lon = getvar(wrfin, "lon", timeidx=timeidx)
#print(to_np(lat))
startofdomain = xy_to_ll(wrfin, 0, 0,timeidx=timeidx)
endofdomain = xy_to_ll(wrfin,283,247,timeidx=timeidx)

print("Start Lat Lon of domain: ", startofdomain)
print("End Lat lon of domain: ", endofdomain)
lat_min, lat_max = 43.1386, 44.2262
lon_min, lon_max = -77.345, -74.468
lat = to_np(lat)
lon = to_np(lon)
lat_mask = (lat > lat_min) & (lat < lat_max)
lon_mask = (lon > lon_min) & (lon < lon_max)
print("Lat mask: ", lat_mask)

# This mask can be used an any data to ensure we are in are modified region
final_mask = lat_mask & lon_mask
print(final_mask)

# This line allows us to keep the 2D Shape, which we would need to plot correctly 
data = to_np(data)
masked_data = np.where(final_mask, data, np.nan)
print(masked_data[masked_data>1])

# Use this to remove nan's for statistical operations
non_nan_mask = ~np.isnan(data)

# Filters and flattens
filtered_data = data[non_nan_mask]

# Get the observed variables
obs_dbz = pyart.io.read_nexrad_archive(closest_file)
display = pyart.graph.RadarMapDisplay(obs_dbz)
comp_ref = pyart.retrieve.composite_reflectivity(obs_dbz, field="reflectivity")
comp_ref_data = comp_ref.fields['composite_reflectivity']['data']

np.mean(filtered_data)
