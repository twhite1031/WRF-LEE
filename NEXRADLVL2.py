import os
import numpy as np
import matplotlib.pyplot as plt
import pyart
import fnmatch
from wrf import (to_np, getvar, latlon_coords,get_cartopy, cartopy_xlim, cartopy_ylim)
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from netCDF4 import Dataset
from matplotlib.colors import from_levels_and_colors
from matplotlib.cm import (get_cmap,ScalarMappable)
from datetime import datetime, timedelta
import wrffuncs
import cartopy.io.shapereader as shpreader
from metpy.plots import USCOUNTIES, ctables


np.set_printoptions(precision=2)

#------------------------------------------------

# User input for file
wrf_date_time = datetime(2022,11,18,1,55,00)
domain = 2
numtimeidx = 4
IOP = 2
ATTEMPT = 1

# Area you would like the plan view to look at (Left Lon, Right Lon, Bottom Lat, Top Lat)
extent = [-77.965996,-75.00000,43.000,44.273301]

wrf_date_time = wrffuncs.round_to_nearest_5_minutes(wrf_date_time)
print("Closest WRF time to match input time: ", wrf_date_time)
timeidx = int((wrf_date_time.minute % 20) // (20 // numtimeidx))
wrf_filename = wrf_date_time
wrf_min = (round((wrf_filename.minute)// 20) * 20)
wrf_filename = wrf_filename.replace(minute=wrf_min)
print("Using data from this file: ", wrf_filename)

# Open the NetCDF file
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_{IOP}/ATTEMPT_{ATTEMPT}/"
pattern = f"wrfout_d0{domain}_{wrf_filename.year:04d}-{wrf_filename.month:02d}-{wrf_filename.day:02d}_{wrf_filename.hour:02d}:{wrf_filename.minute:02d}:00"
ncfile = Dataset(path+pattern)

# Get the maxiumum reflectivity to reference domain
radar_data_dir = "/data2/white/DATA/PROJ_LEE/IOP_2/NEXRADLVL2/HAS012534416/"

# Get the WRF variables
mdbz = getvar(ncfile, "mdbz", timeidx = timeidx)
ua = getvar(ncfile, "ua", units="kt", timeidx= timeidx)
va = getvar(ncfile, "va", units="kt",timeidx= timeidx)

# Get the observed variables
closest_file = wrffuncs.find_closest_radar_file(wrf_date_time, radar_data_dir)
obs_dbz = pyart.io.read_nexrad_archive(closest_file)
display = pyart.graph.RadarMapDisplay(obs_dbz)

# Use these lines to see the fields in the NEXRAD file
#radar_fields = obs_dbz.fields
#print("Observed radar fields: ", radar_fields)

# Get the lat/lon points
lats, lons = latlon_coords(mdbz)

# Get the cartopy projection object
cart_proj = get_cartopy(mdbz)

# Create a figure that will have 3 subplots
fig = plt.figure(figsize=(12,9))

# Set the GeoAxes to the projection used by WRF
ax = plt.axes(projection=cart_proj)

# Download and create the states, land, and oceans using cartopy features
states = cfeature.NaturalEarthFeature(category='cultural', scale='50m',
                                      facecolor='none',
                                      name='admin_1_states_provinces')
land = cfeature.NaturalEarthFeature(category='physical', name='land',
                                    scale='50m',
                                    facecolor=cfeature.COLORS['land'])

lakes = cfeature.NaturalEarthFeature(category='physical',name='lakes',scale='50m',facecolor="none",edgecolor="blue",zorder=2)
ocean = cfeature.NaturalEarthFeature(category='physical', name='ocean',
                                     scale='50m',
                                     facecolor=cfeature.COLORS['water'])
# Special stuff for counties
reader = shpreader.Reader('countyl010g.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, crs.PlateCarree(),zorder=7)

# Add County/State borders
ax.add_feature(COUNTIES, facecolor='none', edgecolor='black',linewidth=1)

ax.set_xlim(cartopy_xlim(mdbz))
ax.set_ylim(cartopy_ylim(mdbz))

gl = ax.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
gl.xlabel_style = {'rotation': 'horizontal','size': 14,'ha':'center'} # Change 14 to your desired font size
gl.ylabel_style = {'size': 14}  # Change 14 to your desired font size
gl.xlines = True
gl.ylines = True

gl.top_labels = False  # Disable top labels
gl.right_labels = False  # Disable right labels
gl.xpadding = 20


levels = [10, 15, 20, 25, 30, 35, 40, 45,50,55,60]

nwscmap = ctables.registry.get_colortable('NWSReflectivity')

#print(obs_dbz.fields)
comp_ref = pyart.retrieve.composite_reflectivity(obs_dbz, field="reflectivity")
display = pyart.graph.RadarMapDisplay(comp_ref)
comp_ref_data = comp_ref.fields['composite_reflectivity']['data']
obs_contour = display.plot_ppi_map("composite_reflectivity",vmin=10,vmax=60,mask_outside=True,ax=ax, colorbar_flag=True, title_flag=False, add_grid_lines=False, cmap=nwscmap,zorder=5)

#Set the x-axis and  y-axis labels
ax.set_xlabel("Longitude", fontsize=8)
ax.set_ylabel("Lattitude", fontsize=8)

# Adjust format for date to use in figure
datetime_obs = wrffuncs.parse_filename_datetime_obs(closest_file)
date_format = datetime_obs.strftime("%Y-%m-%d %H:%M:%S")

ax.set_title("Observed Composite Reflectivity (dBZ) at " + date_format, fontsize=12, fontweight='bold')
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{ATTEMPT}/"

# Show Figure
plt.show()

# Save Figure
plt.savefig(savepath + f"NEXRADplot{wrf_date_time.year:04d}{wrf_date_time.month:02d}{wrf_date_time.day:02d}{wrf_date_time.hour:02d}{wrf_date_time.minute:02d}png")

