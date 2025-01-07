import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from metpy.plots import ctables
from wrf import (getvar, interpline, interplevel, to_np, vertcross, smooth2d, CoordPair, GeoBounds,
                 get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim)
import pyart
from matplotlib.colors import Normalize, from_levels_and_colors
import cartopy.io.shapereader as shpreader
from pyxlma.lmalib.io import read as lma_read
import os
import glob
import wrffuncs
from datetime import datetime, timedelta

# User input for file
wrf_date_time = datetime(2022,11,19,2,50,00)
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

filepath = path+pattern
ncfile = Dataset(path+pattern)

# Locate radar data directory
radar_data_dir = "/data2/white/DATA/PROJ_LEE/IOP_2/NEXRADLVL2/HAS012534416/"

adjusted_datetime_of_wrf = wrffuncs.parse_filename_datetime_wrf(filepath, timeidx)
print("Real time of WRF file:", adjusted_datetime_of_wrf)

closest_file = wrffuncs.find_closest_radar_file(adjusted_datetime_of_wrf, radar_data_dir, "KTYX")
print("closest file: ", closest_file)

# Open the LMA NetCDF file
lmapath = "/data2/white/DATA/PROJ_LEE/IOP_2/LMADATA/" 
lmafilename = "LYLOUT_221119_001000_0600.dat.gz"
ds, starttime  = lma_read.dataset(lmapath + lmafilename)

# Get the WRF variables
mdbz = getvar(ncfile, "mdbz", timeidx = timeidx)
ua = getvar(ncfile, "ua", units="kt", timeidx= timeidx)
va = getvar(ncfile, "va", units="kt",timeidx= timeidx)

# Get the observed variables
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
fig = plt.figure(figsize=(30,15))
ax_wspd = fig.add_subplot(1,2,1, projection=cart_proj)
ax_dbz = fig.add_subplot(1,2,2, projection=cart_proj)

# Set the margins to 0
ax_wspd.margins(x=0,y=0,tight=True)
ax_dbz.margins(x=0,y=0,tight=True)


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

#print(obs_dbz.fields)
comp_ref = pyart.retrieve.composite_reflectivity(obs_dbz, field="reflectivity")
display = pyart.graph.RadarMapDisplay(comp_ref)
comp_ref_data = comp_ref.fields['composite_reflectivity']['data']
print(comp_ref_data)
obs_contour = display.plot_ppi_map("composite_reflectivity",vmin=10,vmax=60,mask_outside=True,ax=ax_dbz, colorbar_flag=False, title_flag=False, add_grid_lines=False, cmap=nwscmap,zorder=5)

lat_lon = [43.8883, -76.1567]

ax_dbz.plot(lat_lon[1], lat_lon[0], marker='^', color='purple', transform=crs.PlateCarree(),markersize=10,zorder=6)  # 'ro' means red color ('r') and circle marker ('o')
ax_wspd.plot(lat_lon[1], lat_lon[0],marker='^', color='purple', transform=crs.PlateCarree(), markersize=10,zorder=6)  # 'ro' means red color ('r') and circle marker ('o')

# Plot LMA event locations
station_filter = (ds.event_stations >= 7)
chi_filter = (ds.event_chi2 <= 1.0)

filter_subset = {'number_of_events':(chi_filter & station_filter)}

# Use this line to filter number of events by slices
#filter1 = ds[filter_subset].isel({'number_of_events':slice(0,10)})

# note that we first filter on all the data select 10000 points, and then on that dataset we further filter 
#art = ds[filter_subset].plot.scatter(x='event_longitude', y='event_latitude',c='black', ax=ax_wspd, 
#                    s=10, vmin=0.0,edgecolor='black',vmax=5000,transform=crs.PlateCarree(),zorder=4)

# Lat and Long of human flash observation(s)
#ax_wspd.scatter([-76.023498],[43.990428],c='red',s=75,marker="X",transform=crs.PlateCarree(),zorder=5)

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

formatted_datetime_wrf = adjusted_datetime_of_wrf.strftime('%Y-%m-%d %H:%M:%S')

# Add a title
ax_wspd.set_title("Simulated Composite Reflectivity (dBZ) at " + formatted_datetime_wrf, fontsize=12, fontweight='bold')
ax_dbz.set_title("Observed Composite Reflectivity (dBZ) at " + formatted_datetime_obs, fontsize=12, fontweight='bold')
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{ATTEMPT}/"

# Save Figure
plt.savefig(savepath + f"compareref{wrf_date_time.year:04d}{wrf_date_time.month:02d}{wrf_date_time.day:02d}{wrf_date_time.hour:02d}{wrf_date_time.minute:02d}D{domain}T{timeidx}A{ATTEMPT}.png")

# Show Figure
plt.show()
