from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import (get_cmap,ScalarMappable)
import matplotlib as mpl
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
import numpy as np
import matplotlib.colors as colors
from wrf import (to_np,interplevel, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords,extract_times)
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import from_levels_and_colors
import math
import wrffuncs
from datetime import datetime, timedelta
from metpy.plots import USCOUNTIES, ctables

# User input for file
wrf_date_time = datetime(2022,11,18,1,55,00)
domain = 2
numtimeidx = 4
IOP = 2
ATTEMPT = 1
savefig = True
windbarbs = False

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

# Get the maxiumum reflectivity and convert units
mdbz = getvar(ncfile, "mdbz", timeidx=timeidx)
ua  = getvar(ncfile, "ua", units="kt")
va = getvar(ncfile, "va", units="kt")
p = getvar(ncfile, "pressure")
u_500 = interplevel(ua, p, 900)
v_500 = interplevel(va, p, 900)


levels = [10, 15, 20, 25, 30, 35, 40, 45,50,55,60]

nwscmap = ctables.registry.get_colortable('NWSReflectivity')

# Get the latitude and longitude points
lats, lons = latlon_coords(mdbz)

# Get the cartopy mapping object
cart_proj = get_cartopy(mdbz)

# Create a figure
fig = plt.figure(figsize=(30,15))
# Set the GeoAxes to the projection used by WRF
ax = plt.axes(projection=cart_proj)

# Download and add the states, lakes and coastlines
states = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none", name="admin_1_states_provinces")
ax.add_feature(states, linewidth=.1, edgecolor="black")
ax.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
ax.coastlines('50m', linewidth=1)

levels = [10, 15, 20, 25, 30, 35, 40, 45,50,55,60]

nwscmap = ctables.registry.get_colortable('NWSReflectivity')

# Make the filled countours with specified levels and range
qcs = plt.contourf(to_np(lons), to_np(lats),mdbz,levels=levels,transform=crs.PlateCarree(),cmap=nwscmap)

# Add a color bar
cbar = plt.colorbar()
cbar.set_label("dBZ",fontsize=10)
# Set the map bounds
ax.set_xlim(cartopy_xlim(mdbz))
ax.set_ylim(cartopy_ylim(mdbz))

# Add the gridlines
gl = ax.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
gl.xlabel_style = {'rotation': 'horizontal','size': 14,'ha':'center'} # Change 14 to your desired font size
gl.ylabel_style = {'size': 14}  # Change 14 to your desired font size
gl.xlines = True
gl.ylines = True

gl.top_labels = False  # Disable top labels
gl.right_labels = False  # Disable right labels
gl.xpadding = 20


# Add the 500 hPa wind barbs, only plotting every 125th data point.
if windbarbs == True:
	plt.barbs(to_np(lons[::25,::25]), to_np(lats[::25,::25]),
          to_np(u_500[::25, ::25]), to_np(v_500[::25, ::25]),
          transform=crs.PlateCarree(), length=6)

#Adjust format for date to use in figure
date_format = wrf_date_time.strftime("%Y-%m-%d %H:%M:%S")

plt.title(f"Simulated Composite Reflectivty (dBZ) at {date_format}",fontsize="14")

savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{ATTEMPT}/"
plt.savefig(savepath+f"MDBZ{wrf_date_time.year:04d}{wrf_date_time.month:02d}{wrf_date_time.day:02d}{wrf_date_time.hour:02d}{wrf_date_time.minute:02d}D{domain}T{timeidx}A{ATTEMPT}.png")
plt.show()
