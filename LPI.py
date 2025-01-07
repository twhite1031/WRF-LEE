from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import (get_cmap,ScalarMappable)
import matplotlib as mpl
from matplotlib.colors import from_levels_and_colors
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
import numpy as np
import matplotlib.colors as colors
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords,extract_times)
import wrffuncs
from datetime import datetime, timedelta

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
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/_IOP_{IOP}/ATTEMPT_{ATTEMPT}/"
pattern = f"wrfout_d0{domain}_{wrf_filename.year:04d}-{wrf_filename.month:02d}-{wrf_filename.day:02d}_{wrf_filename.hour:02d}:{wrf_filename.minute:02d}:00"
ncfile = Dataset(path+pattern)

# Get the maxiumum reflectivity and convert units

lpi = getvar(ncfile, "LPI", timeidx=timeidx)

# Make Official Radar Colormap
dbz_levels = np.arange(5., 75., 5.)
dbz_rgb = np.array([[3,112,255],
                    [3,44,244], [3,0,210],
                    [2,253,2], [1,197,1],
                    [0,142,0], [253,248,2],
                    [229,188,0], [253,149,0],
                    [253,0,0], [212,0,0],
                    [188,0,0],[248,0,253],
                    [152,84,198]], np.float32) / 255.0

dbz_map, dbz_norm = from_levels_and_colors(dbz_levels, dbz_rgb,extend="max")

# Get the latitude and longitude points
lats, lons = latlon_coords(lpi)

# Get the cartopy mapping object
cart_proj = get_cartopy(lpi)

# Create a figure
fig = plt.figure(figsize=(30,15))
# Set the GeoAxes to the projection used by WRF
ax = plt.axes(projection=cart_proj)

# Download and add the states, lakes and coastlines
states = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none", name="admin_1_states_provinces")
ax.add_feature(states, linewidth=1, edgecolor="black")
ax.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
ax.coastlines('50m', linewidth=1)

# Make the filled countours with specified levels and range
levels = [1 + 1+n for n in range(9)]
qcs = plt.contourf(to_np(lons), to_np(lats),lpi,levels=levels, transform=crs.PlateCarree(),cmap="jet",vmin=0,vmax=10)

# Add a color bar
plt.colorbar(ScalarMappable(norm=qcs.norm, cmap=qcs.cmap))

# Set the map bounds
ax.set_xlim(cartopy_xlim(lpi))
ax.set_ylim(cartopy_ylim(lpi))

# Adjust format for date to use in figure
date_format = wrf_date_time.strftime("%Y-%m-%d %H:%M:%S")

# Add the gridlines
plt.title(f"Lighting Potenital Index (J/kg) at {date_format}",fontsize="14")

savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{ATTEMPT}/"
plt.savefig(savepath+f"LPI{wrf_date_time.year:04d}{wrf_date_time.month:02d}{wrf_date_time.day:02d}{wrf_date_time.hour:02d}{wrf_date_time.minute:02d}D{domain}T{timeidx}A{ATTEMPT}.png")

plt.show()
