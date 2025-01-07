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

# User input for file
year = "2022"
month = "11"
day = "18"
hour = "13"
minute = "40"
domain = "2"
timeidx = 2
IOP = 2
ATTEMPT = "1"
var = "LIGHTDENS"

# Open the NetCDF file
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_{IOP}/ATTEMPT_{ATTEMPT}/"
pattern = f"wrfout_d0{domain}_{year}-{month}-{day}_{hour}:{minute}:00"
ncfile = Dataset(path+pattern)

# Get the maxiumum reflectivity and convert units
plot_var = getvar(ncfile, var, timeidx=timeidx)


# Get the latitude and longitude points
lats, lons = latlon_coords(plot_var)

# Get the cartopy mapping object
cart_proj = get_cartopy(plot_var)

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

qcs = plt.contourf(to_np(lons), to_np(lats),plot_var,transform=crs.PlateCarree(),cmap="jet")

# Add a color bar
plt.colorbar()
ax.set_extent([-77.965996,-75.00000,43.000,44.273301],crs=crs.PlateCarree())
# Set the map bounds
#ax.set_xlim(cartopy_xlim(plot_var))
#ax.set_ylim(cartopy_ylim(plot_var))
print(np.max(to_np(plot_var)))

# Add the gridlines
plt.title(f"{var} at {year}{month}{day}{hour}{minute}",{"fontsize" : 14})
plt.show()
