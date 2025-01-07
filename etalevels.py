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
day = "19"
hour = "00"
minute = "00"
domain = "2"
timeidx = 0
IOP = 2
ATTEMPT = "1B"
savefig = True	

# Open the NetCDF file

path = f"/data1/white/WRF_Outputs/PROJ_LEE/IOP_{IOP}/ATTEMPT_{ATTEMPT}/"
pattern = f"wrfout_d0{domain}_{year}-{month}-{day}_{hour}:{minute}:00"
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

# Add the gridlines
plt.title(f"Lighting Potenital Index (J/kg) at {year}{month}{day}{hour}{minute}",{"fontsize" : 14})
if savefig == True:
	plt.savefig(path+f"LPI{year}{month}{day}{hour}{minute}A{ATTEMPT}.png")
plt.show()
