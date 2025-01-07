import numpy as np
from matplotlib import pyplot
from matplotlib.cm import get_cmap
from matplotlib.colors import from_levels_and_colors
from cartopy import crs
from cartopy.feature import NaturalEarthFeature, COLORS
from netCDF4 import Dataset
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,
                 cartopy_xlim, cartopy_ylim, interpline, CoordPair, xy_to_ll)
from scipy.ndimage import label
import cartopy.feature as cfeature
import wrffuncs
from datetime import datetime, timedelta

# User input for file
wrf_date_time = datetime(2022,11,18,1,55,00)
domain = 2
numtimeidx = 4
IOP = 2
ATTEMPT = 1

# Open the NetCDF file

wrf_date_time = wrffuncs.round_to_nearest_5_minutes(wrf_date_time)
print("Closest WRF time to match input time: ", wrf_date_time)
timeidx = int((wrf_date_time.minute % 20) // (20 // numtimeidx))
wrf_filename = wrf_date_time
wrf_min = (round((wrf_filename.minute)// 20) * 20)
wrf_filename = wrf_filename.replace(minute=wrf_min)
print("Using data from this file: ", wrf_filename)

path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_{IOP}/ATTEMPT_{ATTEMPT}/"
pattern = f"wrfout_d0{domain}_{wrf_filename.year:04d}-{wrf_filename.month:02d}-{wrf_filename.day:02d}_{wrf_filename.hour:02d}:{wrf_filename.minute:02d}:00"
wrf_file = Dataset(path+pattern)

# Define the cross section start and end points
cross_start = CoordPair(lat=44.00, lon=-76.75)
cross_end = CoordPair(lat=43.73, lon=-75.5)

# Get the WRF variables
ht = getvar(wrf_file, "z", timeidx=timeidx)
ter = getvar(wrf_file, "ter", timeidx=timeidx)
dbz = getvar(wrf_file, "dbz", timeidx=timeidx)
elecmag = getvar(wrf_file, "ELECMAG", timeidx=timeidx)
max_dbz = getvar(wrf_file, "mdbz", timeidx=timeidx)
Z = 10**(dbz/10.) # Use linear Z for interpolation
lats = getvar(wrf_file, "lat", timeidx=timeidx)
lons = getvar(wrf_file, "lon",timeidx=timeidx)
lats = to_np(lats)
lons = to_np(lons)
ctt = getvar(wrf_file, "ctt",timeidx=timeidx)

# Threshold to identify the snow band (e.g., reflectivity > 20 dBZ)
threshold = 25 

max_dbz = to_np(max_dbz)
snow_band = max_dbz > threshold
print(snow_band)

# Label the connected regions in the snow band
labeled_array, num_features = label(snow_band)

# Find the largest connected region
region_sizes = np.bincount(labeled_array.flatten())

largest_region_label = np.argmax(region_sizes[1:]) + 1
print(largest_region_label)

largest_region = (labeled_array == largest_region_label)
print("Largest region: ", largest_region)

# Get the coordinates of the largest region
lat_indices, lon_indices= np.where(largest_region)

#print("Lats: ", lat_indices)
#print("Lons: ", lon_indices)<MouseMove>
lat_lon_indices = list(zip(lon_indices, lat_indices))
print(lat_lon_indices)
start_coords = xy_to_ll(wrf_file, lat_lon_indices[0][0], lat_lon_indices[0][1],timeidx=timeidx)
end_coords = xy_to_ll(wrf_file, lat_lon_indices[-1][0], lat_lon_indices[-1][1],timeidx=timeidx)
print("start_coords", start_coords)
print("end_coords", end_coords)
# Define start and end points as the first and last points in the snow band
#start_point_idx = coords[0]
#end_point_idx = coords[-1]
print("Calculated Starting Lat: ", start_coords[0])
print("Calculated Starting Lon: ", start_coords[1])
print("Calculated Ending Lat: ", end_coords[0])
print("Calculated Ending Lon: ", end_coords[1])

# Convert indices to coordinates
start_point = CoordPair(lat=float(start_coords[0]), lon=float(start_coords[1]))
end_point = CoordPair(lat=float(end_coords[0]), lon=float(end_coords[1]))

#print(f"Start Point: {start_point}, End Point: {end_point}")
# Compute the vertical cross-section interpolation.  Also, include the lat/lon points along the cross-section in the metadata by setting latlon to True.
z_cross = vertcross(Z, ht, wrfin=wrf_file, start_point=start_point, end_point=end_point, latlon=True, meta=True)
elec_cross = vertcross(elecmag, ht, wrfin=wrf_file, start_point=start_point, end_point=end_point, latlon=True, meta=True)


# Convert back to dBz after interpolation
dbz_cross = 10.0 * np.log10(z_cross)

# Add back the attributes that xarray dropped from the operations above
dbz_cross.attrs.update(dbz_cross.attrs)
dbz_cross.attrs["description"] = "radar reflectivity cross section"
dbz_cross.attrs["units"] = "dBZ"

# To remove the slight gap between the dbz contours and terrain due to the
# contouring of gridded data, a new vertical grid spacing, and model grid
# staggering, fill in the lower grid cells with the first non-missing value
# for each column.

# Make a copy of the z cross data. Let's use regular numpy arrays for this.
dbz_cross_filled = np.ma.copy(to_np(dbz_cross))
elec_cross_filled = np.ma.copy(to_np(elec_cross))
# For each cross section column, find the first index with non-missing
# values and copy these to the missing elements below.
for i in range(dbz_cross_filled.shape[-1]):
    column_vals = dbz_cross_filled[:,i]
    # Let's find the lowest index that isn't filled. The nonzero function
    # finds all unmasked values greater than 0. Since 0 is a valid value
    # for dBZ, let's change that threshold to be -200 dBZ instead.
    first_idx = int(np.transpose((column_vals > -200).nonzero())[0])
    dbz_cross_filled[0:first_idx, i] = dbz_cross_filled[first_idx, i]

for i in range(elec_cross_filled.shape[-1]):
    column_vals = elec_cross_filled[:,i]
    # Let's find the lowest index that isn't filled. The nonzero function
    # finds all unmasked values greater than 0. Since 0 is a valid value
    # for dBZ, let's change that threshold to be -200 dBZ instead.
    first_idx = int(np.transpose((column_vals > -200).nonzero())[0])
    elec_cross_filled[0:first_idx, i] = elec_cross_filled[first_idx, i]


# Get the terrain heights along the cross section line
ter_line = interpline(ter, wrfin=wrf_file, start_point=cross_start,
                      end_point=cross_end)

# Get the lat/lon points
lats, lons = latlon_coords(dbz)

# Get the cartopy projection object
cart_proj = get_cartopy(dbz)

# Create the figure
fig = pyplot.figure(figsize=(12,9))

ax_ctt = fig.add_subplot(1,2,1,projection=cart_proj)
ax_dbz = fig.add_subplot(2,2,2)
ax_emag = fig.add_subplot(2,2,4)
dbz_levels = np.arange(5., 75., 5.)

# Set the margins to 0
ax_ctt.margins(x=0,y=0,tight=True)
ax_emag.margins(x=0,y=0,tight=True)
ax_dbz.margins(x=0,y=0,tight=True)

# Create the color table found on NWS pages.
dbz_rgb = np.array([[4,233,231],
                    [1,159,244], [3,0,244],
                    [2,253,2], [1,197,1],
                    [0,142,0], [253,248,2],
                    [229,188,0], [253,149,0],
                    [253,0,0], [212,0,0],
                    [188,0,0],[248,0,253],
                    [152,84,198]], np.float32) / 255.0

dbz_map, dbz_norm = from_levels_and_colors(dbz_levels, dbz_rgb,
                                           extend="max")

# Make the cross section plot for dbz
dbz_levels = np.arange(5,75,5)
emag_levels = np.linspace(0,17000,10000)


xs = np.arange(0, dbz_cross.shape[-1], 1)
ys = to_np(dbz_cross.coords["vertical"])

exs = np.arange(0, elec_cross.shape[-1], 1)
eys = to_np(elec_cross.coords["vertical"])

emag_contours = ax_emag.contourf(exs, eys ,to_np(elec_cross_filled),levels=emag_levels, cmap="hot_r", extend="max")

dbz_contours = ax_dbz.contourf(xs, ys,to_np(dbz_cross_filled),levels=dbz_levels, cmap=dbz_map, norm=dbz_norm, extend="max")

# Add the color bar
#cb_dbz = fig.colorbar(dbz_contours, ax=ax_dbz)
#cb_dbz.ax.tick_params(labelsize=8)
#cb_dbz.set_label("dBZ", fontsize=10)

# Fill in the mountain area
#ht_fill = ax_dbz.fill_between(xs, 0, to_np(ter_line),
#facecolor="saddlebrown")
#ht_fill_emag = ax_emag.fill_between(xs, 0, to_np(ter_line),
#                                facecolor="saddlebrown")


# Download and create the states, land, and oceans using cartopy features
states = cfeature.NaturalEarthFeature(category='cultural', scale='50m',
                                      facecolor='none',
                                      name='admin_1_states_provinces')
land = cfeature.NaturalEarthFeature(category='physical', name='land',
                                    scale='50m',
                                    facecolor=cfeature.COLORS['land'])

lakes = cfeature.NaturalEarthFeature(category='physical',name='lakes',scale='50m',facecolor="none",edgecolor="blue")
ocean = cfeature.NaturalEarthFeature(category='physical', name='ocean',
                                     scale='50m',
                                     facecolor=cfeature.COLORS['water'])
# Make the pressure contours
#contour_levels = [960, 965, 970, 975, 980, 990]
#c1 = ax_ctt.contour(lons, lats, to_np(smooth_slp), levels=contour_levels, colors="white", transform=crs.PlateCarree(), zorder=3, linewidths=1.0)

# Create the filled cloud top temperature contours
contour_levels = [-80.0, -70.0, -60, -50, -40, -30, -20, -10, 0, 10]
ctt_contours = ax_ctt.contourf(to_np(lons), to_np(lats), to_np(ctt), contour_levels, cmap=get_cmap("Greys"), transform=crs.PlateCarree())

ax_ctt.plot([start_point.lon, end_point.lon],
            [start_point.lat, end_point.lat], color="yellow", marker="o",
            transform=crs.PlateCarree())




# Set the x-ticks to use latitude and longitude labels
coord_pairs = to_np(dbz_cross.coords["xy_loc"])
x_ticks = np.arange(coord_pairs.shape[0])
print(x_ticks)
x_labels = [pair.latlon_str() for pair in to_np(coord_pairs)]
print(x_labels[0])
# Set the desired number of x ticks below
num_ticks = 5
thin = int((len(x_ticks) / num_ticks) + .5)
ax_dbz.set_xticks(x_ticks[::thin])
ax_dbz.set_xticklabels(x_labels[::thin], rotation=60, fontsize=10)

# Set the x-axis and  y-axis labels
ax_dbz.set_xlabel("Latitude, Longitude", fontsize=12)
ax_dbz.set_ylabel("Height (m)", fontsize=12)

# Adjust format for date to use in figure
date_format = wrf_date_time.strftime("%Y-%m-%d %H:%M:%S")

# Add a title
ax_dbz.set_title(f"Cross-Section of Reflectivity (dBZ) at {date_format}", fontsize="14")
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{ATTEMPT}/"

pyplot.savefig(savepath+f"flashcrosssection{wrf_date_time.year:04d}{wrf_date_time.month:02d}{wrf_date_time.day:02d}{wrf_date_time.hour:02d}{wrf_date_time.minute:02d}D{domain}T{timeidx}A{ATTEMPT}.png")
pyplot.show()
