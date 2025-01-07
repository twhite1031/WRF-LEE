import numpy as np
from matplotlib import pyplot
from matplotlib.cm import get_cmap
from matplotlib.colors import from_levels_and_colors
from cartopy import crs
from cartopy.feature import NaturalEarthFeature, COLORS
from netCDF4 import Dataset
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,
                 cartopy_xlim, cartopy_ylim, interpline, CoordPair,xy_to_ll)
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import datetime
import os
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta

# User input for file
wrf_date_time = datetime(2022,11,18,15,10,00)
numtimeidx = 4
domain = 2
IOP = 2
ATTEMPT = 3

# Area you would like the plan view to look at (Left Lon, Right Lon, Bottom Lat, Top Lat)
extent = [-77.965996,-75.00000,43.000,44.273301]

# Adjust datetime to match filenames
def round_to_nearest_5_minutes(dt):
    # Extract the minute value
    minute = dt.minute

    # Calculate the nearest 5-minute mark
    nearest_5 = round(minute / 5) * 5

    # Handle the case where rounding up to 60 minutes
    if nearest_5 == 60:
        dt = dt + timedelta(hours=1)
        nearest_5 = 0

    # Replace the minute value with the nearest 5-minute mark
    rounded_dt = dt.replace(minute=nearest_5, second=0, microsecond=0)

    return rounded_dt
wrf_date_time = round_to_nearest_5_minutes(wrf_date_time)
print("Closest WRF time to match input time: ", wrf_date_time)
timeidx = int((wrf_date_time.minute % 20) // (20 // numtimeidx))
wrf_filename = wrf_date_time
wrf_min = (round((wrf_filename.minute)// 20) * 20)
wrf_filename = wrf_filename.replace(minute=wrf_min)
print("Using data from this file: ", wrf_filename)

# Open the NetCDF file
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_{IOP}/ATTEMPT_{ATTEMPT}/"
pattern = f"wrfout_d0{domain}_{wrf_filename.year:04d}-{wrf_filename.month:02d}-{wrf_filename.day:02d}_{wrf_filename.hour:02d}:{wrf_filename.minute:02d}:00"
wrf_file = Dataset(path+pattern)

# Get the WRF variables
ht = getvar(wrf_file, "z", timeidx=timeidx)
ter = getvar(wrf_file, "ter", timeidx=timeidx)
dbz = getvar(wrf_file, "dbz", timeidx=timeidx)
mdbz = getvar(wrf_file, "mdbz", timeidx=timeidx)
elecmag = getvar(wrf_file, "ELECMAG", timeidx=timeidx)
max_dbz = getvar(wrf_file, "mdbz", timeidx=timeidx)
flash = getvar(wrf_file, "FLSHI", timeidx=timeidx)
ht_agl = getvar(wrf_file, "height_agl", timeidx=timeidx)
ph = getvar(wrf_file, "PH", timeidx=timeidx)
phb = getvar(wrf_file,"PHB",timeidx=timeidx)

flashindexs = np.where(to_np(flash) == 1.0)
# Extract the first value from each array
print("Flash Indexes (can be multiple): ", flashindexs)
flashindex = [arr[0] for arr in flashindexs]
print("Single Flash gridbox index: ", flashindex)
flashloc = xy_to_ll(wrf_file, flashindex[2], flashindex[1], timeidx=timeidx)

# Define the cross section start and end points
cross_start = CoordPair(lat=flashloc[0], lon=flashloc[1]-.5)
cross_end = CoordPair(lat=flashloc[0], lon=flashloc[1]+.5)

print("Flash lat and lon: ",to_np(flashloc))
print("Height agl at 0 index: ", to_np(ht_agl[0,flashindex[1],flashindex[2]]))
print("Height (z) at 0 index: ", to_np(ht[0,flashindex[1],flashindex[2]]))
flashheight = ((phb[flashindex[0], flashindex[1],flashindex[2]] + phb[flashindex[0], flashindex[1],flashindex[2]]) /9.8) - ht[flashindex[0], flashindex[1],flashindex[2]]
print("Flashheight (m): ", to_np(flashheight))
Z = 10**(dbz/10.) # Use linear Z for interpolation

# Compute the vertical cross-section interpolation.  Also, include the lat/lon points along the cross-section in the metadata by setting latlon to True.
z_cross = vertcross(Z, ht, wrfin=wrf_file, start_point=cross_start, end_point=cross_end, latlon=True, meta=True)
elec_cross = vertcross(elecmag, ht, wrfin=wrf_file, start_point=cross_start, end_point=cross_end, latlon=True, meta=True)
#flash_cross = vertcross(flash, ht, wrfin=wrf_file, meta=True)


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
fig = pyplot.figure(figsize=(16,8))
#gs = GridSpec(1,2,width_ratios=[2, 1], figure=fig)

ax_plan = fig.add_subplot(1,2,1, projection=cart_proj)
ax_cross = fig.add_subplot(1,2,2)
dbz_levels = np.arange(5., 75., 5.)

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
# Special stuff for counties
reader = shpreader.Reader('countyl010g.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, crs.PlateCarree())

# Add County/State borders
ax_plan.add_feature(COUNTIES, facecolor='none', edgecolor='black',linewidth=1)

# Make the cross section plot for dbz
dbz_levels = np.arange(5,75,5)
emag_levels = np.arange(1,180002,10000)
flash_levels = np.arange(0,2)

# Deal with the plan view map
mdbz_contours = ax_plan.contour(to_np(lons), to_np(lats), to_np(mdbz), levels=dbz_levels, cmap=dbz_map, norm=dbz_norm, transform=crs.PlateCarree())
mdbz_contoursf = ax_plan.contourf(to_np(lons), to_np(lats), to_np(mdbz), levels=dbz_levels, cmap=dbz_map, norm=dbz_norm, transform=crs.PlateCarree())

ax_plan.plot([cross_start.lon, cross_end.lon],
            [cross_start.lat, cross_end.lat], color="brown", marker="o",
            transform=crs.PlateCarree())
emag_contours = ax_plan.contourf(to_np(lons), to_np(lats), to_np(elecmag)[0,:,:], levels=emag_levels, cmap="hot_r", transform=crs.PlateCarree())

# Put the simulated flash location on the map
ax_plan.scatter(flashloc[1],flashloc[0],s=100,marker="X",c='blue',transform=crs.PlateCarree(),zorder=5)

xs = np.arange(0, dbz_cross.shape[-1], 1)
ys = to_np(dbz_cross.coords["vertical"])

exs = np.arange(0, elec_cross.shape[-1], 1)
eys = to_np(elec_cross.coords["vertical"])

print("XS: ", xs)
print("YS: ", ys)
#fxs = np.arange(0, flash_cross.shape[-1], 1)
#fys = to_np(flash_cross.coords["vertical"])

# Flash location on cross section
#flashcrossloc = np.where(flash_cross == 1)
#print("Flash location on cross section: ", flashcrossloc)

emag_cross_contours = ax_cross.contourf(exs, eys,to_np(elec_cross_filled),levels=emag_levels, cmap="hot_r", extend="max")

dbz_cross_contours = ax_cross.contour(xs, ys ,to_np(dbz_cross_filled),levels=dbz_levels, cmap=dbz_map, norm=dbz_norm, extend="max")

coord_pairs = to_np(dbz_cross.coords["xy_loc"])
print("coord_pairs:", coord_pairs)
indices = [i for i, coord in enumerate(coord_pairs) if coord.x == flashindex[2]]
print("Target x index that matches lat/lon: ", indices)
ax_cross.scatter(indices, flashheight, s=100, marker="x", c='blue',zorder=5)

#flash_cross_contours = ax_cross.contourf(fxs[0:41],fys[0:41], to_np(flash_cross)[0:41], levels=flash_levels, cmap="Blues", extend="max")

# Add the color bar
cb_plan = fig.colorbar(mdbz_contoursf, ax=ax_plan, orientation="vertical")
cb_plan.ax.tick_params(labelsize=8)
cb_plan.set_label("dBZ", fontsize=10)

# Add the color bar
cb_cross = fig.colorbar(emag_contours, ax=ax_cross, orientation='horizontal')
cb_cross.ax.tick_params(labelsize=8)
tick_markers = [10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000,120000,130000,140000,150000,160000,170000,180000]
cb_cross.set_ticks(tick_markers[::2])
selected_ticks = tick_markers[::2]
cb_cross.set_ticklabels([f'{int(tick) // 1000}' for tick in selected_ticks])
cb_cross.set_label("kV/m", fontsize=10)

# Fill in the mountain area
ht_fill_emag = ax_cross.fill_between(xs, 0, to_np(ter_line),
                                facecolor="saddlebrown")

# Set the x-ticks to use latitude and longitude labels
x_ticks = np.arange(coord_pairs.shape[0])
print(x_ticks)
x_labels = [pair.latlon_str() for pair in to_np(coord_pairs)]
print(x_labels[0])

# Add the gridlines
gl_a1 = ax_plan.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
gl_a1.xlabel_style = {'rotation': 'horizontal','size': 10,'ha':'center'} # Change 14 to your desired font size
gl_a1.ylabel_style = {'size': 10}  # Change 14 to your desired font size
gl_a1.xlines = True
gl_a1.ylines = True
gl_a1.top_labels = False  # Disable top labels
gl_a1.right_labels = False  # Disable right labels
gl_a1.xpadding = 20
print("Made gridlines")


# Set the desired number of x ticks below
num_ticks = 5
thin = int((len(x_ticks) / num_ticks) + .5)
#ax_cross.set_xticks(x_ticks[::thin])
#ax_cross.set_xticklabels(x_labels[::thin],rotation=60,fontsize=8)

# Set the view of the plot
ax_plan.set_extent([-77.965996,-75.00000,43.000,44.273301],crs=crs.PlateCarree())
#ax_plan.set_extent([-82.4134,-77.8456,41.291,43.5],crs=crs.PlateCarree())

# Set the x-axis and  y-axis labels
ax_plan.set_xlabel("Longitude", fontsize=10)
ax_plan.set_ylabel("Latitude", fontsize=10)
ax_cross.set_xlabel("Latitude, Longitude", fontsize=10)
ax_cross.set_ylabel("Height (m)", fontsize=10)

# Remove the filled contours now the we have made the colorbar
mdbz_contoursf.remove()

date_format = wrf_date_time.strftime("%Y-%m-%d %H:%M:%S")
#fig.tight_layout()
# Add a title
ax_plan.set_title(f"Plan view of Composite Reflectivity (dBZ) and Electric Field Magnitude (V/m) at " + date_format, fontsize=10, fontweight='bold')
ax_cross.set_title(f"Cross-Section of Composite Reflectivity (dBZ) and Electric Field Magnitude (V/m) at " + date_format, fontsize=10, fontweight='bold')
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{ATTEMPT}/"
strflashheight = int(flashheight)
print("Int version of flash height: ", strflashheight)
pyplot.savefig(savepath+f"3DFLASHI{wrf_filename.year:04d}{wrf_filename.month:02d}{wrf_filename.day:02d}{wrf_filename.hour:02d}{wrf_filename.minute:02d}D{domain}T{timeidx}A{ATTEMPT}H{strflashheight}.png")
pyplot.show()
