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
from matplotlib.colors import BoundaryNorm
from metpy.units import units
import wrffuncs
from datetime import datetime, timedelta


# User input for file
wrf_date_time = datetime(2022,11,18,1,55,00)
domain = 2
numtimeidx = 4
IOP = 2
ATTEMPT = 1
height = 850

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


# Get the WRF variables
z = getvar(ncfile, "z")
mdbz = getvar(ncfile, "mdbz")
p = getvar(ncfile, "pressure")
wspd =  getvar(ncfile, "wspd_wdir", units="kts")[0,:]
ua = getvar(ncfile, "ua", units="kt")
va = getvar(ncfile, "va", units="kt")
ter = getvar(ncfile, "ter", units="m")
tc = getvar(ncfile, "tc")
t2 = getvar(ncfile, "T2")
t2 = to_np(t2) * units.kelvin
t2 = t2.to('degC')
t2 = t2.magnitude

temp_850 = interplevel(tc, p, 850) 
les_temp_diff = t2 - to_np(temp_850)

# Interpolate geopotential height, u, and v winds to 500 hPa
ht_500 = interplevel(z, p, height)
u_500 = interplevel(ua, p, height)
v_500 = interplevel(va, p, height)
wspd_500 = interplevel(wspd, p, height)

# Get the lat/lon points
lats, lons = latlon_coords(ht_500)

# Get the cartopy projection object
cart_proj = get_cartopy(ht_500)

# Create a figure that will have 3 subplots
fig = plt.figure(figsize=(12,9))
ax_ctt = fig.add_subplot(1,2,1,projection=cart_proj)
ax_wspd = fig.add_subplot(2,2,2, projection=cart_proj)
ax_dbz = fig.add_subplot(2,2,4, projection=cart_proj)
# Set the margins to 0
ax_ctt.margins(x=0,y=0,tight=True)
ax_wspd.margins(x=0,y=0,tight=True)
ax_dbz.margins(x=0,y=0,tight=True)


# Download and create the states, land, and oceans using cartopy features
states = cfeature.NaturalEarthFeature(category='cultural', scale='50m',
                                      facecolor='none',
                                      name='admin_1_states_provinces')
land = cfeature.NaturalEarthFeature(category='physical', name='land',
                                    scale='50m',
                                    facecolor=cfeature.COLORS['land'])

lakes = cfeature.NaturalEarthFeature(category='physical',name='lakes',scale='50m',facecolor="none",edgecolor="blue",zorder=5)
ocean = cfeature.NaturalEarthFeature(category='physical', name='ocean',
                                     scale='50m',
                                     facecolor=cfeature.COLORS['water'])
# Make the pressure contours
contour_levels = [15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110, 120]
c1 = ax_ctt.contourf(lons, lats, to_np(wspd_500), levels=contour_levels, cmap=get_cmap("rainbow"), transform=crs.PlateCarree(), zorder=2)

# Create the 500 hPa geopotential height contours
contour_levels = np.arange(1320., 1620., 30.)
ctt_contours = ax_ctt.contour(to_np(lons), to_np(lats), to_np(ht_500), contour_levels, colors="black", transform=crs.PlateCarree(), zorder=3)
ax_ctt.clabel(ctt_contours,inline=1, fontsize=10, fmt="%i")

# Create the color bar for wind speed temperature
cbar = fig.colorbar(c1, ax=ax_ctt, orientation="horizontal", pad=.05)
cbar.set_label('Knots',fontsize=14)


# Add the 500 hPa wind barbs, only plotting every nth data point.
ax_ctt.barbs(to_np(lons[::40,::40]), to_np(lats[::40,::40]),
        to_np(u_500[::40,::40]), to_np(v_500[::40,::40]),
          transform=crs.PlateCarree(), length=6,zorder=4)


# Draw the oceans, land, and states
ax_ctt.add_feature(land)
ax_ctt.add_feature(states, linewidth=.5, edgecolor="black")
ax_ctt.add_feature(lakes)
ax_ctt.add_feature(ocean)

ax_wspd.add_feature(land)
ax_wspd.add_feature(states, linewidth=.5, edgecolor="black")
ax_wspd.add_feature(lakes)
ax_wspd.add_feature(ocean)

ax_dbz.add_feature(land)
ax_dbz.add_feature(states, linewidth=.5, edgecolor="black")
ax_dbz.add_feature(lakes)
ax_dbz.add_feature(ocean)

ax_dbz.set_xlim(cartopy_xlim(mdbz))
ax_dbz.set_ylim(cartopy_ylim(mdbz))

levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

nwscmap = ctables.registry.get_colortable('NWSReflectivity')

mdbz_contour = ax_wspd.contourf(to_np(lons), to_np(lats), to_np(mdbz),levels=levels,cmap=nwscmap, transform=crs.PlateCarree(),zorder=2)
mcbar = fig.colorbar(mdbz_contour, ax=ax_wspd)
mcbar.set_label("dBZ", fontsize=14)

levels = [0,2,4,6,8,10,12,14,16,18,20,22]
isotherm_contour = ax_dbz.contourf(to_np(lons), to_np(lats), les_temp_diff,levels=levels,cmap="jet", transform=crs.PlateCarree(),zorder=2)
lesbar = fig.colorbar(isotherm_contour, ax=ax_dbz)
lesbar.set_label("degC", fontsize=14)
#Set the x-ticks to use latitude and longitude labels
#coord_pairs = to_np(dbz_cross.coords["xy_loc"])
#x_ticks = np.arange(coord_pairs.shape[0])
#x_labels = [pair.latlon_str() for pair in to_np(coord_pairs)]

# Set the desired number of x ticks below
#num_ticks = 5
#thin = int((len(x_ticks) / num_ticks) + .5)
#ax_dbz.set_xticks(x_ticks[::thin])
#ax_dbz.set_xticklabels(x_labels[::thin], rotation=15, fontsize=6)
#ax_wspd.set_xticks(x_ticks[::thin])
#ax_wspd.set_xticklabels(x_labels[::thin], rotation=15, fontsize=6)

#Set the x-axis and  y-axis labels
ax_dbz.set_xlabel("Longitude", fontsize=8)
ax_wspd.set_ylabel("Lattitude", fontsize=8)
ax_dbz.set_ylabel("Lattitude", fontsize=8)

# Adjust format for date to use in figure
date_format = wrf_date_time.strftime("%Y-%m-%d %H:%M:%S")
# Add a shared title at the top with the time label
fig.suptitle(date_format, fontsize=16, fontweight='bold')

# Add a title
ax_ctt.set_title("Simulated 850hPa Wind Speed and Direction (Knots)", fontsize="14")
ax_wspd.set_title("Simulated Composite Reflectivity (dBZ)", fontsize="14")
ax_dbz.set_title("Surface - 850hPa Temperature (degC)", fontsize="14")

savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{ATTEMPT}/"

plt.savefig(savepath+f"4castLES{wrf_date_time.year:04d}{wrf_date_time.month:02d}{wrf_date_time.day:02d}{wrf_date_time.hour:02d}{wrf_date_time.minute:02d}D{domain}T{timeidx}A{ATTEMPT}.png")
plt.show()
