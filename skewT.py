import wrf
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units

# User input for file
year = 2022
month = 11
day = 17
hour = "09"
minute = "20"
domain = 2
timeidx = 0
IOP = 2
ATTEMPT ="1"
gridboxlat=99

lat_lon = [43.45, -76.54]

# Open the NetCDF file
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_{IOP}/ATTEMPT_{ATTEMPT}/"
pattern = f"wrfout_d0{domain}_{year}-{month}-{day}_{hour}:{minute}:00"
wrfin = Dataset(path+pattern)

# Convert desired coorindates to WRF gridbox coordinates
x_y = wrf.ll_to_xy(wrfin, lat_lon[0], lat_lon[1])

# Read skewT variables in
p1 = wrf.getvar(wrfin,"pressure",timeidx=timeidx)
T1 = wrf.getvar(wrfin,"tc",timeidx=timeidx)
Td1 = wrf.getvar(wrfin,"td",timeidx=timeidx)
u1 = wrf.getvar(wrfin,"ua",timeidx=timeidx)
v1 = wrf.getvar(wrfin,"va",timeidx=timeidx)

# Get variables for desired coordinates
p = p1[:,x_y[1],x_y[0]] * units.hPa
T = T1[:,x_y[1],x_y[0]] * units.degC
Td = Td1[:,x_y[1],x_y[0]] * units.degC
u = u1[:,x_y[1],x_y[0]] * units('kt')
v = v1[:,x_y[1],x_y[0]] * units('kt')

#Test if the coordinates are correct
lat1 = wrf.getvar(wrfin,"lat",timeidx=timeidx)
lon1 = wrf.getvar(wrfin,"lon",timeidx=timeidx)
lat = lat1[x_y[1],x_y[0]] * units.degree_north
lon = lon1[x_y[1],x_y[0]] * units.degree_east
print(lat)
print(lon) 

# Example of defining your own vertical barb spacing
skew = SkewT()

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')

# Set spacing interval--Every 50 mb from 1000 to 100 mb
my_interval = np.arange(100, 1000, 50) * units('mbar')

# Get indexes of values closest to defined interval
ix = mpcalc.resample_nn_1d(p, my_interval)

# Plot only values nearest to defined interval values
skew.plot_barbs(p[ix], u[ix], v[ix])

# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-60, 40)
skew.ax.set_xlabel('Temperature ($^\circ$C)')
skew.ax.set_ylabel('Pressure (hPa)')

plt.title(f"SkewT at {year}{month}{day}{hour}{minute}")
#plt.savefig(path+f"skewT{year}{month}{day}{hour}{minute}A{ATTEMPT}.png")
plt.show()

