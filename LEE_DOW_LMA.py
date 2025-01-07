#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')


# In[2]:


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import datetime as dt
import glob
from scipy import stats

import pyart
import nexradaws
conn = nexradaws.NexradAwsInterface()

import cartopy.crs as ccrs
from metpy.plots import USCOUNTIES
import cartopy.feature as cfeature

# Import for LMA data, comes with the leelma workshop install
from pyxlma.lmalib.io import read as lma_read
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[3]:


radar_id = 'KTYX'

# This is a dummy array so that you can rerun all the cells later
# with a different radar file or different plotting criteria without
# rereading the LMA data, too
flash_event_time = np.array([dt.datetime(2000,1,1,1,1),
                             dt.datetime(2000,1,1,1,2)])


# ### Set the plotting criteria
# A couple of options:
# 
# file_type options are 'dat' or 'h5'
# - 'dat' are the standard VHF source files ending in '.dat.gz'
# - 'h5' are produced by lmatools and include the same sources grouped into flashes. The 'events' in this file should match the sources in the 'dat' file exactly
# 
# plot_type options are 'flashes_only' or 'all'
# - 'flashes_only' will only plot VHF sources which were grouped into flashes with a set minimum number of events per flash (and any other plotting criteria). The file_type MUST be 'h5'
# - 'all' will plot any VHF sources meeting the set plotting criteria. This can be from either file_type
# 
# If you want to include LMA sources from ONLY before or after 
# the radar scan, adjust the 'start' and/or 'end' times several cells below

# In[4]:


## Set the plotting criteria!
min_stations = 6 # more stations = more confident it's a good solution
max_chi = 1 # lower reduced chi^2 = more confident it's a good solution
tbuffer = 5*60 # How many seconds before and after the radar scan time do you want to plot LMA data?
max_dist = 2.5e3 # meters, max distance of an LMA source to a radar grid centroid to be plotted

# Set max range for DOW RHI
rng_rhi = 35

file_type = 'dat' # 'h5' or 'dat'
plot_type = 'all' # 'flashes_only' or 'all' - flashes_only MUST be h5 file
min_events_per_flash = 10 # Minimum number of sources per flash

# Set your directory with the LMA files
# LMA_dir = '/Users/vanna.chmielewski/Documents/LEE/LEEflashsortNov/'
LMA_dir = '/Users/vanna.chmielewski/Documents/LEE/Nov19/'

if file_type == 'h5': file_suffix = 'flash.h5'
if file_type == 'dat': file_suffix = 'gz'


# In[5]:


# Start with the DOW file of interest
dow_file = '/Users/vanna.chmielewski/Documents/LEE/IOP2_DOW/cfradial_netcdf/low/cfrad.20221119_023246.306_DOW7_v321_s00_az101.30_RHI.nc'
# Read into pyart
dow_pyart = pyart.io.read(dow_file)
# Find x,y,z of gates
rx,ry,rz = dow_pyart.get_gate_x_y_z(0)
# Find the starttime
dow_starttime = dt.datetime.strptime(dow_pyart.time['units'].split(' ')[-1],
                     '%Y-%m-%dT%H:%M:%SZ')+dt.timedelta(seconds = dow_pyart.time['data'][0])

# Pull the azimuth name from the file name (degrees)
az_name = dow_file.split('/')[-1].split('az')[1].split('_')[0]
# And also the mode reported in the file - convert this to form used later
# for calculations (in radians)
az_rad  = np.deg2rad(stats.mode( dow_pyart.azimuth["data"])[0]-90)

print (dow_starttime, " at azimuth= ", az_name)

# Find the start and end times for pulling LMA data
start = dow_starttime - dt.timedelta(seconds=tbuffer)
end   = dow_starttime + dt.timedelta(seconds=tbuffer)


# ### Either download the WSR-88D data for the previous 10-minute period OR provide the path to pre-downloaded data

# In[6]:


# Create datetime objects for potential start/end times of interest:
start88=dow_starttime-dt.timedelta(minutes=10)
end88 = dow_starttime

# Create a folder to store the data
downloadloc = f'{start88:%Y%m%d}_{radar_id}'

#Determine what 88D scans are avaiable for the radar site and times listed
scans = conn.get_avail_scans_in_range(start88, end88, radar_id)

print("There are {} scans available between {} and {}\n".format(len(scans), start88, end88))

#download the data
results = conn.download(scans, downloadloc)
files = glob.glob(f'{downloadloc}/*V06')

#####
# # OR if already downloaded just replace the above with:
#####
# files = glob.glob('path/to/88Ddata/*V06').sort()


# ### Align DOW RHI and WSR-88D data

# In[7]:


# Find all the WSR-88D file times
ktyx_starttimes = [dt.datetime.strptime(ktyx_file.split('/')[-1],"KTYX%Y%m%d_%H%M%S_V06")
                  for ktyx_file in files]

# Find difference in time from those to the DOW scan time
options = np.array([np.abs((dow_starttime - ktyx_starttime).total_seconds()) 
                    for ktyx_starttime in ktyx_starttimes])

# Pull the WSR-88D file with the minimum difference in time
which_file = options.argmin()
ktyx_starttime = ktyx_starttimes[which_file]
ktyx_file      = files[which_file]
# Read into a pyart object
radar = pyart.io.read(ktyx_file)


# In[8]:


# Find 88D-centered x/y of the DOW scan starting point
g1_lon = dow_pyart.get_gate_lat_lon_alt(sweep=0)[1][0,0] # Longitude
g1_lat = dow_pyart.get_gate_lat_lon_alt(sweep=0)[0][0,0] # Latitude
g1_x, g1_y = pyart.core.geographic_to_cartesian_aeqd(g1_lon,g1_lat, 
                                                    radar.longitude['data'][0], 
                                                    radar.latitude['data'][0], 
                                                    R=6370997.0)

# Find 88D-centered x/y of the DOW scan at rng_rhi distance
g2_x =  np.cos(az_rad)*rng_rhi
g2_y = -np.sin(az_rad)*rng_rhi
# And the lat/lon of that point
g2_lon, g2_lat = pyart.core.cartesian_to_geographic_aeqd(g1_x+g2_x*1e3, g1_y+g2_y*1e3,
                                                        radar.longitude['data'][0], 
                                                        radar.latitude['data'][0], 
                                                        R=6370997.0)


# ### Read in LMA data
# 
# Extra check here is so that you can rerun this cell and update the time if you read in a different DOW data or skip it if it's already in memory

# In[9]:


# Don't read in data if it's already been read
if (start >= min(flash_event_time))&(
    end   <= max(flash_event_time)):
    print ('Flash data already in memory')
    pass
else:
    print ('Flash data in memory for incorrect time. Reading new data now.')
    # This makes sure to grab files on either side of the radar scan if the radar
    # scan was near the 10-minute interval according to the buffer
    filenames = glob.glob('{}L*{}000_0600.dat.{}'.format(LMA_dir,
                 start.strftime('%y%m%d_%H%M')[:-1],file_suffix))
    filenames = filenames + glob.glob('{}L*{}000_0600.dat.{}'.format(LMA_dir,
                 end.strftime(  '%y%m%d_%H%M')[:-1],file_suffix))
    filenames = list(set(filenames))
    if file_type == 'h5':
        flashes = pd.DataFrame()
        flash_events = pd.DataFrame()
        for filename in filenames:
            timeobj = dt.datetime.strptime(filename.split('/')[-1], 
                                           "LYLOUT_%y%m%d_%H%M%S_0600.dat.flash.h5")
            # This is the flash table
            flashes2 = pd.read_hdf(filename,'flashes/LMA_{}00_600'.format(
                                            timeobj.strftime('%y%m%d_%H%M')))
            # This is the event (VHF source) table
            flash_events2 = pd.read_hdf(filename,'events/LMA_{}00_600'.format(
                                            timeobj.strftime('%y%m%d_%H%M')))
            # Flash ID's are not unique between files. This writes new ones 
            # in the second file, if it exists
            if flashes.shape[0]>0:
                flashes2.flash_id      = flashes2['flash_id']     +flashes.flash_id.max()+1
                flash_events2.flash_id = flash_events2['flash_id']+flashes.flash_id.max()+1
            else:
                pass
            flashes      = pd.concat([flashes,flashes2])
            flash_events = pd.concat([flash_events,flash_events2])         

        # Make a series of datetime objects for each event
        flash_event_time = np.array([dt.datetime(*start.timetuple()[:3])+
                                     dt.timedelta(seconds = i) for
                            i in flash_events.time])
        
    # If .dat.gz files, read them in and create the same series of datetime objects
    elif file_type == 'dat':
        filenames.sort()
        lma_data, starttime = lma_read.dataset(filenames)
        flash_event_time = pd.Series(lma_data.event_time) # because time comparisons


# ### Put those along the DOW RHIs 

# In[14]:


try:
    if file_type == 'h5':
        # Select all the sources meeting the criteria set above
        selection_event_dow = (flash_event_time>=start)&(
                               flash_event_time <end)&(
                               flash_events.chi2<=max_chi)&(
                               flash_events.stations>=min_stations)
        
        lma_lon = flash_events.lon[selection_event_dow].values
        lma_lat = flash_events.lat[selection_event_dow].values
        lma_z_dow = flash_events.alt[selection_event_dow] - dow_pyart.altitude['data'][0]

    if file_type == 'dat':
        # Select all the good sources according to the criteria above
        selection_event_dow = (flash_event_time>=start)&(
                               flash_event_time< end)&(
                               lma_data.event_chi2.values<=max_chi)&(
                               lma_data.event_stations.values>=min_stations)
        
        lma_lon = lma_data.event_longitude.values[selection_event_dow]
        lma_lat = lma_data.event_latitude.values[selection_event_dow]
        lma_z_dow = lma_data.event_altitude.values[selection_event_dow] - radar.altitude['data'][0]
        
    # Pull x,y,z on a radar-centered grid
    lma_x_dow, lma_y_dow = pyart.core.geographic_to_cartesian_aeqd(
                                                lma_lon, lma_lat,
                                                dow_pyart.longitude['data'][0], 
                                                dow_pyart.latitude['data'][0], R=6370997.0)
        
except AttributeError:
    pass

# Rotate the cartesian coordinates to the RHI angle
try:
    cartesian_lma_r  = lma_x_dow*np.cos(az_rad)  - lma_y_dow*np.sin(az_rad) # Cartesian x
    dists            = lma_y_dow*np.cos(az_rad)  + lma_x_dow*np.sin(az_rad) # Cartesian y
    # Create a new subset of those within distance 'max_dist' from the RHI plane
    sel2_dow = (np.abs(dists)<max_dist)
except NameError:
    pass

# If you want to know how many flashes were identified in this view, this will find them
# Find flashes with at least one VHF source meeting the distance and time threshold
if (len(sel2_dow)>0)&(file_type=='h5'):
    ids_in_rhi     = flash_events.flash_id[selection_event_dow][sel2_dow]
    flashes_in_rhi = flashes.loc[flashes.flash_id.isin(ids_in_rhi.unique())]
    print ('Count of flashes meeting criteria = ',np.sum(flashes_in_rhi.n_points>=min_events_per_flash))


# ## Plot on 88D lowest-level PPI for reference

# In[17]:


display = pyart.graph.RadarMapDisplay(radar)
sweep=0
projection = ccrs.PlateCarree()

fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111,aspect='equal', projection=projection)

cm= display.plot_ppi_map('reflectivity',sweep, vmin=-20, vmax=60, 
                     colorbar_label='Reflectivity (dBZ)', 
                     title='KTYX Reflectivity {}'.format(ktyx_starttime.strftime('%H:%M:%S')),
                     cmap='pyart_HomeyerRainbow', colorbar_flag=True,ax=ax,
                     projection=projection, 
                     lon_lines=None, lat_lines=None, add_grid_lines=False)

# The RHI
ax.arrow(g1_lon, g1_lat, 
         g2_lon[0]-g1_lon,g2_lat[0]-g1_lat,
         color='black', linewidth=1, head_width=0)
ax.scatter(g1_lon, g1_lat, color='k', marker='x')

# If VHF sources exist, plot them
try:
    # Only those in flashes and meeting criteria set above
    if plot_type == 'flashes_only':
        for flash in flashes.flash_id[flashes.n_points>=min_events_per_flash]:
            plt.scatter(lma_lon[(flash_events.flash_id == flash)&selection_event_dow], 
                        lma_lat[(flash_events.flash_id == flash)&selection_event_dow],
                       color='0.5',s=1) 
            if len(flash_events.flash_id[selection_event_dow][sel2_dow] == flash)>0:
                plt.scatter(lma_lon[sel2_dow][flash_events.flash_id[selection_event_dow][sel2_dow] == flash], 
                            lma_lat[sel2_dow][flash_events.flash_id[selection_event_dow][sel2_dow] == flash],
                           color='k',s=1) 
    # all sources meeting criteria
    if plot_type == 'all':
        plt.scatter(lma_lon, lma_lat, color='0.5',s=1)
        plt.scatter(lma_lon[sel2_dow], lma_lat[sel2_dow], color='k',s=1)
        
except NameError:
    pass

# Add gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=1,
                    color="gray",alpha=0.3,linestyle="--",)
gl.right_labels = False
gl.top_labels = False

# Add map features
ax.add_feature(USCOUNTIES, linewidth=0.25)
ax.add_feature(cfeature.LAKES, facecolor='None',edgecolor='darkblue', linewidth=1)

# Set bounds
plt.xlim(-76.5,-75.6)
plt.ylim(43.5,44.3)
plt.tight_layout()
# plt.savefig('KTLX{}_2_counties_plus.png'.format(dow_file.split('/')[-1][6:-3]), 
#             dpi=500, facecolor='white')
plt.show()


# In[18]:


hgt_rhi = 5.5 # Max height shown

fig = plt.figure(figsize = (5,2))
ax1 = fig.add_subplot(111)
# The DOW RHI
plt.contourf((rx**2+ry**2)**0.5/1e3,rz/1e3,
            dow_pyart.fields['DBZHCC']['data'],
            cmap = 'pyart_HomeyerRainbow', levels = np.arange(-20,62,2))
plt.colorbar(label = 'Reflectivity (dBZ)')

if plot_type == 'flashes_only':
    for flash in flashes.flash_id[flashes.n_points>=min_events_per_flash]:
        if len(flash_events.flash_id[selection_event_dow][sel2_dow])>0:
            plt.scatter(cartesian_lma_r[sel2_dow][flash_events.flash_id[selection_event_dow][sel2_dow] == flash]/1e3, 
                              lma_z_dow[sel2_dow][flash_events.flash_id[selection_event_dow][sel2_dow] == flash]/1e3,
                       c='k',s=1) 
if plot_type == 'all':
    if len(cartesian_lma_r[sel2_dow])>0:
        plt.scatter(cartesian_lma_r[sel2_dow]/1e3, 
                          lma_z_dow[sel2_dow]/1e3,c='k',s=1) 

plt.ylabel('Height (km)')        
plt.xlabel('Distance (km)')        
plt.ylim(0, hgt_rhi)
plt.xlim(0, rng_rhi)

plt.tight_layout()
# plt.savefig('DOWaz{}_{}.png'.format(az_name,start.strftime(%y%m%d_%H%M%S)), 
#             dpi=500, facecolor='white')


# ## And a 6-panel version

# In[19]:


fig = plt.figure(figsize = (11,6))
ax1 = fig.add_subplot(321)
plt.contourf((rx**2+ry**2)**0.5/1e3,rz/1e3,
            dow_pyart.fields['DBZHC']['data'],
            cmap = 'pyart_NWSRef',
            levels = np.arange(-20,65,1))
plt.colorbar(label = 'Reflectivity (dBZ)')
plt.ylim(0, hgt_rhi)
plt.xlim(0, rng_rhi)
if plot_type == 'flashes_only':
    for flash in flashes.flash_id[flashes.n_points>=min_events_per_flash]:
        if len(flash_events.flash_id[selection_event_dow][sel2_dow])>0:
            plt.scatter(cartesian_lma_r[sel2_dow][flash_events.flash_id[selection_event_dow][sel2_dow] == flash]/1e3, 
                              lma_z_dow[sel2_dow][flash_events.flash_id[selection_event_dow][sel2_dow] == flash]/1e3,
                       c='k',s=1) 
if plot_type == 'all':
    if len(cartesian_lma_r[sel2_dow])>0:
        plt.scatter(cartesian_lma_r[sel2_dow]/1e3, 
                          lma_z_dow[sel2_dow]/1e3,c='k',s=1) 
        
#Velocity
ax2 = fig.add_subplot(322)
display = pyart.graph.RadarMapDisplay(dow_pyart)
display.plot('VEL', 0, vmin=-30, vmax=30, colorbar_label='(m/s)', cmap = 'pyart_NWSVel', 
             axislabels=('Range (km)','Height (km)'), title=f'Velocity')
display.set_limits((0, rng_rhi), (0, hgt_rhi), ax=ax2)
if plot_type == 'flashes_only':
    for flash in flashes.flash_id[flashes.n_points>=min_events_per_flash]:
        if len(flash_events.flash_id[selection_event_dow][sel2_dow])>0:
            plt.scatter(cartesian_lma_r[sel2_dow][flash_events.flash_id[selection_event_dow][sel2_dow] == flash]/1e3, 
                              lma_z_dow[sel2_dow][flash_events.flash_id[selection_event_dow][sel2_dow] == flash]/1e3,
                       c='k',s=1) 
if plot_type == 'all':
    if len(cartesian_lma_r[sel2_dow])>0:
        plt.scatter(cartesian_lma_r[sel2_dow]/1e3, 
                          lma_z_dow[sel2_dow]/1e3,c='k',s=1) 
    
#Spectrum Width
ax6 = fig.add_subplot(323)
display = pyart.graph.RadarMapDisplay(dow_pyart)
colors1 = plt.cm.binary_r(np.linspace(0.2,0.8,33))
colors2= plt.cm.gnuplot_r(np.linspace(0.,0.7,100))
colors = np.vstack((colors1, colors2[10:121]))
zdrcolors1 = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
display.plot('WIDTH', 0, vmin=0, vmax=14, colorbar_label='(m/s)', cmap = zdrcolors1, 
             axislabels=('Range (km)','Height (km)'), title=f'Spectrum Width')
display.set_limits((0, rng_rhi), (0, hgt_rhi), ax=ax6)
if plot_type == 'flashes_only':
    for flash in flashes.flash_id[flashes.n_points>=min_events_per_flash]:
        if len(flash_events.flash_id[selection_event_dow][sel2_dow])>0:
            plt.scatter(cartesian_lma_r[sel2_dow][flash_events.flash_id[selection_event_dow][sel2_dow] == flash]/1e3, 
                              lma_z_dow[sel2_dow][flash_events.flash_id[selection_event_dow][sel2_dow] == flash]/1e3,
                       c='k',s=1) 
if plot_type == 'all':
    if len(cartesian_lma_r[sel2_dow])>0:
        plt.scatter(cartesian_lma_r[sel2_dow]/1e3, 
                          lma_z_dow[sel2_dow]/1e3,c='k',s=1) 

#Differential Reflectivity
ax3 = fig.add_subplot(324)
display = pyart.graph.RadarMapDisplay(dow_pyart)
colors1 = plt.cm.binary_r(np.linspace(0.,0.8,33))
colors2= plt.cm.gist_ncar(np.linspace(0.,1,100))
colors = np.vstack((colors1, colors2[10:121]))
zdrcolors2 = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
display.plot('ZDRC', 0, vmin=-2, vmax=6, colorbar_label='(dB)', 
             cmap = zdrcolors2, axislabels=('Range (km)','Height (km)'), title=f'Differential Reflectivity')
display.set_limits((0, rng_rhi), (0, hgt_rhi), ax=ax3)
if plot_type == 'flashes_only':
    for flash in flashes.flash_id[flashes.n_points>=min_events_per_flash]:
        if len(flash_events.flash_id[selection_event_dow][sel2_dow])>0:
            plt.scatter(cartesian_lma_r[sel2_dow][flash_events.flash_id[selection_event_dow][sel2_dow] == flash]/1e3, 
                              lma_z_dow[sel2_dow][flash_events.flash_id[selection_event_dow][sel2_dow] == flash]/1e3,
                       c='k',s=1) 
if plot_type == 'all':
    if len(cartesian_lma_r[sel2_dow])>0:
        plt.scatter(cartesian_lma_r[sel2_dow]/1e3, 
                          lma_z_dow[sel2_dow]/1e3,c='k',s=1) 

#Specific Differential Phase
ax5 = fig.add_subplot(325)
display = pyart.graph.RadarMapDisplay(dow_pyart)
colors1 = plt.cm.binary(np.linspace(0.2,0.8,24))
colors2= plt.cm.gnuplot_r(np.linspace(0.,1,100))
colors = np.vstack((colors1, colors2[10:121]))
zdrcolors1 = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
display.plot('KDP', 0, vmin=-1, vmax=4, colorbar_label='(degrees/km)', cmap = zdrcolors1, 
             axislabels=('Range (km)','Height (km)'), title=f'Specific Differential Phase')
display.set_limits((0, rng_rhi), (0, hgt_rhi), ax=ax5)
if plot_type == 'flashes_only':
    for flash in flashes.flash_id[flashes.n_points>=min_events_per_flash]:
        if len(flash_events.flash_id[selection_event_dow][sel2_dow])>0:
            plt.scatter(cartesian_lma_r[sel2_dow][flash_events.flash_id[selection_event_dow][sel2_dow] == flash]/1e3, 
                              lma_z_dow[sel2_dow][flash_events.flash_id[selection_event_dow][sel2_dow] == flash]/1e3,
                       c='k',s=1) 
if plot_type == 'all':
    if len(cartesian_lma_r[sel2_dow])>0:
        plt.scatter(cartesian_lma_r[sel2_dow]/1e3, 
                          lma_z_dow[sel2_dow]/1e3,c='k',s=1) 

#Correlation Coefficient
ax4 = fig.add_subplot(326)
display = pyart.graph.RadarMapDisplay(dow_pyart)
display.plot('RHOHV', 0, vmin=0.7, vmax=1.02, colorbar_label='%', cmap = 'pyart_SCook18',
             axislabels=('Range (km)','Height (km)'), title=f'Correlation Coefficient')
display.set_limits((0, rng_rhi), (0, hgt_rhi), ax=ax4)
if plot_type == 'flashes_only':
    for flash in flashes.flash_id[flashes.n_points>=min_events_per_flash]:
        if len(flash_events.flash_id[selection_event_dow][sel2_dow])>0:
            plt.scatter(cartesian_lma_r[sel2_dow][flash_events.flash_id[selection_event_dow][sel2_dow] == flash]/1e3, 
                              lma_z_dow[sel2_dow][flash_events.flash_id[selection_event_dow][sel2_dow] == flash]/1e3,
                       c='k',s=1) 
if plot_type == 'all':
    if len(cartesian_lma_r[sel2_dow])>0:
        plt.scatter(cartesian_lma_r[sel2_dow]/1e3, 
                          lma_z_dow[sel2_dow]/1e3,c='k',s=1) 
plt.tight_layout()
# plt.savefig('{}.png'.format(dow_file.split('/')[-1][6:-3]), dpi=200, facecolor='white')


# In[ ]:




