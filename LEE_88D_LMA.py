#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('pylab', 'inline')


# In[2]:


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import pandas as pd
import numpy as np
import datetime as dt
import glob
from scipy import spatial

import pyart
import nexradaws
conn = nexradaws.NexradAwsInterface()
import cartopy.crs as ccrs

# Import for LMA data, comes with the leelma workshop install
from pyxlma.lmalib.io import read as lma_read
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[3]:


# Set the WSR-88D radar name
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
# If you want to include LMA sources from before the radar scan, adjust the selection_event several cells below here and potentially the LMA ingest cell, too

# In[13]:


## Set the plotting criteria!
min_stations = 6 # more stations = more confident it's a good solution
max_chi = 1 # lower reduced chi^2 = more confident it's a good solution
tbuffer = 10*60 # How many seconds AFTER the radar scan time do you want to plot LMA data?
max_dist = 2.5e3 # meters, max distance of an LMA source to a radar cross-section plane to be plotted

file_type = 'dat' # 'h5' or 'dat'
plot_type = 'all' # 'flashes_only' or 'all' - flashes_only MUST be h5 file
min_events_per_flash = 10 # Minimum number of sources per flash

# Set your directory with the LMA files
LMA_dir = '/data2/white/DATA/PROJ_LEE/LMADATA/'
# LMA_dir = '/Users/vanna.chmielewski/Documents/LEE/Nov19/'

if file_type == 'h5': file_suffix = 'flash.h5'
if file_type == 'dat': file_suffix = 'gz'


# ### Either download the WSR-88D data for a time period OR provide the path to pre-downloaded data

# In[7]:


#Create datetime objects for the start & end times to download:
start=dt.datetime(2022,11,19,0,5,0)
end = start + dt.timedelta(minutes=10)

# Create a folder to store the data
downloadloc = f'/data2/white/DATA/PROJ_LEE/IOP_2/NEXRADLVL2/DOWNLOADS/'

# Determine What scans are avaiable for the radar site and times listed
scans = conn.get_avail_scans_in_range(start, end, radar_id)

print("There are {} scans available between {} and {}\n".format(len(scans), start, end))
print(scans[0:4])

#download the data
results = conn.download(scans, downloadloc)

files = sorted(glob.glob(f'{downloadloc}KTYX*V06'))

#####
# # OR if already downloaded just replace the above with:
#####
# files = sorted(glob.glob('path/to/88Ddata/*V06'))


# In[8]:


files


# In[25]:


# Parse through the times to find all file times
# Could use this to set up a loop if you want to look at a specific period
ktyx_times = [dt.datetime.strptime(ktyx_file.split('/')[-1],"KTYX%Y%m%d_%H%M%S_V06") 
              for ktyx_file in files]

# Select a file of interest
print("files: ", files)
which_file = 0

# Read it into a pyart object and find start and end times to search the LMA data
radar = pyart.io.read(files[which_file])
start = ktyx_times[which_file]
end   = ktyx_times[which_file]+dt.timedelta(seconds=tbuffer)
print("New start: ", start)
print("New end: ", end)
# Get the x,y,z locations of the radar gates for later
rx,ry,rz = radar.get_gate_x_y_z(0)


# In[26]:


print (start)


# In[27]:


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
    print(filenames)
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


# In[28]:


if file_type == 'h5':
# Select all the good sources according to the criteria above
    selection_event = (flash_event_time>=start)&(
                       flash_event_time< start+dt.timedelta(seconds=tbuffer))&(
                       flash_events.chi2<=max_chi)&(
                       flash_events.stations>=min_stations)

    # Pull x,y,z on a radar-centered grid
    lma_z_ktyx = flash_events.alt[selection_event] - radar.altitude['data'][0]
    lma_x_ktyx, lma_y_ktyx = pyart.core.geographic_to_cartesian_aeqd(
                                            flash_events.lon.values[selection_event], 
                                            flash_events.lat.values[selection_event],
                                            radar.longitude['data'][0], 
                                            radar.latitude['data'][0], R=6370997.0)

if file_type == 'dat':
    # Select all the good sources according to the criteria above
    selection_event = (flash_event_time>=start)&(
                       flash_event_time< start+dt.timedelta(seconds=tbuffer))&(
                       lma_data.event_chi2.values<=max_chi)&(
                       lma_data.event_stations.values>=min_stations)

    # Pull x,y,z on a radar-centered grid
    lma_z_ktyx = lma_data.event_altitude.values[selection_event] - radar.altitude['data'][0]
    lma_x_ktyx, lma_y_ktyx = pyart.core.geographic_to_cartesian_aeqd(
                                            lma_data.event_longitude.values[selection_event], 
                                            lma_data.event_latitude.values[selection_event],
                                            radar.longitude['data'][0], 
                                            radar.latitude['data'][0], R=6370997.0)


# ### Quick plot check
# Note: use these coordinates for drawing an arbitrary cross section through the radar scan

# In[29]:


sweep=0 # sweep 0 = lowest level
display = pyart.graph.RadarDisplay(radar)
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111,aspect='equal')
display.plot('reflectivity',sweep, vmin=10, vmax=64, colorbar_label='(dBZ)', 
             axislabels=('East West distance from radar (km)',
                         'North South distance from radar (km)'), 
             cmap='pyart_NWSRef',colorbar_flag=True)

# # Make sure it looks reasonable from top-down view
try:
    plt.scatter((lma_x_ktyx)/1e3,
                (lma_y_ktyx)/1e3, color='k',s=1) # all sources meeting criteria
except:
    pass

plt.xlim(-75,75)
plt.ylim(-75,75)


# # To look at a specific RHI

# In[30]:


az_sel = 10 # Look at this RHI, in degrees, meteorological reference

# Rotate the LMA reference frame
az_rad_ktyx = np.deg2rad(az_sel-90)
cartesian_lma_r_ktyx = lma_x_ktyx*np.cos(az_rad_ktyx) - lma_y_ktyx*np.sin(az_rad_ktyx) # Cartesian x
dists_ktyx           = lma_y_ktyx*np.cos(az_rad_ktyx) + lma_x_ktyx*np.sin(az_rad_ktyx) # Cartesian y

# Create a new subset for plotting any VHF sources within distance 'max_dist'
sel2_ktyx = np.abs(dists_ktyx)<max_dist

if file_type == 'h5':
    # Count flashes with at least one VHF source meeting the distance and time threshold
    ids_in_rhi_ktyx     = flash_events.flash_id[selection_event][sel2_ktyx]
    flashes_in_rhi_ktyx = flashes.loc[flashes.flash_id.isin(ids_in_rhi_ktyx.unique())]
    print ('Count of flashes = ',np.sum(flashes_in_rhi_ktyx.n_points>=min_events_per_flash))


# ### Plots
# 
# In the top-down view, any sources which will be drawn on the RHI will be plotted in blue

# In[31]:


sweep=0
display = pyart.graph.RadarDisplay(radar)
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111,aspect='equal')
display.plot('reflectivity',sweep, colorbar_label='(dBZ)', 
             axislabels=('East West distance from radar (km)',
                         'North South distance from radar (km)'), 
              vmin=-32, vmax=64.0,  cmap='pyart_HomeyerRainbow',
             colorbar_flag=True)
plt.arrow(0,0, 50*np.sin(np.deg2rad(az_sel)),
               50*np.cos(np.deg2rad(az_sel)), 
          color='black', linewidth=2, head_width=2)

try:
    # # Make sure it looks reasonable from top-down view
    if plot_type == 'flashes_only':
        for flash in flashes.flash_id[flashes.n_points>=min_events_per_flash]:
                plt.scatter((lma_x_ktyx[(flash_events.flash_id[selection_event] == flash)])/1e3,
                            (lma_y_ktyx[(flash_events.flash_id[selection_event] == flash)])/1e3, color='k',s=1) # all sources meeting criteria
                if np.sum(sel2_ktyx)>0:
                    plt.scatter(lma_x_ktyx[sel2_ktyx][flash_events[selection_event].flash_id[sel2_ktyx] == flash]/1e3, 
                                lma_y_ktyx[sel2_ktyx][flash_events[selection_event].flash_id[sel2_ktyx] == flash]/1e3,
                               color='b',s=1)  
    if plot_type == 'all':
        plt.scatter((lma_x_ktyx)/1e3,
                    (lma_y_ktyx)/1e3, color='k',s=1) # all sources meeting criteria
        if np.sum(sel2_ktyx)>0:
            plt.scatter((lma_x_ktyx)[sel2_ktyx]/1e3,
                        (lma_y_ktyx)[sel2_ktyx]/1e3, color='b',s=1) # within range of radar scan
except:
    pass

plt.xlim(-55,20)
plt.ylim(-20,55)
# plt.savefig('KTYX_{}_RHI{}_topdown.png'.format(
#             radar.time['units'].split(' ')[-1], az_sel),
#            dpi=200, facecolor='white')


# In[32]:


# Create a cross section
xsect = pyart.util.cross_section_ppi(radar, [az_sel])

# Set the colorbar label
colorbar_label = "Equivalent \n reflectivity factor \n (dBZ)"

display = pyart.graph.RadarDisplay(xsect)
fig = plt.figure()
ax1 = fig.add_subplot(111)
display.plot("reflectivity", 0, vmin=-32, vmax=64.0, colorbar_label=colorbar_label)
plt.ylim(0, 5)
plt.xlim(0,50)

if np.sum(sel2_ktyx)>0:
    if plot_type == 'flashes_only':
        for flash in flashes.flash_id[flashes.n_points>=min_events_per_flash]:
            plt.scatter(cartesian_lma_r_ktyx[sel2_ktyx][flash_events[selection_event].flash_id[sel2_ktyx] == flash]/1e3, 
                                  lma_z_ktyx[sel2_ktyx][flash_events[selection_event].flash_id[sel2_ktyx] == flash]/1e3,
                           color='k',s=1)
    if plot_type == 'all':
        plt.scatter(cartesian_lma_r_ktyx[sel2_ktyx]/1e3, 
                              lma_z_ktyx[sel2_ktyx]/1e3,
                           color='k',s=1)
    
plt.tight_layout()
# plt.savefig('KTYX_{}_RHI{}.png'.format(
#             radar.time['units'].split(' ')[-1], az_sel),
#            dpi=200, facecolor='white')
#plt.show()


# ## To plot an arbitrary cross-section using a nearest neighbor approach

# In[33]:


# This creates the look up table
# Only run once!
des_tree = spatial.KDTree(np.array([radar.gate_x['data'].ravel(),
                                    radar.gate_y['data'].ravel(),
                                    radar.gate_z['data'].ravel()]).T)


# In[34]:


mx,xx = 15,12 # left and right end points (x, km) KTYX relative (select from map above)
my,xy = 15,25 # left and right end point (y, km)
spacing = 0.1 # grid spacing (km)

# This will automatically figure out the cross section grid (m) through the points above
nn = int(((xx-mx)**2+(xy-my)**2)**0.5/spacing) 
des_x = np.linspace(mx,xx,nn)*1e3
des_y = np.linspace(my,xy,nn)*1e3
des_z = np.arange(0,6,0.1)*1e3

desx_grid,desz_grid = np.meshgrid(des_x,des_z)
desy_grid,desz_grid = np.meshgrid(des_y,des_z)
new_x = np.arange(0,np.shape(des_x)[0]*spacing ,spacing)*1e3

# Find the radar gate closest to the cross-section grid
dists, indext = des_tree.query(np.array([desx_grid.ravel(), 
                                         desy_grid.ravel(), 
                                         desz_grid.ravel()]).T)


# Look for LMA sources near the radar grid points
new_angle = np.arctan2(xy-my,xx-mx) # Angle of cross section
# Recenter on the left point
new_lma_x = lma_x_ktyx-mx*1e3 
new_lma_y = lma_y_ktyx-my*1e3
# Rotate the coordinate frame
new_lma_r = new_lma_x*np.cos(-new_angle) - new_lma_y*np.sin(-new_angle) # Cartesian x
new_dists = new_lma_y*np.cos(-new_angle) + new_lma_x*np.sin(-new_angle) # Cartesian y


# In[35]:


sweep=0
display = pyart.graph.RadarDisplay(radar)
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111,aspect='equal')
display.plot('reflectivity',sweep, vmin=-32, vmax=64, colorbar_label='(dBZ)', 
             axislabels=('East West distance from radar (km)',
                         'North South distance from radar (km)'), 
             cmap='pyart_HomeyerRainbow', colorbar_flag=True)
plt.arrow(np.min(des_x)/1e3, np.min(des_y)/1e3, 
        -(np.min(des_x))/1e3+np.max(des_x)/1e3,
        -(np.min(des_y))/1e3+np.max(des_y)/1e3, color='black', linewidth=2, head_width=2)

if np.sum((new_dists<max_dist)&(new_dists>-max_dist))>0:
    if plot_type == 'flashes_only':
        for flash in flashes.flash_id[flashes.n_points>=min_events_per_flash]:
                plt.scatter(lma_x_ktyx[flash_events[selection_event].flash_id == flash]/1e3, 
                            lma_y_ktyx[flash_events[selection_event].flash_id == flash]/1e3,
                           color='k',s=1) 
                plt.scatter(lma_x_ktyx[(flash_events[selection_event].flash_id == flash)&(new_dists<max_dist)&(new_dists>-max_dist)]/1e3, 
                            lma_y_ktyx[(flash_events[selection_event].flash_id == flash)&(new_dists<max_dist)&(new_dists>-max_dist)]/1e3,
                           color='b',s=1) 
    if plot_type == 'all':
        plt.scatter((lma_x_ktyx)/1e3,
                    (lma_y_ktyx)/1e3, color='k',s=1) # all sources meeting criteria
        plt.scatter(lma_x_ktyx[(new_dists<max_dist)&(new_dists>-max_dist)]/1e3, 
                    lma_y_ktyx[(new_dists<max_dist)&(new_dists>-max_dist)]/1e3,c='b',s=2)

radar_id = radar.metadata['instrument_name']
plt.tight_layout()

plt.xlim(-55,20)
plt.ylim(-20,55)
# plt.savefig('KTYX_{}plan_mx{}_xx{}_my{}_xy{}.png'.format(
#             radar.time['units'].split(' ')[-1], mx, xx, my, xy),
#            dpi=200, facecolor='white')


# In[38]:


fig = plt.figure(figsize=(8,3))
plt.contourf(new_x/1e3, desz_grid[:,0]/1e3,
             radar.fields['reflectivity']['data'].ravel()[indext].reshape(np.shape(desz_grid)),
             levels = np.arange(-32,65,1),cmap='pyart_HomeyerRainbow', )
plt.colorbar(label='Reflectivity (dBZ)')

if np.sum((new_dists<max_dist)&(new_dists>-max_dist))>0:
    if plot_type == 'flashes_only':
        for flash in flashes.flash_id[flashes.n_points>=min_events_per_flash]:
                plt.scatter( new_lma_r[(flash_events[selection_event].flash_id == flash)&(new_dists<max_dist)&(new_dists>-max_dist)]/1e3, 
                            lma_z_ktyx[(flash_events[selection_event].flash_id == flash)&(new_dists<max_dist)&(new_dists>-max_dist)]/1e3,
                           color='k',s=1) 
    if plot_type == 'all':
        plt.scatter(new_lma_r[(new_dists<max_dist)&(new_dists>-max_dist)]/1e3, 
                   lma_z_ktyx[(new_dists<max_dist)&(new_dists>-max_dist)]/1e3,
                           color='k',s=1)

plt.xlim(0, np.max(new_x)/1e3)
plt.ylim(0,6)
plt.ylabel('Altitude (km AGL)')
plt.xlabel('Distance (km)')
plt.tight_layout()
# plt.savefig('KTYX_{}cs_mx{}_xx{}_my{}_xy{}.png'.format(
#             radar.time['units'].split(' ')[-1], mx, xx, my, xy),
#            dpi=200, facecolor='white')


# ### Same thing can be applied to all other radar moments
# 
# Examples below
#ELAT
# In[ ]:





# In[40]:


fig = plt.figure(figsize=(8,3))
colors1 = plt.cm.binary_r(np.linspace(0.,0.8,33))
colors2= plt.cm.gist_ncar(np.linspace(0.,1,100))
colors = np.vstack((colors1, colors2[10:121]))
zdrcolors2 = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
plt.contourf(new_x/1e3, desz_grid[:,0]/1e3,
             radar.fields['differential_reflectivity']['data'].ravel()[indext].reshape(np.shape(desz_grid)),
             levels = np.arange(-2,6,0.1),
             cmap=zdrcolors2, )
plt.colorbar(label='ZDR (DB)')
if plot_type == 'flashes_only':
    for flash in flashes.flash_id[flashes.n_points>=min_events_per_flash]:
            plt.scatter( new_lma_r[(flash_events[selection_event].flash_id == flash)&(new_dists<max_dist)&(new_dists>-max_dist)]/1e3, 
                        lma_z_ktyx[(flash_events[selection_event].flash_id == flash)&(new_dists<max_dist)&(new_dists>-max_dist)]/1e3,
                       color='k',s=1) 
if plot_type == 'all':
    plt.scatter(new_lma_r[(new_dists<max_dist)&(new_dists>-max_dist)]/1e3, 
               lma_z_ktyx[(new_dists<max_dist)&(new_dists>-max_dist)]/1e3,
                       color='k',s=1)
plt.xlim(0, np.max(new_x)/1e3)
plt.ylim(0,6)
plt.ylabel('Altitude (km AGL)')
plt.xlabel('Distance (km)')
plt.tight_layout()
# plt.savefig('KTYX_zdr_{}cs_mx{}_xx{}_my{}_xy{}.png'.format(
#             radar.time['units'].split(' ')[-1], mx, xx, my, xy),
#            dpi=200, facecolor='white')


# In[41]:


fig = plt.figure(figsize=(8,3))
plt.contourf(new_x/1e3, desz_grid[:,0]/1e3,
             radar.fields['velocity']['data'].ravel()[indext].reshape(np.shape(desz_grid)),
             levels = np.arange(-40,40,1),
             cmap='pyart_NWSVel', )
plt.colorbar(label='Velocity (m/s)')
if plot_type == 'flashes_only':
    for flash in flashes.flash_id[flashes.n_points>=min_events_per_flash]:
            plt.scatter( new_lma_r[(flash_events[selection_event].flash_id == flash)&(new_dists<max_dist)&(new_dists>-max_dist)]/1e3, 
                        lma_z_ktyx[(flash_events[selection_event].flash_id == flash)&(new_dists<max_dist)&(new_dists>-max_dist)]/1e3,
                       color='k',s=1) 
if plot_type == 'all':
    plt.scatter(new_lma_r[(new_dists<max_dist)&(new_dists>-max_dist)]/1e3, 
               lma_z_ktyx[(new_dists<max_dist)&(new_dists>-max_dist)]/1e3,
                       color='k',s=1)
plt.xlim(0, np.max(new_x)/1e3)
plt.ylim(0,6)
plt.ylabel('Altitude (km AGL)')
plt.xlabel('Distance (km)')
plt.tight_layout()
# plt.savefig('KTYX_vel_{}cs_mx{}_xx{}_my{}_xy{}.png'.format(
#             radar.time['units'].split(' ')[-1], mx, xx, my, xy),
#            dpi=200, facecolor='white')


# In[42]:


fig = plt.figure(figsize=(8,3))
colors1 = plt.cm.binary_r(np.linspace(0.2,0.8,33))
colors2= plt.cm.gnuplot_r(np.linspace(0.,0.7,100))
colors = np.vstack((colors1, colors2[10:121]))
zdrcolors1 = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
plt.contourf(new_x/1e3, desz_grid[:,0]/1e3,
             radar.fields['spectrum_width']['data'].ravel()[indext].reshape(np.shape(desz_grid)),
             levels = np.arange(0,14.1,0.1),
             cmap=zdrcolors1, )
plt.colorbar(label='Spectrum Width (m/s)')
if plot_type == 'flashes_only':
    for flash in flashes.flash_id[flashes.n_points>=min_events_per_flash]:
            plt.scatter( new_lma_r[(flash_events[selection_event].flash_id == flash)&(new_dists<max_dist)&(new_dists>-max_dist)]/1e3, 
                        lma_z_ktyx[(flash_events[selection_event].flash_id == flash)&(new_dists<max_dist)&(new_dists>-max_dist)]/1e3,
                       color='k',s=1) 
if plot_type == 'all':
    plt.scatter(new_lma_r[(new_dists<max_dist)&(new_dists>-max_dist)]/1e3, 
               lma_z_ktyx[(new_dists<max_dist)&(new_dists>-max_dist)]/1e3,
                       color='k',s=1)
plt.xlim(0, np.max(new_x)/1e3)
plt.ylim(0,6)
plt.ylabel('Altitude (km AGL)')
plt.xlabel('Distance (km)')
plt.tight_layout()
# plt.savefig('KTYX_spw_{}cs_mx{}_xx{}_my{}_xy{}.png'.format(
#             radar.time['units'].split(' ')[-1], mx, xx, my, xy),
#            dpi=200, facecolor='white')


# In[43]:


fig = plt.figure(figsize=(8,3))
plt.contourf(new_x/1e3, desz_grid[:,0]/1e3,
             radar.fields['cross_correlation_ratio']['data'].ravel()[indext].reshape(np.shape(desz_grid)),
             levels = np.arange(0.7,1.03,0.01),
             cmap='pyart_SCook18', )
plt.colorbar(label='Correlation Coefficient')
if plot_type == 'flashes_only':
    for flash in flashes.flash_id[flashes.n_points>=min_events_per_flash]:
            plt.scatter( new_lma_r[(flash_events[selection_event].flash_id == flash)&(new_dists<max_dist)&(new_dists>-max_dist)]/1e3, 
                        lma_z_ktyx[(flash_events[selection_event].flash_id == flash)&(new_dists<max_dist)&(new_dists>-max_dist)]/1e3,
                       color='k',s=1) 
if plot_type == 'all':
    plt.scatter(new_lma_r[(new_dists<max_dist)&(new_dists>-max_dist)]/1e3, 
               lma_z_ktyx[(new_dists<max_dist)&(new_dists>-max_dist)]/1e3,
                       color='k',s=1)
plt.xlim(0, np.max(new_x)/1e3)
plt.ylim(0,6)
plt.ylabel('Altitude (km AGL)')
plt.xlabel('Distance (km)')
plt.tight_layout()
# plt.savefig('KTYX_cc_{}cs_mx{}_xx{}_my{}_xy{}.png'.format(
#             radar.time['units'].split(' ')[-1], mx, xx, my, xy),
#            dpi=200, facecolor='white')


# In[44]:


fig = plt.figure(figsize=(8,3))
plt.contourf(new_x/1e3, desz_grid[:,0]/1e3,
             radar.fields['differential_phase']['data'].ravel()[indext].reshape(np.shape(desz_grid)),
             levels = np.arange(0,360,1),
             cmap='pyart_SCook18', )
plt.colorbar(label='Differential Phase (degrees)')
if plot_type == 'flashes_only':
    for flash in flashes.flash_id[flashes.n_points>=min_events_per_flash]:
            plt.scatter( new_lma_r[(flash_events[selection_event].flash_id == flash)&(new_dists<max_dist)&(new_dists>-max_dist)]/1e3, 
                        lma_z_ktyx[(flash_events[selection_event].flash_id == flash)&(new_dists<max_dist)&(new_dists>-max_dist)]/1e3,
                       color='k',s=1) 
if plot_type == 'all':
    plt.scatter(new_lma_r[(new_dists<max_dist)&(new_dists>-max_dist)]/1e3, 
               lma_z_ktyx[(new_dists<max_dist)&(new_dists>-max_dist)]/1e3,
                       color='k',s=1)
plt.xlim(0, np.max(new_x)/1e3)
plt.ylim(0,6)
plt.ylabel('Altitude (km AGL)')
plt.xlabel('Distance (km)')
plt.tight_layout()
plt.savefig('KTYX_difph_{}cs_mx{}_xx{}_my{}_xy{}.png'.format(
            radar.time['units'].split(' ')[-1], mx, xx, my, xy),
           dpi=200, facecolor='white')


# In[45]:


kdpdata = pyart.retrieve.kdp_vulpiani(radar, 
                                      phidp_field ='differential_phase', 
                                      band ='S' , windsize = 34)
kdpdata = kdpdata[0]['data'] #Retreive the kdp 'data' from the the kdpdata that was calculated using the Vulpiani method 
mask = np.logical_and(kdpdata > -0.01, kdpdata < 0.01) #mask the values from -0.01 to 0.01 so they do not plot
kdpdata = np.where(mask, np.nan, kdpdata) #Apply the mask to kdpdata
radar.add_field('specific_differential_phase_hv', {'data': kdpdata.data})  #Add a new field to the radar data dictionary with the kdp data


# In[46]:


fig = plt.figure(figsize=(8,3))
plt.contourf(new_x/1e3, desz_grid[:,0]/1e3,
             kdpdata.ravel()[indext].reshape(np.shape(desz_grid)),
             levels = np.arange(-1,4.1,0.1),
             cmap='pyart_SCook18', )
plt.colorbar(label='Differential Phase (degrees)')
if plot_type == 'flashes_only':
    for flash in flashes.flash_id[flashes.n_points>=min_events_per_flash]:
            plt.scatter( new_lma_r[(flash_events[selection_event].flash_id == flash)&(new_dists<max_dist)&(new_dists>-max_dist)]/1e3, 
                        lma_z_ktyx[(flash_events[selection_event].flash_id == flash)&(new_dists<max_dist)&(new_dists>-max_dist)]/1e3,
                       color='k',s=1) 
if plot_type == 'all':
    plt.scatter(new_lma_r[(new_dists<max_dist)&(new_dists>-max_dist)]/1e3, 
               lma_z_ktyx[(new_dists<max_dist)&(new_dists>-max_dist)]/1e3,
                       color='k',s=1)
plt.xlim(0, np.max(new_x)/1e3)
plt.ylim(0,6)
plt.ylabel('Altitude (km AGL)')
plt.xlabel('Distance (km)')
plt.tight_layout()
plt.savefig('KTYX_phidp_{}cs_mx{}_xx{}_my{}_xy{}.png'.format(
            radar.time['units'].split(' ')[-1], mx, xx, my, xy),
           dpi=200, facecolor='white')

plt.show()
# In[ ]:





# In[ ]:




