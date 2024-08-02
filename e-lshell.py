#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:21:30 2024

@author: tshao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 3:23:06 2024

@author: tshao
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl

import datetime as dt

import os
from matplotlib.backends.backend_pdf import PdfPages


#d1s = [dt.datetime(2023, 4, 22), dt.datetime(2023, 3, 22), dt.datetime(2023, 11, 4)]
#d2s = [dt.datetime(2023, 4, 24), dt.datetime(2023, 3, 24), dt.datetime(2023, 11, 6)]

d1 = dt.datetime(2024, 5, 10)
d2 = dt.datetime(2024, 5, 16)

graphed = 'CHD-X'

def getCHD(d1):
    if d1.item()<dt.datetime(2022,1,1):
        CHDpath = r'/data/cferrada/calet/CHD/level1.1/obs'
        filepath = os.path.join(CHDpath,d1.item().strftime('%Y'),'CHD_' + d1.item().strftime('%y%m%d') + '.dat')
        return pd.read_csv(filepath, sep = r'\s+', header = None, engine = 'python')
    if d1.item()>dt.datetime(2024,5,15):
        CHDpath = r'/home/tshao/e-'
        filepath = os.path.join(CHDpath, d1.item().strftime('%m%d%y.dat'))
        return pd.read_csv(filepath, sep = r'\s+', header = None, engine = 'python')
    CHDpath = r'/home/tshao/e-l'
    filepath = os.path.join(CHDpath,d1.item().strftime('%m%d%y_L.dat'))
    return pd.read_csv(filepath, sep = r'\s+', header = None, engine = 'python')

def getCHDRange(d1, d2):
    columns = ['Time', 'Time duration', 'CHD-X', 'CHD-Y', 'Latitude', 'Longitude', 'Altitude', 'Flag', 'L']
    td = dt.timedelta(days=1)
    days = np.arange(d1, d2, td)
    file = pd.DataFrame(data = [])
    for day in days:
        file = pd.concat([file,getCHD(day)], ignore_index=True)
    #hello world
    file.columns = columns[:len(file.columns)]
    for i in range(len(file)):
        if file['Longitude'][i]>180:
            file.loc[i, 'Longitude'] = file['Longitude'][i] - 360
    file['Time'] = pd.to_datetime(file['Time'], unit='s')
    return file
#head = pd.read_fwf(getPath(d1), nrows = 80)

def binSearch(arr,d):
    i1 = 0
    i2 = len(arr)-1
    while (i1<=i2):
        i = int((i1+i2)/2)
        if arr[i]<d:
            i1 = i + 1
        elif arr[i]>d:
            i2 = i - 1
        else:
            return i

    
def plot(d1, d2, lat1=-90, lat2=90, lon1=-180, lon2=180):
    fig = plt.figure(figsize=(5.5,4.25))
    map = Basemap(projection='cyl', resolution = 'c', llcrnrlat = lat1,
                  urcrnrlat=lat2, llcrnrlon = lon1, urcrnrlon=lon2, )
    ref1 = dt.datetime(2024,5,10)
    ref2 = dt.datetime(2024,5,10,23,59,59)
    if d1<=dt.datetime(2024,5,15):
        ref1 = d1
        ref2 = d2
    #plot electron
    #fig = plt.figure(figsize=(11,8.5))
    #map = Basemap(projection='ortho', resolution='l', lat_0=-26.6, lon_0=-49.0)
    #map.bluemarble()
    map.drawcoastlines(color = 'gray')
    parallels = np.arange(-80,81,20.)
    map.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(-350.,351.,40.)
    map.drawmeridians(meridians,labels=[True,False,False,True])

    scale = mpl.colors.LogNorm(vmin = 10**2, vmax = 10**6)

    CHD = getCHDRange(d1,d2)
    lons, lats = map(CHD['Longitude'], CHD['Latitude'])
    b = map.scatter(lons, lats,
                    c=CHD['CHD-X'],
                    s = 1,
                    norm = scale,
                    cmap = 'rainbow')
    plt.text(-170,-80, 'CHD-X(>1.5MeV)')
    plt.colorbar(b, label = 'Count Rate(s^-1)', location = 'bottom')
    #plot L-shells

    CHD = getCHDRange(ref1,ref2)
    lons,lats = map(CHD['Longitude'], CHD['Latitude'])
    data = CHD['L']
    #point_mask = np.isfinite(data)  # Points to keep.
    x = lons
    y = lats
    z = data
    levels = [1.5,2,3,4,5,8]
    a = plt.tricontour(x, y, z, #x[point_mask[:,0]], y[point_mask[:,0]], z[point_mask[:,0]],
                   colors = ['black'], levels = levels, alpha = 0.9, linewidths = 1)
    plt.clabel(a, a.levels, inline=True, fontsize = 10)
    #plot ion
    title = 'Electron Counts from' + d1.strftime(' %B %d, %Y')
    if d1-d2>dt.timedelta(days=1):
        title = title + d2.strftime(' to %B %d, %Y')
    plt.title(title)
    plt.tight_layout()
    return fig

def ltime(CHD, color = [], lat1=-91, lat2=91, lon1=-181, lon2=181, cmin = 0):
    #l time plot lmao
    mask = [True if color[i]>=cmin and CHD['Longitude'][i]<lat2 and CHD['Longitude'][i]>lat1
            and CHD['Latitude'][i]<lon2 and CHD['Latitude'][i]>lon1 else False for i in range(len(CHD))]
    fig = plt.figure(figsize=(10,4.25))
    time = CHD['Time']
    data = CHD['L']
    if not color.empty:
        scale = mpl.colors.LogNorm(vmin = 10**2, vmax = 10**6)
        plt.scatter(time[mask], data[mask], c=color[mask], cmap='rainbow', norm=scale, s=0.01)
        plt.colorbar(label = 'CHD-X(>1.5MeV)')
    else:
        plt.scatter(time[mask], data[mask], s = 0.1)
    plt.ylabel('L-shell')
    plt.ylim(0,8)
    plt.xticks(np.arange(time[0], time[len(time)-1], dt.timedelta(hours = 24)))
    plt.tight_layout()
    return fig

def dbd(d1,d2):
    td = dt.timedelta(days=1)
    days = np.arange(d1,d2+td,td)
    pp = PdfPages(d1.strftime('../imgs/%Y%m%d.pdf'))
    CHD = getCHDRange(d1,d2.replace(hour=23,minute=59,second=59))
    pp.savefig(ltime(CHD, color=CHD['CHD-X']))
    for day in days:
        start = day.item().replace(hour=0,minute=0,second=0)
        end = day.item().replace(hour=23,minute=59,second=59)
        pp.savefig(plot(start,end))
    pp.close()

#for i in range(len(d1s)):
#    dbd(d1s[i],d2s[i])
#dbd(d1, d2)
CHD = getCHDRange(d1,d2.replace(hour=23,minute=59,second=59))
fig = ltime(CHD, color=CHD['CHD-X'], cmin=10**4)
fig.savefig(d1.strftime('../imgs/%Y%m%d_ltime.png'))
                                                          