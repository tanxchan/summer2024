#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:12:59 2024

@author: tshao
"""

#(30)                Proton               ch01                   (0.78-1.09             MeV)(counts/cm2/str/sec/MeV)
#(31)                Proton               ch02                   (0.98-1.38)
#(32)                Proton               ch03                   (1.21-1.60)
#(33)                Proton               ch04                   (1.42-1.89)
#(34)                Proton               ch05                   (1.82-2.38)
#(35)                Proton               ch06                   (2.23-3.03)
#(36)                Proton               ch07                   (2.81-3.75)
#(37)                Proton               ch08                   (3.67-5.10)
#(38)                Proton               ch09                   (4.87-7.03)
#(39)                Proton               ch10                   (7.01-10.35)
#(40)                Proton               ch11                   (9.70-16.66)
#(41)                Proton               ch12                   (16.06-30.96)
#(42)                Proton               ch13                   (25.63-46.56)
#(43)                Proton               ch14                   (38.47-94.54)
#(44)                Proton               ch15                   (81.39-193.55)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl

import datetime as dt

import os
from matplotlib.backends.backend_pdf import PdfPages


d1 = dt.datetime(2015, 3, 10)
d2 = dt.datetime(2015, 3, 14)
    
path = r'/Users/tshao/Downloads'

ranges = [(0.78,1.09),(0.98,1.38),(1.21,1.38),(1.21,1.60),(1.42,1.89),(1.82,2.38),(2.23,3.03),
          (2.81,3.75),(3.67,5.10),(4.87,7.03),(7.01,10.35),(9.70,16.66),(16.06,30.96),(25.63,46.56),
          (38.49,94.54),(81.39,193.55)]

columnNames = ['UT','GPS','Longitude','Latitude','Altitude',
'MLON','MLAT','MLT','LVALUE','BABS','B0','BN','BE','BD','ECL',
'Electron','Electron','Electron','Electron','Electron','Electron','Electron',
'sigma','sigma','sigma','sigma','sigma','sigma','sigma',
'Proton','Proton','Proton','Proton','Proton','Proton','Proton','Proton','Proton','Proton','Proton','Proton','Proton','Proton','Proton',
'sigma','sigma','sigma','sigma','sigma','sigma','sigma','sigma','sigma','sigma','sigma','sigma','sigma','sigma','sigma',
'Alpha','Alpha','Alpha','Alpha','Alpha','Alpha','sigma','sigma','sigma','sigma','sigma','sigma',
'Heavy','a','X','Y','Z','FLAG_INTERPOLATED','MLON_MAG','MLAT_MAG','MLT_SM',]



def getPath(d1):
    filepath = os.path.join(path, d1.strftime('%y%m') + '.txt')
    return filepath

head = pd.read_fwf(getPath(d1), nrows = 80)

def getRange(d1, d2):
    td = dt.timedelta(days = 29)
    if d1.day>d2.day:
        days = np.arange(d1, d2+td, td)
    else:
        days = np.arange(d1, d2, td)
    bff = pd.DataFrame(data = [])
    for day in days:
        filepath = getPath(day.item())
        file = pd.read_fwf(filepath, skiprows=range(81), Header=None)
        file.columns=columnNames
        print(file)
        bff = pd.concat([bff,file], ignore_index=True)
    return bff

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
    return i

file = getRange(d1,d2)
    

proton = file['Proton']
proton = np.transpose(proton.to_numpy())
use = file[['UT', 'GPS', 'Longitude', 'Latitude', 'Altitude','LVALUE']]

use.loc[:, 'Longitude'] = (use['Longitude']+180)%360-180
use.loc[:, 'UT'] = [dt.datetime.strptime(i, '%Y/%m/%d-%H:%M:%S') for i in use['UT']]


def dbd(d1,d2,i=-1):
    td = dt.timedelta(days=1)
    days = np.arange(d1,d2+td,td)
    pp = PdfPages(d1.strftime('Documents/%Y%m%d_i.pdf'))
    if i != -1:
        for day in days:
            start = day.item().replace(hour=0,minute=0,second=0)
            end = day.item().replace(hour=23,minute=59,second=59)
            pp.savefig(plot(i,start,end))
    else:
        for i in range(len(ranges)-1):
            pp.savefig(plot(i,d1,d2.replace(hour=23,minute=59,second=59)))
    pp.close()

pp = PdfPages(d1.strftime('/Documents/%Y%m%d.pdf'))
def plot(i, d1, d2):
    i1 = binSearch(use['UT'], d1)
    i2 = binSearch(use['UT'], d2)
    fig = plt.figure(figsize=(8,8))
    map = Basemap(projection='cyl', resolution = 'c', llcrnrlat = -90,
                  urcrnrlat=90, llcrnrlon = -180, urcrnrlon=180, )
    #map = Basemap(projection='ortho', resolution='l', lat_0=-26.6, lon_0=-49.0)
    #map.bluemarble()
    map.drawcoastlines(color = 'gray')
    parallels = np.arange(-80,81,20.)
    map.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(-350.,351.,40.)
    map.drawmeridians(meridians,labels=[True,False,False,True])
    
    scale = mpl.colors.LogNorm(vmin = 1, vmax = 10**2)
    
    Lons = use.loc[i1:i2, ['Longitude']].to_numpy()
    Lons.transpose()
    Lats = use.loc[i1:i2, ['Latitude']].to_numpy()
    Lats.transpose()
    Lons, Lats = map(Lons, Lats)
    map.scatter(Lons, Lats, 
                norm=scale, alpha = 1,
                c=proton[i, i1:i2+1].transpose(), s = 0.1, cmap = 'rainbow')
    title = 'Proton Counts from ' + d1.strftime('%m/%d/%Y to ') + d2.strftime('%m/%d/%Y ') + ' MeV:' + str(ranges[i][0]) + '-' + str(ranges[i][1]) 
    plt.title(title)
    plt.colorbar(label = 'counts/cm2/str/sec/MeV', location='bottom', shrink=1)
    data = use.loc[i1:i2, ['LVALUE']].to_numpy()
    point_mask = np.isfinite(data)  # Points to keep.
    x = Lons[:,0]
    y = Lats[:,0]
    z = data[:,0]
    levels = [1.5,2,3,4,5]
    a = plt.tricontour(x[point_mask[:,0]], y[point_mask[:,0]], z[point_mask[:,0]],
                   colors = ['black'], levels = levels, alpha = 0.9, linewidths = 1)
    plt.clabel(a, a.levels, inline=True, fontsize = 10)
    plt.tight_layout()
    return fig
#for i in range(15):
    #pp.savefig(plot(i))
dbd(d1,d2,-1)
#pp.close()