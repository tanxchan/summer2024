#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:23:06 2024

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


d1s = [dt.datetime(2023, 4, 22), dt.datetime(2023, 3, 22), dt.datetime(2023, 11, 4)]
d2s = [dt.datetime(2023, 4, 24), dt.datetime(2023, 3, 24), dt.datetime(2023, 11, 6)]

d1 = dt.datetime(2024, 5, 11)
d2 = dt.datetime(2024, 5, 15)

ref1 = dt.datetime(2018, 3, 1)
ref2 = dt.datetime(2018, 3, 9)

if (d2<ref2 and d1>ref1):
    ref1 = d1
    ref2 = d2

path = r'/Users/tshao/Downloads'

CHDpath = r'/Users/tshao/Documents/data'

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
months = ['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
columns = ['Time', 'Time duration', 'CHD-X', 'CHD-Y', 'Latitude', 'Longitude', 'Altitude', 'Flag']

graphed = 'CHD-X'


def getPath(d1):
    filepath = os.path.join(path, d1.strftime('%y%m') + '.txt')
    return filepath

def getCHD(d1):
    #filepath = os.path.join(CHDpath,d1.item().strftime('%Y'),'CHD_' + d1.item().strftime('%y%m%d') + '.dat')
    filepath = os.path.join(CHDpath,d1.item().strftime('%m%d%y.dat'))
    return pd.read_csv(filepath, sep = r'\s+', header = None, engine = 'python')
    
def getCHDRange(d1, d2):
    td = dt.timedelta(days=1)
    days = np.arange(d1, d2+td, td)
    file = pd.DataFrame(data = [])
    for day in days:
        file = pd.concat([file,getCHD(day)], ignore_index=True)
    #hello world
    file.columns = columns
    for i in range(len(file)):
        if file['Longitude'][i]>180:
            file.loc[i, 'Longitude'] = file['Longitude'][i] - 360
    return file
#head = pd.read_fwf(getPath(d1), nrows = 80)

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

file = getRange(ref1,ref2)

def plot(d1,d2):
    #plot electron
    #fig = plt.figure(figsize=(11,8.5))
    fig = plt.figure(figsize=(5.5,4.25))
    map = Basemap(projection='cyl', resolution = 'c', llcrnrlat = -90,
                  urcrnrlat=90, llcrnrlon = -180, urcrnrlon=180, )
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

    use = file[['UT', 'GPS', 'Longitude', 'Latitude', 'Altitude','LVALUE']]
    
    use.loc[:, 'Longitude'] = (use['Longitude']+180)%360-180
    use.loc[:, 'UT'] = [dt.datetime.strptime(i, '%Y/%m/%d-%H:%M:%S') for i in use['UT']]

    i1 = binSearch(use['UT'], ref1)
    i2 = binSearch(use['UT'], ref2)
    Lons = use.loc[i1:i2, ['Longitude']].to_numpy()
    Lons.transpose()
    Lats = use.loc[i1:i2, ['Latitude']].to_numpy()
    Lats.transpose()
    Lons, Lats = map(Lons, Lats)
    
    data = use.loc[i1:i2, ['LVALUE']].to_numpy()
    point_mask = np.isfinite(data)  # Points to keep.
    x = Lons[:,0]
    y = Lats[:,0]
    z = data[:,0]
    levels = [1.5,2,3,4,5]
    a = plt.tricontour(x[point_mask[:,0]], y[point_mask[:,0]], z[point_mask[:,0]],
                   colors = ['black'], levels = levels, alpha = 0.9, linewidths = 1)
    plt.clabel(a, a.levels, inline=True, fontsize = 10)
    #plot ion
    title = 'Electron Counts from' + d1.strftime(' %B %d, %Y')
    if d1-d2>dt.timedelta(days=1):
        title = title + d2.strftime(' to %B %d, %Y')
    if d2<=ref2:
        i = 0
        proton = file['Proton']
        proton = np.transpose(proton.to_numpy())
        i1 = binSearch(use['UT'], d1)
        i2 = binSearch(use['UT'], d2)
        map.scatter(Lons, Lats, 
                    norm=scale, alpha = 1,
                    c=proton[i, i1:i2+1].transpose(), s = 0.1, cmap = 'winter')
        plt.colorbar(label = 'counts/cm2/str/sec/MeV', location='bottom', shrink=1)
        title = 'Proton Counts from ' + d1.strftime('%m/%d/%Y to ') + d2.strftime('%m/%d/%Y ') + ' MeV:' + str(ranges[i][0]) + '-' + str(ranges[i][1]) 
    plt.title(title)
    plt.tight_layout()
    return fig

def dbd(d1,d2):
    td = dt.timedelta(days=1)
    days = np.arange(d1,d2+td,td)
    pp = PdfPages(d1.strftime('Documents/%Y%m%d.pdf'))
    for day in days:
        start = day.item().replace(hour=0,minute=0,second=0)
        end = day.item().replace(hour=23,minute=59,second=59)
        pp.savefig(plot(start,end))
    pp.close()


for i in range(len(d1s)):
    dbd(d1s[i],d2s[i])
