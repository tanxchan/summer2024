import spacepy.irbempy as ib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap

import spacepy.time as spt
import spacepy.coordinates as spc

import datetime as dt

import os

months = ['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
columns = ['Time', 'Time duration', 'CHD-X', 'CHD-Y', 'Latitude', 'Longitude', 'Altitude', 'Flag']

graphed = 'CHD-X'

alt = 6600

ref = dt.datetime(2015, 9, 10, 16)

lat1 = -89
lat2 = 89
lon1 = -180
lon2 = 180
dlat = 1
dlon = 1

path = r'/data/cferrada/calet/CHD/level1.1/obs'
savePath = r'/home/tshao/imgs/'

d1 = dt.datetime(2023, 4, 22)
d2 = dt.datetime(2023, 4, 25)

def getCHD(d1):
    filepath = os.path.join(path,d1.item().strftime('%Y'),'CHD_' + d1.item().strftime('%y%m%d') + '.dat')
    return pd.read_csv(filepath, sep = r'\s+', header = None, engine = 'python')
    
def getRange(d1, d2):
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

def plotCHD(d1, d2, data, lons, lats):
    plt.subplots(figsize=(11,3))
    
    map = Basemap(projection='cyl', resolution = 'l', llcrnrlat = -90,
                  urcrnrlat=90, llcrnrlon = -180, urcrnrlon=180, )
    map.drawcoastlines(color = 'gray')
    parallels = np.arange(-80,81,20.)
    map.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(-350.,351.,40.)
    map.drawmeridians(meridians,labels=[True,False,False,True])
    
    lons, lats = map(lons, lats)
    xx, yy = np.meshgrid(lons, lats)
    levels = [1.5,2,3,4,5]
    a = map.contour(xx, yy, data, levels, color = 'grey')#plot l shells
    
    scale = mpl.colors.LogNorm(vmin = 10**2, vmax = 10**6)
    
    CHD = getRange(d1,d2)
    lons, lats = map(CHD['Longitude'], CHD['Latitude'])
    b = map.scatter(lons, lats,  
                    c=CHD['CHD-X'], 
                    s = 1, 
                    norm = scale, 
                    cmap = 'winter')
    plt.text(-170,-80, 'CHD-X(>1.5MeV)')
    #plt.colorbar(a, label = 'L-Shells', location='left', ticks = levels)
    plt.clabel(a, a.levels, inline=True, fontsize = 10)
    plt.colorbar(b, label = 'Count Rate(s^-1)', location = 'right')
    
    title = 'Storm on' + d1.strftime(' %B %d, %Y')
    if d1!=d2:
        title = title + d2.strftime(' to %B %d, %Y')
    
    plt.title(title)
    
    exists = False
    for root, dirs, files in os.walk(savePath, topdown=False):
        for name in dirs:
            if name == d1.strftime('%y%m%d'):
                exists = True
    if not exists:
        os.mkdir(os.path.join(savePath, d1.strftime('%y%m%d')))
    plt.savefig(os.path.join(savePath, d1.strftime('%y%m%d'), title + '.png'))
    plt.close()

def lrange(lons, lat):
    coords = np.ndarray(shape=[lons.shape[0],3])
    for i in range(lons.shape[0]):
        coords[i] = [alt, lat, lons[i]]
    loci = spc.Coords(coords, 'GDZ', 'sph')
    ticks = spt.Ticktock(np.full((lons.shape[0]),ref),'UTC')
    l = ib.get_Lm(ticks, loci,[90])
    return l

def lshell(lons, lats):
    ls = np.ndarray(shape=[lats.shape[0],lons.shape[0]])
    for i in range(lats.shape[0]):
        ls[i] = lrange(lons, lats[i])['Lm'][:,0]
    return ls

lats = np.arange(lat1, lat2, dlat)
lons = np.arange(lon1, lon2, dlon)

data = lshell(lons, lats)

plotCHD(d1,d2,data,lons,lats)
days = np.arange(d1,d2+dt.timedelta(days=1),dt.timedelta(days=1))
for day in days:
    plotCHD(day.item(),day.item(),data,lons,lats)
