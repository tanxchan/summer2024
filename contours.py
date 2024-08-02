#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:40:23 2024

@author: tshao
"""

import spacepy.irbempy as ib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap

import spacepy.time as spt
import spacepy.coordinates as spc

#import apexpy
import datetime as dt

def lshell(lon, lat):
     loci = spc.Coords([[400, lon, lat]], 'GDZ', 'sph')
     stime = dt.datetime(2017,9,10,0,0)
     ticks = spt.Ticktock(stime, 'UTC')
     alpha = [90]
     l = ib.get_Lm(ticks, loci, alpha)
     return l





stime = dt.datetime(2017, 9, 10, 0, 0)

map = Basemap(projection='cyl', resolution = 'l', llcrnrlat = -90, 
              urcrnrlat=90, llcrnrlon = -180, urcrnrlon=180, )

map.drawcoastlines(color = 'gray')

lonrange = np.arange(-180,180,1)
latrange = np.arange(-89, 89, 1)
lons, lats = np.meshgrid(lonrange,latrange)

print(lons, lats)

data = []
alt = 400
ticks = spt.Ticktock(stime, 'UTC')


for lat in latrange:
    dat = []
    for lon in lonrange:
        alpha = [90]
        loci = spc.Coords([[400, lon, lat]], 'GDZ', 'sph')
        l = ib.get_Lm(ticks, loci, alpha)
        
        #apex_iss = apexpy.Apex(stime, refh=alt)
        #alat, alon = apex_iss.geo2apex(lat, lon, alt)
        
        # Get the apex height
        #aalt = apex_iss.get_apex(alat, alt)
        
        # Convert from apex height in km to L-shell
        #l = 1.0 + aalt / apex_iss.RE      
        dat.append(l)
        
    data.append(dat)
    
map.contour(lons, lats, data)

