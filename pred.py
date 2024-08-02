#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:11:40 2024

@author: tshao
"""

import pandas as pd
import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import r_regression
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
'''
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D'''

from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages

import keras
import torch
import tensorflow as tf
#import tensorflow_decision_forests as tfdf
from sklearn.model_selection import TimeSeriesSplit

d1 = dt.datetime(2015, 3, 15)#start date

d2 = dt.datetime(2015, 3, 24, 0, 1)#end date

#d1 = dt.datetime(2024, 5, 10) #beginning of analysis
#d2 = dt.datetime(2024, 5, 14, 0, 1) #end of analysis

ref = dt.datetime(2015, 3, 15)
#ref = dt.datetime(2024, 5, 8)#beginning of omni file
#scuffed because omni data is by request of a range

avglength = dt.timedelta(days =1)#minutes=30)#length of running averages
#u can make it 1 minute to make it the same as the true values

#print(t)


ranges = [(0.78,1.09),(0.98,1.38),(1.21,1.38),(1.21,1.60),(1.42,1.89),(1.82,2.38),(2.23,3.03),
          (2.81,3.75),(3.67,5.10),(4.87,7.03),(7.01,10.35),(9.70,16.66),(16.06,30.96),(25.63,46.56),
          (38.49,94.54),(81.39,193.55)]

#Useful functions
def binSearch(arr,d):
    i1 = 0
    i2 = len(arr)-1
    i = 0
    while (i1<i2):
        i = int((i1+i2)/2)
        if arr[i]<d:
            i1 = i + 1
        elif arr[i]>d:
            i2 = i
        else:
            return i
    return i

def transmute(file, dates, d1, d2, td):
    #np.arange(d1, d2, dt.datetime(minutes=1))
    index = binSearch(dates, d1)
    total = 0
    n = 0
    arr = []
    d1 = d1+td
    while d1<=d2 and index<len(file):
        total = total + file[index]
        #print(total)
        n = n + 1
        if dates[index]>d1 or index == len(file)-1:
            print(dates[index], n)
            d1 = d1+td
            avg = np.NAN if n == 0 else total/n
            n = 0
            total = 0
            arr.append(avg)
        index = index + 1
    return arr

def runningavg(file, dates, d1, d2, td, avglength):#also functions as transmute
    #O(n) because it uses sliding window
    n = 0
    total = 0
    start = d1
    end = d1+avglength
    i1 = binSearch(dates,start)
    i2 = binSearch(dates,end)
    total = sum(file[i1:i2])
    arr = []
    while end < d2:
        start = start + td
        end = end + td
        while i2<len(file) and dates[i2] < end:
            total += file[i2]
            i2 += 1
        while dates[i1] < start:
            total -= file[i1]
            i1 += 1
        n = i2-i1
        #avg is 0 because or else it doesn't work with linreg
        #if u wanna do smth smart then probably do pd.dropna but like this works well enough
        avg = 0 if n == 0 else total/n
        arr.append(avg)
    return arr

#file formatting is disgusting, functions to get dataframes from them
def getOmni(ref, d1, d2, trim = False):#ref is the beginning of the omni file
    omnipath = r'/Users/tshao/Documents/omni'
    cols = ['Day','Time','$B_Z-IMF$','$V_{SW}$','$N_{SW}$','AE']
    filepath = os.path.join(omnipath, ref.strftime('%y%m%d.txt'))
    null_vals = [np.NAN, np.NAN, 9999.99, 99999.9, '999.990', 99999]#values that = null in file fmt
    omni = pd.read_fwf(filepath, header = None, skiprows=range(68), na_values=null_vals)
    omni.drop(axis=0, inplace=True, index=range(len(omni)-3, len(omni)))#bottom 3 lines are weird
    omni.columns = cols
    #omni.drop(columns = 'AE', axis=1, inplace=True)#the AE is really missing
    dts = []
    for i in range(len(omni['Day'])):#could be replaced by an arange
        d = dt.datetime.strptime(str(omni.iloc[i,0])+str(omni.iloc[i,1]), '%d-%m-%Y%H:%M:%S.000')
        dts.append(d)
    omni.insert(0, 'Datetime', dts)
    omni.drop(columns = ['Day', 'Time'], axis=1, inplace=True)
    if trim:
        start = binSearch(omni['Datetime'], d1)
        end = binSearch(omni['Datetime'], d2-dt.timedelta(minutes=1))
        omni = omni[start:end+1]
    return omni


def getdst(d1):
    dstpath = r'/Users/tshao/Documents/dst'
    dstpath = os.path.join(dstpath, d1.strftime('dst%y%m.dat'))
    cols = ['Date', 'Cent', 'Base']#column names
    nulls = [np.NAN, np.NAN]#null values, don't really need
    w = [10,6,4]#widths for readfwf
    for i in range(24):
        cols.append(i)
        nulls.append(9999)
        w.append(4)
    cols.append('DMV')
    nulls.append(9999)
    w.append(4)
    
    dst = pd.read_fwf(dstpath, header=None, na_values=nulls, widths=w)
    dst.columns = cols
    t = []
    dsts = []
    for i in range(len(dst)):
        for j in range(24):
            dsts.append(dst.loc[i, j])
            t.append(dt.datetime.strptime(dst.loc[i, 'Date'] + str(j), 'DST%y%m*%d%H'))
    return dsts, t

def getCHD(d1):
    path = r'/Users/tshao/Documents/data'
    filepath = os.path.join(path,d1.item().strftime('%m%d%y.dat'))
    return pd.read_csv(filepath, sep = r'\s+', header = None, engine = 'python')
    
def getCHDRange(d1, d2):
    columns = ['Time', 'Time duration', 'CHD-X', 'CHD-Y', 'Latitude', 'Longitude', 'Altitude', 'Flag']
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
    file['Time'] = pd.to_datetime(file['Time'], unit='s')
    return file

def getPath(d1):
    path = r'/Users/tshao/Downloads'
    filepath = os.path.join(path, d1.strftime('%y%m') + '.txt')
    return filepath

def getRange(d1, d2):
    columnNames = ['UT','GPS','Longitude','Latitude','Altitude',
                    'MLON','MLAT','MLT','LVALUE','BABS','B0','BN','BE','BD','ECL',
                    'Electron','Electron','Electron','Electron','Electron','Electron','Electron',
                    'sigma','sigma','sigma','sigma','sigma','sigma','sigma',
                    'Proton','Proton','Proton','Proton','Proton','Proton','Proton','Proton','Proton','Proton','Proton','Proton','Proton','Proton','Proton',
                    'sigma','sigma','sigma','sigma','sigma','sigma','sigma','sigma','sigma','sigma','sigma','sigma','sigma','sigma','sigma',
                    'Alpha','Alpha','Alpha','Alpha','Alpha','Alpha','sigma','sigma','sigma','sigma','sigma','sigma',
                    'Heavy','a','X','Y','Z','FLAG_INTERPOLATED','MLON_MAG','MLAT_MAG','MLT_SM',]
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
        
#LSTM code:
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

#pretty plot functions:

def overlay(data, dts, title):
    fig, ax = plt.subplots(nrows=len(data)+1, ncols=1, sharex=True, layout='tight', figsize=(11,8))
    
    for i in range(len(data)):
        ax[i].set_ylabel(cols[i+1])
        #ax[i].scatter(dts, data[i], s=0.1)
        ax[i].plot(dts,data[i])
        ax[i].set_xticks(np.arange(dts[0],dts[len(dts)-1],dt.timedelta(days = 2.5)))
        
    i1 = binSearch(t, d1)
    i2 = binSearch(t, d2)
    
    ax[-1].plot(t[i1:i2], dsts[i1:i2])
    ax[-1].set_ylabel('DST')
    ax[-1].set_xticks(np.arange(dts[0],dts[len(dts)-1],dt.timedelta(days = 2.5)))
    
    #plt.title('Proton')
    ax[0].set_title(title)#d1.strftime('Storm on %Y/%m/%d'))
    return fig, ax

def regplot(avged, s, cols):
    for i in range(len(avged)-1):
        pred = [[avged[i][j]] for j in range(len(avged[i]))]
        
        reg = LinearRegression().fit(pred, avged[-1])
        #print(cols[i+2] + ' Correlation: ' + str(reg.score(pred, avged[5])))
        #print('Coefficient: ' + str(reg.coef_[0]))
        
        plt.scatter(avged[i], avged[-1], s=s)
        x = np.linspace(min(avged[i]), max(avged[i]), 100)
        plt.plot(x, x*reg.coef_ + reg.intercept_, label = 'x*'+str(reg.coef_[0])+
                 '+'+str(reg.intercept_), color = 'orange')
        
        r = r_regression(np.array(avged[i]).reshape(-1,1),
                         np.array(avged[-1])).reshape(-1,1)[0][0]
        print(cols[i+1] + ' Pearson R: ' + 
              str(r))
        plt.legend()
        plt.title(cols[i+1] + ' vs. ' + cols[-1] + ' Correlation: ' + str(r))
        
        plt.xlabel(cols[i+1])
        plt.ylabel('Proton')
        plt.show()

def loss(history):
    fig = plt.Figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val loss'])
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss')
    return fig

def predplot(reg, model, ravged, X):
    t1 = np.arange(d1+avglength,d2,dt.timedelta(minutes=1))
    fig = plt.Figure()
    plt.plot(t1, reg.predict(np.array(ravged[:-1]).transpose()),
             label = 'Linear Regression', linewidth = 1, alpha = 0.5, color = 'r')
    plt.plot(t1[2:], model.predict(X)[:,0], label = 'LSTM', 
             linewidth = 1, alpha = 0.5, color = 'g')
    plt.plot(t1, ravged[-1], label = 'Actual', linewidth = 1,
             alpha = 0.5, color = 'blue')
    plt.legend()
    plt.xticks(np.arange(d1, d2, dt.timedelta(days = 2)))
    plt.ylabel('Average Proton Count')
    plt.title('Predicted and Actual with avglength ' + str(avglength))
    plt.xlabel('Date')
    return fig

omni = getOmni(ref, d1, d2, trim = True)
#actual code to be run:
dsts, t = getdst(d1)

if d1>dt.datetime(2018, 3, 1, 0):
    chd = getCHDRange(d1, d2)
    
    avgchd = transmute(chd['CHD-X'], chd['Time'], d1, d2, dt.timedelta(minutes=1))
    ravgchd = runningavg(chd['CHD-X'], chd['Time'], d1, d2, dt.timedelta(minutes=1), dt.timedelta(days=1))
    
    omni.insert(len(omni.columns), 'Longitude', 
                transmute(chd['Longitude'], chd['Time'], d1, d2, dt.timedelta(minutes=1)))#insert the averaged long
    omni.insert(len(omni.columns), 'Latitude', 
                transmute(chd['Latitude'], chd['Time'], d1, d2, dt.timedelta(minutes=1)))#insert the averaged lat
    omni.insert(len(omni.columns), 'Electron', avgchd)#insert the averaged electron count


#get proton data:
if d2<dt.datetime(2018, 3, 1, 0):
    file = getRange(d1,d2)#proton data
    proton = file['Proton']
    proton = np.transpose(proton.to_numpy())
    use = file[['UT', 'GPS', 'Longitude', 'Latitude', 'Altitude','LVALUE']]
    
    use.loc[:, 'Longitude'] = (use['Longitude']+180)%360-180
    use.loc[:, 'UT'] = [dt.datetime.strptime(i, '%Y/%m/%d-%H:%M:%S') for i in use['UT']]
    
    td = dt.timedelta(minutes=1)
    p = transmute(proton[0],use['UT'], d1, d2, td)[1:]#i think the first datapoint is weird so
    l = transmute(use['LVALUE'],use['UT'], d1, d2, td)[1:]#yeah
    #cannot be fucked to fix the issue
    omni.insert(len(omni.columns), 'L-Value', l, True)
    
    omni.insert(len(omni.columns), 'ISS Proton Count', p, True)
    #p = runningavg(proton[0],use['UT'], d1, d2, td, avglength)


#avglength = dt.timedelta(days=1)

#lmao i shoulda done this with the proton data but oop wtvr

omni.dropna(inplace=True)#Uncomment if doing actual analysis

cols = omni.columns

data = np.array([[float(i) for i in omni[cols[j]]] for j in range(1,len(cols))])

#fig.show()
#pred = data[0:5].transpose()

#scuffed code for linreg with log
mask = [False if data[5][i] == 0 else True for i in range(len(data[0]))]

logprot = np.log(data[5][mask])

for i in range(5):
    pred = data[i].transpose()
    pred = pred.reshape(-1,1)
    
    reg = LinearRegression().fit(pred[mask], logprot)
    
    print(cols[i+1] + ' Correlation:' + str(reg.score(pred[mask], logprot)))

#b = transmute(data[0], dts, d1,d2,dt.timedelta(days=1))

overlay(data, [i.to_pydatetime() for i in omni['Datetime']], d1.strftime('True values for storm on %Y/%m/%d'))
plt.show()

#p = transmute(proton[0], use['UT'], d1,d2,dt.timedelta(days=1))

#avged = [transmute(data[i], dts, d1,d2,dt.timedelta(days=1)) for i in range(len(data))]

#avged[5] = p

#overlay(avged, np.arange(d1 + dt.timedelta(days=1),d2,dt.timedelta(days=1)),
 #       d1.strftime('Daily average values for storm on %Y/%m/%d'))
#plt.show()

#regplot(data, 10, cols)
#d = runningavg(data[0],dts, d1, d2, td, dt.timedelta(days=1))

#plt.tight_layout()

#running average

#overlays = PdfPages(d1.strftime('./%Y%m%d_o.pdf'))
predictions = PdfPages(d1.strftime('./%Y%m%d_p.pdf'))
for h in range(3, 49, 3):
    avglength = dt.timedelta(hours=h)
    x = np.arange(d1+avglength, d2, dt.timedelta(days=0, minutes = 1))
    
    ravged = [runningavg(data[i], [i.to_pydatetime() for i in omni['Datetime']], d1, d2, dt.timedelta(days=0, minutes = 1), avglength) for i in range(len(data))]
    
    #o, ax = overlay(ravged, x, d1.strftime('Running average values for storm on %Y/%m/%d with avglength ' + str(avglength)))
    #overlays.savefig(o)
    #plt.show()
    
    dataset = np.array(ravged).transpose()
    #dataset = np.array(data).transpose()
    # choose a number of time steps
    n_steps = 3
    # convert into input/output
    X, y = split_sequences(dataset, n_steps)
    # the dataset knows the number of features, e.g. 2
    n_features = X.shape[2]
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    
    tscv = TimeSeriesSplit()
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index, :], X[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
    
    ''''history = model.fit(#use the timeseries split, which is supposed to prevent overfitting
        X_train,
        y_train,
        batch_size=64,
        epochs=100,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(X_test, y_test),
    )'''''
    
    history = model.fit(X, y, epochs=10)
    
    reg = LinearRegression().fit(np.array(ravged[0:-1]).transpose(), ravged[-1])
    
    plt.show()
    preds = predplot(reg, model, ravged, X)
    predictions.savefig(preds)
    plt.show()
    
#overlays.close()
predictions.close()
#regplot(ravged, 10, cols)



