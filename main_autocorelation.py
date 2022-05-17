#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 00:09:13 2021

@author: vite
"""


from functions import *
from wrappers import *
from my_functions import *
import matplotlib.pyplot as plt
import neuroseries as nts
import pandas as pd
import pickle5 as pickle
import os
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import numpy as np

"""
Functions
""" 
def calc_width(aucorr_i):
    """
    

    Parameters
    ----------
    aucorr : pandas.core.series.Seriespandas.core.series.Series
                autocorrelogram.

    Returns
    -------
    positionINindex: position in the aucorr of the number close to the width
    number: y value of the aucorr  associated to the position
    width : in ms

    """
    aucorr = aucorr_i.values
    nbins = len(aucorr)
    array = aucorr[:int(nbins/2)+1] #get the first half
    array = np.flip(array) #invert order
    # array = array-1
    half= (array.max()-array.min())/2 #from peak to trough
    threshold = array.min() + half
    print(half, threshold)
    for i in array:
        if i>=threshold:
            number=i
            break
    positionINarray = np.where((aucorr==number)==True)[0][0]
    positionINindex = aucorr_i.index[positionINarray]
    width = abs(positionINindex)*2
    return positionINindex, number, width

session = 'B0702-211111'
data_directory_load = '/Users/vite/OneDrive - McGill University/PeyracheLab/Data/Tetractys/AB0702/' + session + '/data'
dir2save_plots = data_directory_load + '/plots'
if not os.path.exists(dir2save_plots):
    os.mkdir(dir2save_plots)
##############################################################################
# load data
spikes = pickle.load(open(data_directory_load + '/spikes.pickle', 'rb'))
shank = pickle.load(open(data_directory_load  + '/shank.pickle', 'rb'))
episodes = pickle.load(open(data_directory_load + '/episodes.pickle', 'rb'))
position = pickle.load(open(data_directory_load  + '/position.pickle', 'rb'))
wake_ep = pickle.load(open(data_directory_load  + '/wake_ep.pickle', 'rb'))
sleep_ep = pickle.load(open(data_directory_load  + '/sleep_ep.pickle', 'rb'))
#read sws
sws = pickle.load(open(data_directory_load  + '/sws.pickle', 'rb'))
sws_duration = sws['end'] - sws['start']
index = sws_duration.index[sws_duration==sws_duration.max()][0]
sws_sel = sws.iloc[[index]]
#read rem
rem = pickle.load(open(data_directory_load  + '/rem.pickle', 'rb'))
rem_duration = rem['end'] - rem['start']
index = rem_duration.index[rem_duration==rem_duration.max()][0]
rem_sel = rem.iloc[[index]]


##############################################################################
# Select shanks 
shanks_sel = [2,3]
neur_list = []
for i in range(len(shank)):
    if shank[i] in shanks_sel:
        neur_list.append(i)
neurons_sel = np.asarray(neur_list)
neurons_sel = [7,8,10,11,12,14,15,17,18,19]
spikes_sel = {key:val for key, val in spikes.items() if key in neurons_sel}
##############################################################################
# Compute autocorrelations

#Define parameters
binsize_wake = 2
nbins_wake = 30
binsize_sws = 2
nbins_sws = 30

#Calculate autocorrelations
autocorrs_wake, frates = compute_AutoCorrs(spikes_sel, wake_ep, 
                                binsize = binsize_wake, nbins=nbins_wake)
autocorrs_wake = autocorrs_wake.rolling(window=10, win_type='gaussian', 
                                center = True, min_periods = 1).mean(std = 2.0)
autocorrs_sws, frates = compute_AutoCorrs(spikes_sel, sws_sel, 
                                binsize = binsize_sws, nbins=nbins_sws)
autocorrs_sws = autocorrs_sws.rolling(window=10, win_type='gaussian', 
                                center = True, min_periods = 1).mean(std = 2.0)
autocorrs_rem, frates = compute_AutoCorrs(spikes_sel, rem_sel, 
                                binsize = binsize_wake, nbins=nbins_wake)
autocorrs_rem = autocorrs_rem.rolling(window=10, win_type='gaussian', 
                                center = True, min_periods = 1).mean(std = 2.0)

#Example
neuron = 2
ID = session
epoch = 'wake'
props = dict(boxstyle='round', facecolor='linen', edgecolor = 'linen', alpha=0.5)
pos, number, width = calc_width(autocorrs_rem[neuron])
plt.figure()
plt.plot(autocorrs_wake[neuron])
plt.plot(pos, number, '*')
plt.annotate("width = " + str(width), xytext=(15, 1.3), xy=(10.3, number), 
              arrowprops=dict(arrowstyle="->", color='black'), bbox = props, fontsize=8)
plt.title(session + " '" + epoch +"'")
plt.xlabel('Time (ms)')
plt.show()

##############################################################################
# Calculate with for all neurons in different epochs
autocorrs = [autocorrs_wake, autocorrs_sws, autocorrs_rem]  
epochs = ['wake', 'sws', 'rem']
#Calculate the widths, store them in a DataFrame and plot them on the autocorrelograms.
df_widths = pd.DataFrame(columns = ['indx','neuron', 'width', 'epoch'])
df_widths['indx'] = range(len(spikes.keys())*3)

# c = 0
# for autocorr, epoch in zip (autocorrs, epochs): 
#     for i in spikes.keys():
#         pos, number, width = calc_width(autocorr[i])
#         df_widths.loc[i+c, ('neuron')]= i
#         df_widths.loc[i+c, ('width')] = width
#         df_widths.loc[i+c, ('epoch')]= epoch
#     c+=len(spikes.keys())
c = 0
raws = round(len(spikes)/5)
for autocorr, epoch in zip (autocorrs, epochs): 
    plt.figure(figsize =(50, 50))
    for n,i in enumerate(spikes_sel.keys()):
        plt.subplot(5,raws+1,i+1)
        pos, number, width = calc_width(autocorr[i])
        print(i+c)
        df_widths.loc[i+c, ('neuron')]= i
        df_widths.loc[i+c, ('width')] = width
        df_widths.loc[i+c, ('epoch')]= epoch
        plt.plot(autocorr[i])
        plt.plot(pos, number, '*')
        plt.title (width, fontsize=8)
    plt.suptitle(epoch)
    plt.show()
    plt.savefig(dir2save_plots + '/autocorrs_' + epoch + '.png')
    c+=len(spikes.keys())

#Plot widths per neuron
plt.figure()
plt.plot(df_widths[df_widths['epoch']=='wake']['width'].values, '*')
plt.axhline(df_widths[df_widths['epoch']=='wake']['width'].mean(), color='k', linestyle='dashed', linewidth=1)
plt.plot(df_widths[df_widths['epoch']=='rem']['width'].values, '+')
plt.axhline(df_widths[df_widths['epoch']=='rem']['width'].mean(), color='k', linestyle='dashed', linewidth=1)
plt.xlabel('neuron')
plt.ylabel('width')
plt.show    

#plot a histogram of the widths
plt.figure()
df_widths[df_widths['epoch']=='wake']['width'].hist( alpha=0.5)
plt.axvline(df_widths[df_widths['epoch']=='wake']['width'].mean(), color='blue', linestyle='dashed', linewidth=1)
df_widths[df_widths['epoch']=='rem']['width'].hist(alpha=0.5)
plt.axvline(df_widths[df_widths['epoch']=='rem']['width'].mean(), color='orange', linestyle='dashed', linewidth=1)
plt.xlabel('width')
plt.ylabel('frequency')
plt.legend(['wake','rem'])
plt.show()
#


#Boxplots
plt.figure()
sns.boxplot(x='epoch', y='width',data=df_widths, whis=[0, 100], width=.6,palette='vlag')
sns.stripplot(x='epoch', y='width', data=df_widths,
              size=4, color=".3", linewidth=0)
plt.title(session)
plt.show()

###############################################################################
# Save data

for string, Object in zip(["df_autocorr_widths"], [df_widths]):
    with open (data_directory_load +'/'+ string + '.pickle', 'wb') as handle:
        pickle.dump(Object,handle, protocol = 4)