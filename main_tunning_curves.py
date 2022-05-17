#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 10:37:24 2021

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

tuningcurve = tuning_curves[0]

plt.figure()
plt.plot(array)
plt.show()
def width_tun(tuningcurve, limit_righthalf = 0.1775, plotT=False):
    array = tuningcurve.values
    nbins = len(array)
    posmax = np.where((array==array.max())==True)[0][0]
    lim = nbins - int(nbins/3)
    #find the position in the array of the middle point 
    x = int(nbins/2)
    if posmax > lim:
        array = np.append (array[x:], array[:x])
        posmax = np.where((array==array.max())==True)[0][0]  
    # if pos_max < lim_low or pos_max > lim_high: 
    #     print("flag")
    #     array = np.append (array[x:], array[:x])
    #     
    peak2trough = array.max() - array.min()
    half_val = peak2trough/2
    threshold = array.min() + half_val
    array2search = array[posmax:posmax+20]
    array2search = np.flip(array2search)
    for i in array2search:
        if i>=threshold:
            number=i
            break
    positionWidthINarray = np.where((array==number)==True)[0][0] #right extreme of the half
    halfwidthNbins = positionWidthINarray - posmax
    widthNradians = 2*(((2*np.pi)/len(array))*halfwidthNbins)
    
    if plotT == True:
        props = dict(boxstyle='round', facecolor='linen', edgecolor = 'linen', alpha=0.5)
        plt.figure()
        plt.plot(array)
        plt.vlines(posmax, array.min(), array.max())
        plt.xlabel('bins')
        plt.ylabel("firing rate")
        plt.plot(positionWidthINarray, number, '*')
        plt.annotate("width = " + str(halfwidthNbins*2), xytext=(50, 1.3), xy=(positionWidthINarray, number), 
              arrowprops=dict(arrowstyle="->", color='black'), bbox = props, fontsize=8)
        plt.title('half width of the tuning curve')
        plt.show()
        
    return array, posmax, positionWidthINarray, number, halfwidthNbins, widthNradians

##############################################################################
session = 'B0201-210923'
data_directory_load = '/Users/vite/OneDrive - McGill University/PeyracheLab/Data/MotorHead/B0201/B0201-210923/data'
dir2save_plots = data_directory_load + '/plots'

tuning_curves = pickle.load(open(data_directory_load + '/' + 'tuning_curves.pickle', 'rb'))

neurons_selected = [*range(19)]
df_widthsT = pd.DataFrame(columns = ["neuron", "width"])
df_widthsT['neuron'] = neurons_selected
props = dict(boxstyle='round', facecolor='linen', edgecolor = 'linen', alpha=0.5)
fig = plt.figure()
for i,t in enumerate(neurons_selected): 
    ax = fig.add_subplot(4, 5, i+1)
    array, posmax, positionWidthINarray, number, halfwidthNbins, widthNradians = width_tun(tuning_curves[t])   
    df_widthsT.loc[i, ('width')]=widthNradians
    ax.plot(array)
    ax.axvline(posmax, 0, 1)
    ax.plot(positionWidthINarray, number, '*')
    ax.annotate("width = " + str(halfwidthNbins*2), xytext=(50, 1.3), xy=(positionWidthINarray, number), 
          arrowprops=dict(arrowstyle="->", color='black'), bbox = props, fontsize=8)
    ax.set_title(t)
plt.show()
plt.savefig(dir2save_plots + '/' + 'HalfWidth_Tuning_curves')

###############################################################################
# Save data
for string, Object in zip(["df_widths_tuningcurvers"], [df_widthsT]):
    with open (data_directory_load +'/'+ string + '.pickle', 'wb') as handle:
        pickle.dump(Object,handle, protocol = 4)