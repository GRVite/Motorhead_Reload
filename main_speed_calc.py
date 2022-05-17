#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:23:53 2021

@author: vite
"""

session = 'B0201-210923'

widths_A = pickle.load(open(data_directory_load + '/df_autocorr_widths.pickle', 'rb'))
widths_T = pickle.load(open(data_directory_load + '/df_widths_tuningcurves.pickle', 'rb'))

speed = widths_A.drop(columns = ['width'])

wake = widths_A.groupby(['epoch']).get_group('wake')['width'].values/widths_T['width'].values.flatten()
sws =  widths_A.groupby(['epoch']).get_group('sws')['width'].values/widths_T['width'].values.flatten()
rem =  widths_A.groupby(['epoch']).get_group('rem')['width'].values/widths_T['width'].values.flatten()


speed['speed'] = np.stack([wake, sws,rem]).flatten()

for e in ['wake', 'sws', 'rem']:
    widths_A.groupby(['epoch']).get_group('wake')['width'].values/widths_T['width'].values.flatten()
plt.figure()
plt.boxplot([wake,sws,rem])
plt.show()

#Boxplots
plt.figure()
sns.boxplot(x='epoch', y='speed',data=speed, whis=[0, 100], width=.6,palette='vlag')
sns.stripplot(x='epoch', y='speed', data=speed,
              size=4, color=".3", linewidth=0)
plt.title(session)
plt.show()  