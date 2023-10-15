#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt


class plot_base:
    def __init__(self,):
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rc('font', family='Times New Roman', weight='normal', size=16)
        plt.rc('mathtext', fontset='stix')
        
    def double_y_color(self,ax1,ax2,line1,line2):
        ax1.yaxis.label.set_color(line1.get_color())
        ax2.yaxis.label.set_color(line2.get_color())
        ax1.tick_params(axis = 'y', colors = line1.get_color())
        ax2.tick_params(axis = 'y', colors = line2.get_color())
        
    def axis_tick(self,ax1,axis_fontsize=16):
        ax1.tick_params(labelsize=axis_fontsize)
        labels = ax1.get_xticklabels() + ax1.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        
    def plot_base(self,x,y):
        fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, facecolor='white', figsize=(8,6), dpi=100)
        ax.plot(x,y)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        return fig, ax
        
        
class plot_EMS(plot_base):
    def __init__(self,):
        super(plot_EMS,self).__init__()
        
    def plot_EMS_1(self,veh_spd,res):
        soc_vec = res['Bat_soc']
        Dmd_pwr = res['P_dem_e'] / 1000
        Fcs_pwr = res['FCS_pwr'] / 1000
        Bat_pwr = res['Bat_pwr'] / 1000
        len_x = veh_spd.size
        tt = range(len_x)
        
        fig, [ax1,ax2,] = plt.subplots(2, 1, sharex=True, sharey=False, facecolor='white', figsize=(8,6), dpi=100)

        line11, = ax1.plot(tt,veh_spd, 'r', label='Speed')
        ax12 = ax1.twinx()
        line12, = ax12.plot(tt,soc_vec,'k', label='SOC')

        ax1.set_ylabel('Speed [m/s]')
        ax12.set_ylabel('SOC [-]')
        ax1.legend(loc='best', handles = [line11,line12])
        self.double_y_color(ax1,ax12,line11,line12)

        line21, = ax2.plot(tt,Fcs_pwr, 'r', label='Fcs_pwr')
        line22, = ax2.plot(tt,Bat_pwr, 'b-', label='Bat_pwr')
        line23, = ax2.plot(tt,Dmd_pwr, 'k-', label='Dmd_pwr')

        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Power [kW]')
        ax2.set_xlim((0, len_x))
        ax2.legend(loc='best', handles = [line21,line22,line23])

        return fig, [ax1,ax2,]
        