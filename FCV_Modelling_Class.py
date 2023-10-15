#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d

from My_plot import plot_base

class FCV_model():
    """
    Modelling of an FCV from advisor.
    """
    def __init__(self):
        ## Vehicle parameters
        self.veh_whl_radius = 0.282
        self.veh_mass = 1380
        self.veh_rrc  = 0.009
        self.veh_air_density = 1.2
        self.veh_FA = 2.0
        self.veh_CD = 0.335
        self.veh_gravity = 9.81
        self.veh_fd_ratio = 6.67
        self.eff_dc_dc = 0.94
        self.eff_dc_ac = 0.95
        self.eff_fd = 0.90
        
        ## Fuel cell parameters
        self.fc_pwr_min, self.fc_pwr_max = 0, 50*1000
        self.fc_pwr_map = np.array([0, 2, 5, 7.5, 10, 20, 30, 40, 50]) * 1000 # % kW (net) including parasitic losses
        self.fc_eff_map = np.array([10, 33, 49.2, 53.3, 55.9, 59.6, 59.1, 56.2, 50.8]) / 100 # % efficiency indexed by fc_pwr
        self.fc_fuel_map = np.array([0.012, 0.05, 0.085, 0.117, 0.149, 0.280, 0.423, 0.594, 0.821]) # fuel use map (g/s)
        fc_fuel_lhv = 120.0*1000 # (J/g), lower heating value of the fuel
        self.fc_fuel_map2 = self.fc_pwr_map * (1. / self.fc_eff_map) / fc_fuel_lhv # fuel consumption map (g/s)
        self.fc_eff_func = interp1d(self.fc_pwr_map, self.fc_eff_map, kind = 'linear', fill_value = 'extrapolate')
        self.fc_fuel_func = interp1d(self.fc_pwr_map, self.fc_fuel_map, kind = 'linear', fill_value = 'extrapolate')
        self.fc_fuel_func2 = interp1d(self.fc_pwr_map, self.fc_fuel_map2, kind = 'linear', fill_value = 'extrapolate')
        
        ## Motor parameters
        # efficiency map indexed vertically by mc_map_spd and horizontally by mc_map_trq
        self.mc_map_spd = np.arange(0, 10001, 1000) * (2 * np.pi) / 60 # motor speed list (rad/s)
        self.mc_map_trq = np.arange(-200, 201, 20) * 4.448 / 3.281 # motor torque list (Nm)
        self.mc_eff_map = np.array([
        [0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7],
        [0.78,0.78,0.79,0.8,0.81,0.82,0.82,0.82,0.81,0.77,0.7,0.77,0.81,0.82,0.82,0.82,0.81,0.8,0.79,0.78,0.78],
        [0.85,0.86,0.86,0.86,0.87,0.88,0.87,0.86,0.85,0.82,0.7,0.82,0.85,0.86,0.87,0.88,0.87,0.86,0.86,0.86,0.85],
        [0.86,0.87,0.88,0.89,0.9,0.9,0.9,0.9,0.89,0.87,0.7,0.87,0.89,0.9,0.9,0.9,0.9,0.89,0.88,0.87,0.86],
        [0.81,0.82,0.85,0.87,0.88,0.9,0.91,0.91,0.91,0.88,0.7,0.88,0.91,0.91,0.91,0.9,0.88,0.87,0.85,0.82,0.81],
        [0.82,0.82,0.82,0.82,0.85,0.87,0.9,0.91,0.91,0.89,0.7,0.89,0.91,0.91,0.9,0.87,0.85,0.82,0.82,0.82,0.82],
        [0.79,0.79,0.79,0.78,0.79,0.82,0.86,0.9,0.91,0.9,0.7,0.9,0.91,0.9,0.86,0.82,0.79,0.78,0.79,0.79,0.79],
        [0.78,0.78,0.78,0.78,0.78,0.78,0.8,0.88,0.91,0.91,0.7,0.91,0.91,0.88,0.8,0.78,0.78,0.78,0.78,0.78,0.78],
        [0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.8,0.9,0.92,0.7,0.92,0.9,0.8,0.78,0.78,0.78,0.78,0.78,0.78,0.78],
        [0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.88,0.92,0.7,0.92,0.88,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78],
        [0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.8,0.92,0.7,0.92,0.8,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78],
        ])
        # max torque curve of the motor indexed by mc_map_spd
        self.mc_max_trq = np.array([200, 200, 200, 175.2, 131.4, 105.1, 87.6, 75.1, 65.7, 58.4, 52.5]) * 4.448/3.281 # (N*m)
        self.mc_max_gen_trq = -1 * np.array([200, 200, 200, 175.2, 131.4, 105.1, 87.6, 75.1, 65.7, 58.4, 52.5]) * 4.448/3.281 # (N*m), estimate
        # motor efficiency & maximum torque
        self.mc_eff_func = interp2d(self.mc_map_trq, self.mc_map_spd, self.mc_eff_map)
        self.mc_max_trq_func = interp1d(self.mc_map_spd, self.mc_max_trq, kind = 'linear', fill_value = 'extrapolate')
        self.mc_max_gen_trq_func = interp1d(self.mc_map_spd, self.mc_max_gen_trq, kind = 'linear', fill_value = 'extrapolate')

        ## Battery parameters
        Num_cell = 25
        self.ess_Q = 26 * 3600 # coulombs, battery package capacity
        # resistance and OCV list
        self.ess_soc_map = np.arange(0, 1.01, 0.1)
        # module's resistance to being discharged, indexed by ess_soc
        self.ess_r_dis_map = np.array([40.7, 37.0, 33.8, 26.9, 19.3, 15.1, 13.1, 12.3, 11.7, 11.8, 12.2]) / 1000 * Num_cell # (ohm)
        # module's resistance to being charged, indexed by ess_soc
        self.ess_r_chg_map = np.array([31.6, 29.8, 29.5, 28.7, 28.0, 26.9, 23.1, 25.0, 26.1, 28.8, 47.2]) / 1000 * Num_cell # (ohm)
        # module's open-circuit (a.k.a. no-load) voltage, indexed by ess_soc
        self.ess_voc_map = np.array([11.70, 11.85, 11.96, 12.11, 12.26, 12.37, 12.48, 12.59, 12.67, 12.78, 12.89]) * Num_cell # (V)
        # resistance and OCV
        self.ess_voc_func = interp1d(self.ess_soc_map, self.ess_voc_map, kind = 'linear', fill_value = 'extrapolate')
        self.ess_r_dis_func = interp1d(self.ess_soc_map, self.ess_r_dis_map, kind = 'linear', fill_value = 'extrapolate')
        self.ess_r_chg_func = interp1d(self.ess_soc_map, self.ess_r_chg_map, kind = 'linear', fill_value = 'extrapolate')  
        #Battery limitations
        self.ess_min_volts = 9.5 * Num_cell
        self.ess_max_volts = 16.5 * Num_cell

    def run_Pdmd(self, veh_spd, veh_acc):        
        ## Longitudinal force balance equation
        # Wheel speed (rad/s)
        w_whl = veh_spd / self.veh_whl_radius
        # Wheel torque (Nm)
        F_roll = self.veh_mass * self.veh_gravity * self.veh_rrc * (veh_spd > 0)
        F_drag = 0.5 * self.veh_air_density * self.veh_FA * self.veh_CD * (veh_spd ** 2)
        F_acc = self.veh_mass * veh_acc
        T_whl = self.veh_whl_radius * (F_acc + F_roll + F_drag)

        ## Calculate motor speed, torque and efficiency
        mc_spd = w_whl * self.veh_fd_ratio
        mc_trq = T_whl * self.eff_fd**(-np.sign((T_whl))) / self.veh_fd_ratio
        mc_eff = (mc_spd == 0) + (mc_spd != 0) * self.mc_eff_func(mc_trq, mc_spd)
        if mc_eff.ndim == 2:
            mc_eff = np.diag(mc_eff)
        inf_mc = (np.isnan(mc_eff)) + (mc_trq < 0) * (mc_trq < self.mc_max_gen_trq_func(mc_spd)) +                (mc_trq >= 0) * (mc_trq > self.mc_max_trq_func(mc_spd))

        mc_outpwr = mc_trq * mc_spd
        mc_inpwr =  mc_outpwr * ((mc_eff*self.eff_dc_ac)**(-np.sign((mc_outpwr))))

        inf_tot = inf_mc != 0

        ## Output
        out = {}
        out['P_dem_m'] = mc_outpwr
        out['P_dem_e'] = mc_inpwr
        out['Mot_spd'] = mc_spd
        out['Mot_trq'] = mc_trq
        out['Mot_pwr'] = mc_outpwr
        out['Mot_eta'] = mc_eff
        out['Inf_tot'] = inf_tot
        
        for key,value in out.items(): # unified output
            if type(value) is not np.ndarray:
                out[key] = np.array([value,])
            elif value.shape is ():
                out[key] = np.array([value+0,])
        
        return  out


    def run(self, veh_spd, veh_acc, ess_soc, fc_pwr, output_soc_f_only=False, dt=1.0):        
        ## Longitudinal force balance equation
        # Wheel speed (rad/s)
        w_whl = veh_spd / self.veh_whl_radius
        # Wheel torque (Nm)
        # F_roll = self.veh_mass * self.veh_gravity * self.veh_rrc * (1 if veh_spd > 0 else 0)
        F_roll = self.veh_mass * self.veh_gravity * self.veh_rrc * (veh_spd > 0)
        F_drag = 0.5 * self.veh_air_density * self.veh_FA * self.veh_CD * (veh_spd ** 2)
        F_acc = self.veh_mass * veh_acc
        T_whl = self.veh_whl_radius * (F_acc + F_roll + F_drag)

        ## Calculate fuel cell fuel consumption
        fc_pwr = np.clip(fc_pwr, self.fc_pwr_min, self.fc_pwr_max) * (T_whl >= 0)
        fc_eff = self.fc_eff_func(fc_pwr)
        fc_fuel = self.fc_fuel_func2(fc_pwr) * dt
        
        inf_fc = (fc_pwr < self.fc_pwr_min) + (fc_pwr > self.fc_pwr_max)

        ## Calculate motor speed, torque and efficiency
        mc_spd = w_whl * self.veh_fd_ratio
        mc_trq = T_whl * self.eff_fd**(-np.sign((T_whl))) / self.veh_fd_ratio
        mc_eff = (mc_spd == 0) + (mc_spd != 0) * self.mc_eff_func(mc_trq, mc_spd)
        inf_mc = (np.isnan(mc_eff)) + (mc_trq < 0) * (mc_trq < self.mc_max_gen_trq_func(mc_spd)) +                (mc_trq >= 0) * (mc_trq > self.mc_max_trq_func(mc_spd))

        mc_outpwr = mc_trq * mc_spd
        mc_inpwr =  mc_outpwr * ((mc_eff*self.eff_dc_ac)**(-np.sign((mc_outpwr))))

        ## Calculate battery voltage, resistance, current, power and efficiency
        ess_pwr = mc_inpwr - fc_pwr * self.eff_dc_dc
        ess_eff = (ess_pwr > 0) + (ess_pwr <= 0) * 0.9
        ess_voc = self.ess_voc_func(ess_soc)
        ess_r_int = (ess_pwr > 0) * self.ess_r_dis_func(ess_soc) + (ess_pwr <= 0) * self.ess_r_chg_func(ess_soc)
        ess_cur = ess_eff * (ess_voc - np.sqrt(ess_voc ** 2 - 4 * ess_r_int * ess_pwr)) / (2*ess_r_int)
        ess_volt = ess_voc - ess_cur * ess_r_int

        inf_ess = (ess_voc ** 2 < 4 * ess_r_int * ess_pwr) + (ess_volt < self.ess_min_volts) + (ess_volt > self.ess_max_volts)

        ## Calculate next state of charge (SoC)
        ess_soc_new = ess_soc - ess_cur * dt / self.ess_Q
        ess_soc_new = np.clip((np.conjugate(ess_soc_new) + ess_soc_new) / 2, 0, 1)

        inf_tot = inf_fc + inf_mc + inf_ess != 0

        ## Output
        out = {}
        if output_soc_f_only:
            out['Bat_soc'] = ess_soc_new
        else:
            out['P_dem_m'] = mc_outpwr
            out['P_dem_e'] = mc_inpwr
            out['Mot_spd'] = mc_spd
            out['Mot_trq'] = mc_trq
            out['Mot_pwr'] = mc_outpwr
            out['Mot_eta'] = mc_eff
            out['Bat_soc'] = ess_soc_new
            out['Bat_vol'] = ess_volt
            out['Bat_cur'] = ess_cur
            out['Bat_pwr'] = ess_pwr
            out['FCS_pwr'] = fc_pwr
            out['FCS_eta'] = fc_eff
            out['FC_fuel'] = fc_fuel
            out['Inf_tot'] = inf_tot
        
        for key,value in out.items(): # unified output
            if type(value) is not np.ndarray:
                out[key] = np.array([value,])
            elif value.shape is ():
                out[key] = np.array([value+0,])
        
        return  out

        
    def plot_map_fcs(self,):
        plot_base()
        
        fig, ax1, = plt.subplots(1, 1, sharex=False, sharey=False, facecolor='white', figsize=(6,4), dpi=300)
        fig.subplots_adjust(left=0.11, right=0.88, top=0.99, bottom=0.13,hspace=0.05,wspace=0.1)

        line1, = ax1.plot(self.fc_pwr_map / 1000, self.fc_eff_map * 100, 'r', lw = 2)
        ax2 = ax1.twinx()
        line2, = ax2.plot(self.fc_pwr_map / 1000, self.fc_fuel_map2, 'g', lw = 2)

        # ax1.set_xlabel('Net power [kW]')
        # ax1.set_ylabel('Efficiency [%]')
        # ax2.set_ylabel('Hydrogen consumption rate [g/s]')
        
        ax1.set_xlabel('$P_{fcs}$ [kW]')
        ax1.set_ylabel('${\eta }_{fcs}$ [%]')
        ax2.set_ylabel('${\dot{m}}_{fcs}$ [g/s]')

        ax1.set_xlim((0, 50))
        ax1.set_ylim((0, 65))
        ax2.set_ylim((0, 0.9))

        plot_base().double_y_color(ax1,ax2,line1,line2)

        ax1.grid(linestyle='-')
        
        return fig, ax1
    
    def plot_map_mot(self,):
        plot_base()
        
        # Set to None if the torque exceeds the maximum torque
        T,_ = np.meshgrid(self.mc_map_trq, self.mc_map_spd)
        mc_eff_map = np.copy(self.mc_eff_map)
        for ii in range(len(self.mc_map_spd)):
            spd = self.mc_map_spd[ii]
            trq_pos = self.mc_max_trq_func(spd)
            trq_neg = self.mc_max_gen_trq_func(spd)
            none_idx = np.union1d(np.where(T[ii,:]>trq_pos),np.where(T[ii,:]<trq_neg))
            mc_eff_map[ii,none_idx] = None

        mot_spd_list = self.mc_map_spd / ((2 * np.pi) / 60)

        fig, ax1, = plt.subplots(1, 1, sharex=False, sharey=False, facecolor='white', figsize=(6,4), dpi=300)
        fig.subplots_adjust(left=0.14, right=0.94, top=0.99, bottom=0.13,hspace=0.05,wspace=0.1)

        line1 = ax1.contour(mot_spd_list, self.mc_map_trq, mc_eff_map.T, label='Efficiency')
        plt.clabel(line1, fmt='%.2f', fontsize=8)

        line2, = ax1.plot(mot_spd_list, self.mc_max_trq, 'r', label='External characteristics')
        line3, = ax1.plot(mot_spd_list, self.mc_max_gen_trq, 'r')

        # ax1.set_xlabel('Speed [r/min]')
        # ax1.set_ylabel('Torque [Nm]')
        
        ax1.set_xlabel('${{\omega }_{mot}}$ [r/min]')
        ax1.set_ylabel('${{T}_{mot}}$ [Nm]')
        
        ax1.set_xlim(0, 10000)
        ax1.set_ylim(-280, 280)
        ax1.grid(linestyle='-')

        ax1.legend(handles=[line2,], loc = 0, framealpha=1)
        
        return fig, ax1
    
    def plot_map_bat(self,):
        plot_base()
        
        fig, ax1, = plt.subplots(1, 1, sharex=False, sharey=False, facecolor='white', figsize=(6,4), dpi=300)
        fig.subplots_adjust(left=0.11, right=0.88, top=0.99, bottom=0.13,hspace=0.05,wspace=0.1)

        line1, = ax1.plot(self.ess_soc_map,self.ess_r_dis_map, 'r', lw = 2, label='Discharging')
        line12, = ax1.plot(self.ess_soc_map,self.ess_r_chg_map, 'r--', lw = 2, label='Charging')
        ax2 = ax1.twinx()
        line2, = ax2.plot(self.ess_soc_map,self.ess_voc_map, 'g', lw = 2, label='Open-circuit voltage [V]')

        # ax1.set_xlabel('Battery SOC [-]')
        # ax1.set_ylabel('Internal resistance [$\Omega $]')
        # ax2.set_ylabel('Open-circuit voltage [V]')
        
        ax1.set_xlabel('$SoC$ [-]')
        ax1.set_ylabel('$R_0$ [$\Omega $]')
        ax2.set_ylabel('$V_{OC}$ [V]')

        ax1.set_xlim((0, 1))
        ax1.legend(handles=[line1,line12,], loc = 0, framealpha=1)

        plot_base().double_y_color(ax1,ax2,line1,line2)

        ax1.grid(linestyle='-')
        
        return fig, ax1
