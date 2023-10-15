#!/usr/bin/env python
# coding: utf-8

import numpy as np

from FCV_Modelling_Class import FCV_model

class ECMS():
    def __init__(self,veh_spd,veh_acc,soc_ini = 0.6,soc_tgt = 0.6,grid_u=100): # input: scalar / vector
        self.env = FCV_model()
        self.veh_spd = np.array([veh_spd,]).flatten()
        self.veh_acc = np.array([veh_acc,]).flatten()
        self.soc_ini = soc_ini
        self.soc_tgt = soc_tgt
        
        self.spd_len = self.veh_spd.size
        self.fcs_pwr_list = np.linspace(0,50*1000,grid_u)
        
        self.LHV = 120*1000
        
    def opti_step(self,spd_t,acc_t,soc_t,k,output_all=True): # input: scalar
        out_list = self.env.run(spd_t, acc_t, soc_t, self.fcs_pwr_list)
        H = out_list['FC_fuel'] * self.LHV + k * out_list['Bat_cur'] * self.env.ess_voc_func(out_list['Bat_cur'])
        # H = out_list['FC_fuel'] * self.LHV + k * out_list['Bat_pwr']
        H_f_index = np.where(out_list['Inf_tot'] == 0)
        H_f = H[H_f_index]
        H_min = np.nanmin(H_f)
        H_min_index = np.where(H_f == H_min)[0][0]

        index_opt = H_f_index[0][H_min_index]
        
        if output_all:
            for key,value in out_list.items():
                if out_list[key].size>1:
                    out_list[key] = out_list[key][index_opt]
            return out_list
        else:
            return out_list['Bat_soc'][index_opt]
        
    def find_k_shooting(self,dsoc,k_1,k_2,max_iter,print_iter):
        jj = 1
        while jj < max_iter:
            soc = self.soc_ini
            k = (k_1 + k_2) / 2
            for i in range(self.spd_len):
                spd = self.veh_spd[i]
                acc = self.veh_acc[i]
                
                soc = self.opti_step(spd,acc,soc,k,output_all=False)

            if print_iter:
                print(f'Iter is {jj:3d}, SOC is {soc:.6f}, k is {k:.4f}.')

            if soc - self.soc_tgt > dsoc:
                k_1, k_2 = k_1, k
            elif soc - self.soc_tgt < -dsoc:
                k_1, k_2 = k, k_2
            else:
                break

            jj += 1
            
        return k
        
    def find_k_pi(self,soc, soc_vector_i, k_0, k_p, k_i):
        k = k_0 + k_p * (self.soc_tgt - soc) + k_i * np.sum(self.soc_tgt - soc_vector_i)
        return k
        
        
    def ECMS_shooting(self,dsoc = 5e-5,k_1 = 1,k_2 = 3,max_iter = 100,k_0=1.5,fixed_k=False,print_detail=False):
        soc = self.soc_ini

        res = {}
        keys = ('P_dem_m','P_dem_e','Mot_spd','Mot_trq','Mot_pwr','Mot_eta',\
                'Bat_soc','Bat_vol','Bat_cur','Bat_pwr','FCS_pwr','FCS_eta','FC_fuel','Inf_tot',)
        for key in keys:
            res[key] = np.zeros([self.spd_len,])
            
        if fixed_k:
            k = k_0
        else:
            k = self.find_k_shooting(dsoc,k_1,k_2,max_iter,print_iter=print_detail)
        
        for i in range(self.spd_len):
            spd = self.veh_spd[i]
            acc = self.veh_acc[i]
            
            res_step = self.opti_step(spd,acc,soc,k)
            soc = res_step['Bat_soc']
            
            for key in res_step.keys():
                res[key][i] = res_step[key]
                
        return res,k
    
    def ECMS_pi(self, k_0 =1.2, k_p = 100, k_i = 0.5):
        soc = self.soc_ini
        soc_vector_i = np.array([self.soc_ini,])

        res = {}
        keys = ('P_dem_m','P_dem_e','Mot_spd','Mot_trq','Mot_pwr','Mot_eta',\
                'Bat_soc','Bat_vol','Bat_cur','Bat_pwr','FCS_pwr','FCS_eta','FC_fuel','Inf_tot',)
        for key in keys:
            res[key] = np.zeros([self.spd_len,])
        
        for i in range(self.spd_len):
            spd = self.veh_spd[i]
            acc = self.veh_acc[i]
            
            k = self.find_k_pi(soc, soc_vector_i, k_0, k_p, k_i)
            res_step = self.opti_step(spd,acc,soc,k)
            soc = res_step['Bat_soc']
            soc_vector_i = np.append(soc_vector_i,soc)
            
            for key in res_step.keys():
                res[key][i] = res_step[key]

        return res

    