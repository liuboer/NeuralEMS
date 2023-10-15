clear;clc;close all

cycle_info_cell = load('cycle_info_cell');
cycles_test = cycle_info_cell.cycle_info_cell([6,11,22],1);

cycles = cycles_test;

C_CFN = [];

for ij = 1:length(cycles)

clear veh_spd;

veh_spd = load(cycles{ij}); % nx1, km/h
name_cell = fieldnames(veh_spd);
veh_spd = getfield(veh_spd,name_cell{1});
C_CFN = [C_CFN;veh_spd];
end

C_CFN = repmat(C_CFN,1,1);

save('D:\Data_Programs\BoLiu\MatlabTools\StandardDrivingCycles\StandardCycle_kph_column\C_CFN.mat','C_CFN')
