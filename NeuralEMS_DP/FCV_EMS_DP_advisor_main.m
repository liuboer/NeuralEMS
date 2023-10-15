% DP-based EMS of an FCV.
clear;clc;close all

Flag = 3; % 0-train, 1-test_cs_60, 2-test_cs_90, 3-test_cd_70_50
savefig_bool = 0;

par.u_last = 0;
par.gpu_bool = 0;
par.dt = 1.0;

cycle_info_cell = load('FCV_EMS_DP_advisor_results\cycle_info_cell');
cycles_train = cycle_info_cell.cycle_info_cell([2,3,4,5,14,17,19,21,24,26,27,29,30,31,33,35],1);
cycles_test_cs = cycle_info_cell.cycle_info_cell([6,11,16,22],1);
cycles_test_cd = {'C_CFN','RealDC'};

if Flag == 0
    cycles = cycles_train;
    SOC_0 = 0.60;
    SOC_T_list = 0.58:0.01:0.62;
    soc_min = 0.55;
    soc_max = 0.65;
elseif Flag == 1
    cycles = cycles_test_cs;
    SOC_0 = 0.60;
    SOC_T_list = 0.60;
    soc_min = 0.55;
    soc_max = 0.65;
elseif Flag == 2
    cycles = cycles_test_cs;
    SOC_0 = 0.90;
    SOC_T_list = 0.90;
    soc_min = 0.85;
    soc_max = 0.95;
elseif Flag == 3
    cycles = cycles_test_cd;
    SOC_0 = 0.70;
    SOC_T_list = 0.50;
    soc_min = 0.45;
    soc_max = 0.75;
end

Results_all_m = cell(length(SOC_T_list),1);
train_data_m = cell(length(SOC_T_list),length(cycles));
dp_res_m = cell(length(SOC_T_list),length(cycles));

for ji = 1:length(SOC_T_list)
for ij = 1:length(cycles)

clear veh_spd;

veh_spd = load(cycles{ij}); % nx1, km/h

name_cell = fieldnames(veh_spd);
veh_spd = getfield(veh_spd,name_cell{1}) / 3.6; % m/s
veh_spd = reshape(veh_spd, 1,[]); % 1xn, m/s
veh_acc = [diff(veh_spd),0]; % end with 0
N = length(veh_spd);

SOC_T = SOC_T_list(ji);
deta  = 0.0001;
fc_pwr_min = 0;
fc_pwr_max = 50*1000;

clear grd

grd.Nx{1}    = 10000 + 1;    % state variable, SOC
grd.X0{1}    = SOC_0;
grd.Xn{1}.hi = soc_max; 
grd.Xn{1}.lo = soc_min;
grd.XN{1}.hi = SOC_T+deta;
grd.XN{1}.lo = SOC_T;

grd.Nu{1}    = 50 + 1;   % control variable, P_fcs
grd.Un{1}.hi = fc_pwr_max;
grd.Un{1}.lo = fc_pwr_min;

if par.gpu_bool
grd.Nu{2}    = 1;   % control variable, none
grd.Un{2}.hi = 0;
grd.Un{2}.lo = 0;
end

% define problem
clear prb
prb.W{1} = veh_spd; 
prb.W{2} = veh_acc; 
prb.Ts = 1;
prb.N  = (N-1)*1/prb.Ts + 1;

% set options
options = dpm();
options.MyInf = 1e10;
options.BoundaryMethod = 'Line'; % also possible: 'none' or 'LevelSet';
if strcmp(options.BoundaryMethod,'Line')
    %these options are only needed if 'Line' is used
    options.Iter = 100;
    options.Tol = 1e-10;
    options.FixedGrid = 0;
end

tic
if par.gpu_bool
[res, dyn] = dpm_gpu(@FCV_EMS_DP_advisor_model,par,grd,prb,options);
keys = fieldnames(res);
for i = 1:length(keys)
    key = keys(i);
    key = key{1};
    res.(key) = gather(res.(key));
end

for i = 1:length(res.X)
    res.X{i} = gather(res.X{i});
end

else
    
[res, dyn] = dpm(@FCV_EMS_DP_advisor_model,par,grd,prb,options);
end
%%
TOC = toc;
SOC = res.X{1,1};
H2 = sum(res.FC_fuel);
fprintf('SOC_0: %.4f , SOC_T: %.4f , H2 Consumption: %.4f g.\n', SOC_0, SOC(end), H2);

Results_all_m{ji,1}(ij,:) = [TOC,SOC(end),H2];

%% π¶¬ ∑÷≈‰
figure(ij)
subplot(211)
yyaxis left
plot(veh_spd);
ylabel('Speed [m/s]')
yyaxis right
plot(SOC(2:end));grid on
xlabel('Time [-]')
ylabel('SOC [-]')
xlim([0 N])

subplot(212)
plot(res.FCS_pwr/1000, 'r-');hold on
plot(res.P_dem_e/1000, 'k-');hold off;grid on;
xlabel('Time [s]')
ylabel('power [kW]')
legend('FCS_{pwr}','P_{dem}');
xlim([0 N])

if SOC_0==0.6 && SOC_T==0.6 && savefig_bool
    saveas(gcf,['FCV_EMS_DP_advisor_results/Figure/' cycles{ij} '.png'])
end
%% 

try
    train_data_m{ji,ij} = [veh_spd;veh_acc;SOC(1:end-1);SOC_T*ones(size(veh_spd));...
        cumsum(veh_spd)./sum(veh_spd);res.P_dem_m/1000;res.FCS_pwr/1000];
    dp_res_m{ji,ij} = res;
catch
    train_data_m{ji,ij} = [];
    dp_res_m{ji,ij} = [];
end

end

end

%% save results

if Flag == 0
    res_mxSoC_T_train.train_data_m = train_data_m;
    res_mxSoC_T_train.Results_all_m = Results_all_m;
    res_mxSoC_T_train.cycles = cycles;
    res_mxSoC_T_train.SOC_T_list = SOC_T_list;
    res_mxSoC_T_train.grd = grd;
    res_mxSoC_T_train.options = options;
    res_mxSoC_T_train.dp_res_m = dp_res_m;
    
%     save('FCV_EMS_DP_advisor_results/res_mxSoC_T_train_20220628','res_mxSoC_T_train')
else
    res_mxSoC_T_test.train_data_m = train_data_m;
    res_mxSoC_T_test.Results_all_m = Results_all_m;
    res_mxSoC_T_test.cycles = cycles;
    res_mxSoC_T_test.SOC_T_list = SOC_T_list;
    res_mxSoC_T_test.grd = grd;
    res_mxSoC_T_test.options = options;
    res_mxSoC_T_test.dp_res_m = dp_res_m;
    if Flag == 1
    save('FCV_EMS_DP_advisor_results/res_mxSoC_T_test_20220628_CS@60','res_mxSoC_T_test')
    elseif  Flag == 2
    save('FCV_EMS_DP_advisor_results/res_mxSoC_T_test_20220628_CS@90','res_mxSoC_T_test')
    elseif Flag == 3
    save('FCV_EMS_DP_advisor_results/res_mxSoC_T_test_20220628_CD@70-50','res_mxSoC_T_test')
    end
end