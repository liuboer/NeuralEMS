clear all; clc

cycle_list = dir('D:\Data_Programs\BoLiu\MatlabTools\StandardDrivingCycles\StandardCycle_kph_column\*.mat');
cycle_info = struct([]);
for i = 1:length(cycle_list)
cycle_list_name_i = cycle_list(i).name;

cycle = load(cycle_list_name_i);

% spd_kph_column = eval([cycle_list_name_i(1:end-4),'']);
cycle_name = fieldnames(cycle);
spd_kph_column = getfield(cycle,cycle_name{1});

cycle_info(i).name = cycle_list_name_i(1:end-4);
cycle_info(i).len_t = length(spd_kph_column);
cycle_info(i).len_d = sum(spd_kph_column) / 3.6;
end

% save to EXCEL
cycle_info_cell = permute(squeeze(struct2cell(cycle_info)),[2,1]);
% save('cycle_info_cell.mat','cycle_info_cell')
% xlswrite('cycle_info_raw.xlsx',cycle_info_cell,'sheet1');