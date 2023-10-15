clear; clc

load res_mxSoC_T_train_20220628.mat

% train_data: 7xT in each cell
% row: v、a、SOC、target SOC、s_、p_mot_m(kW)、P_fcs(kW)
%% multi-step input, one-step output

train_data_m = res_mxSoC_T_train.train_data_m(:);
train_data = reshape(train_data_m,1,[]);

num_step_backward = 1;
num_step_forward = 1;
train_data_new = cell(size(train_data));
for i = 1:length(train_data)
    for j = 1:length(train_data{i})-num_step_backward-num_step_forward
       train_data_new{i}(:,j) =  [reshape(train_data{i}([1:5,7],j+num_step_backward),[],1);...
           reshape(train_data{i}(6,j:j+num_step_backward+num_step_forward),[],1)];
    end
end

if num_step_backward == 1 && num_step_forward == 1
    train_data_matrix_11 = cell2mat(train_data_new);
    save('train_data_matrix_11','train_data_matrix_11');
elseif num_step_backward == 0 && num_step_forward == 0
    train_data_matrix_00 = cell2mat(train_data_new);
    save('train_data_matrix_00','train_data_matrix_00');
end