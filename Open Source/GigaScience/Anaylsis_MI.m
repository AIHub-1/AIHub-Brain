%% Anaylsis_MI
clear all; clc; close all;
%% initialization
DATADIR = 'WHERE\IS\DATA';
%% MI
MIDATA = 'EEG_MI.mat';
STRUCTINFO = {'EEG_MI_train', 'EEG_MI_test'};
SESSIONS = {'session1', 'session2'};
TOTAL_SUBJECTS = 54;
FS=100;

%% PERFORMANCE PARAMETERS
params = { 'task',{'mi_off','mi_on'}; ...
    'channel_index', [8:11 13:15 18:21 33:41]; ...
    'band', [8 30]; ...
    'time_interval', [1000 3500]; ...
    'CSPFilter', 2; ...
    };

% for MI_cv
Niteration = 10; 
% for CSSP
tau=[0.01:0.01:0.15]*1000; 
% for FBCSP
filterbank = [4 8;8 12;12 16;16 20;20 24;24 28;28 32;32 36;36 40]; 
NUMfeat = 4; 
% for BSSFO
bssfo_param = {'init_band', [4 40]; ...
    'numBands', 30; ...
    'numIteration', 10; ...
    'mu_band', [7 15]; ...
    'beta_band', [14 30]; ...
    };
%% validation
for sessNum = 1:length(SESSIONS)
    session = SESSIONS{sessNum};
    fprintf('\n%s validation\n',session);
    for subNum = 1:TOTAL_SUBJECTS
        subject = sprintf('s%d',subNum);
        fprintf('LOAD %s ...\n',subject);
        
        data = importdata(fullfile(DATADIR,session,subject,MIDATA));
        
        CNT{1} = prep_resample(data.(STRUCTINFO{1}), FS,{'Nr', 0});
        CNT{2} = prep_resample(data.(STRUCTINFO{2}), FS,{'Nr', 0});        

        ACC.MI_cv(subNum,sessNum) = mi_performance(CNT,params,Niteration);
        ACC.MI_off2on(subNum,sessNum) = mi_performance_off2on(CNT,params);     
        ACC.MI_CSSP(subNum,sessNum) = cssp_off2on(CNT,params,tau);
        ACC.MI_FBCSP(subNum,sessNum) = fbcsp_off2on(CNT,params,filterbank,NUMfeat);
        ACC.MI_BSSFO(subNum,sessNum) = bssfo_off2on(CNT,params,bssfo_param);
        
        fprintf('CSP_crsval = %f\n',ACC.MI_cv(subNum,sessNum));
        fprintf('CSP = %f\n',ACC.MI_off2on(subNum,sessNum));
        fprintf('CSSP = %f\n',ACC.MI_CSSP(subNum,sessNum));
        fprintf('FBCSP = %f\n',ACC.MI_FBCSP(subNum,sessNum));
        fprintf('BSSFO = %f\n',ACC.MI_BSSFO(subNum,sessNum));
        clear CNT
    end
end
