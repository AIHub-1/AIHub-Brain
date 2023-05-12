% using OpenBMI
% offline signal plotting

%%
clc; clear all; close all;

%% data load
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
marker={'1','1';'2','2';'3','3'};
fs=500;   %chan_resampling

%% data path
filepath='C:\Users\cvpr\Documents\data';


%% eeg - ear
% ear EEG - eeg file
filename='record-[2018.01.18-15.46.53]';
file=fullfile(filepath,filename);
[ear_EEG.data, ear_EEG.marker, ear_EEG.info]=Load_EEG(file,{'device','brainVision';'marker',marker;'fs',fs});
ear_cnt=opt_eegStruct({ear_EEG.data, ear_EEG.marker, ear_EEG.info}, field);

%% Preprocessing
ear_cnt=prep_filter(ear_cnt, {'frequency', [5 60]});
ear_smt=prep_segmentation(ear_cnt, {'interval', [0 5000]});

%% eeg - cap
% cap EEG data
cap_file=sprintf('cap_%s',filename);
file=fullfile(filepath,cap_file);
[cap_EEG.data, cap_EEG.marker, cap_EEG.info]=Load_EEG(file,{'device','brainVision';'marker',marker;'fs',fs});
cap_cnt=opt_eegStruct({cap_EEG.data, cap_EEG.marker, cap_EEG.info}, field);

%% Preprocessing
cap_cnt=prep_filter(cap_cnt, {'frequency', [5 60]});
cap_smt=prep_segmentation(cap_cnt, {'interval', [0 5000]});
