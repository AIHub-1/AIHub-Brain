clear all; close all; clc;

%%
dd = 'dir';
%% file list 
filelist={'filename'};


%%
for ff= 1:length(filelist),

    file= filelist{ff};
    opt= [];
    
    fprintf('** Processing of %s **\n', file);
    
    % load the header file
    try,
        hdr= eegfile_readBVheader([dd '\' file]);
    catch
        fprintf('%s/%s not found.\n', dd, file);
        continue;
    end
    
    % filtering with Chev filter
    Wps= [42 49]/hdr.fs*2;
    [n, Ws]= cheb2ord(Wps(1), Wps(2),3, 40);
    [filt.b, filt.a]= cheby2(n, 50, Ws);
    
    % Load channel information
    [cnt, mrk_orig]= eegfile_loadBV([dd '\' file],  ...
        'filt',filt,'clab',{'not','EMG*'});

    cnt.title= ['saved_dir' file];

    % Load mrk file, Assign the trigger information into mrk variable
    % If you want to convert another task's data, please check the trigger
    % information into Image_Arrow function.
    mrk = Imag_Arrow(mrk_orig);
    
    % Assign the channel montage information into mnt variable
    mnt = getElectrodePositions(cnt.clab);
    
    % Assign the sampling rate into fs_orig variable
    fs_orig= mrk_orig.fs;
    
    var_list= {'fs_orig',fs_orig, 'mrk_orig',mrk_orig, 'hdr',hdr};
    
    % Convert the .eeg raw data file to .mat file
    eegfile_saveMatlab(cnt.title, cnt, mrk, mnt, ...
        'channelwise',1, ...
        'format','int16', ...
        'resolution', NaN);       

end

disp('All EEG Data Converting was Done!');
