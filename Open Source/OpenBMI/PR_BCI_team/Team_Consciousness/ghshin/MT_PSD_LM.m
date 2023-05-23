%% MT analysis
clear; close all; clc;

addpath('E:\Users\SHIN\eeglab14_1_2b\');
eeglab;
%% 
format long g
fs = 250;
range = {[0.5 4],[4 7],[7 12],[12 15],[15 30],[30 50]};

Time = {'BN', '24'};
Trigger_LM = ["S 55", "S 56"]; %Location Memory
%% MT_textfile load
Path_M = 'H:\1. Journal\NeuroImage\SLEEP_DATA_24H\';
DirGroup_M = dir(fullfile(Path_M,'*'));
FileNamesGroup_M = {DirGroup_M.name};
FileNamesGroup_M = FileNamesGroup_M(1,3:end);
Time_M = {'BN', '24'};
% Time_M = {'BN', 'AN'};
removal_word = [".jpg"];
%% PSD analysis

for n = 1:size(FileNamesGroup_M,2)
%% LM Task (Succeful / Forgotten)
    % Learning memory
    Data_L=textscan(fopen([FileNamesGroup_M{n} '_' Time_M{1} '_learning_visuo.txt']), '%s %s %s %s');
    Data_L=[Data_L{:}];
    Hash_Table_L = str2double(erase(Data_L([2:end],4),removal_word));
    Hash_Table_L =[cellfun(@(x) sprintf('%02d',x),num2cell(Hash_Table_L),'UniformOutput',false) Data_L([2:end],3)];
    Hash_Table_L = sortrows(Hash_Table_L,1);
    
    % Recall
    for t = 1:size(Time,2)
        Data_recall=textscan(fopen([FileNamesGroup_M{n} '_' Time_M{t} '_recall_visuo.txt']), '%s %s %s %s %s %s %s');   
        Data_recall=[Data_recall{:}];
        Hash_Table_recall = str2double(erase(Data_recall([2:end],3),removal_word));
        Hash_Table_recall = [cellfun(@(x) sprintf('%02d',x),num2cell(Hash_Table_recall),'UniformOutput',false)...
            Data_recall([2:end],4) Data_recall([2:end],6) Data_recall([2:end],2)];
        Hash_Table_recall = sortrows(Hash_Table_recall,1);
%% Picture memory
        % Old/new decision   
        for i = 1:size(Hash_Table_recall,1)
            if string(Hash_Table_recall(i,2)) == 'o'
                res(i,t) = 1;
            elseif string(Hash_Table_recall(i,2)) == 'n'
                res(i,t) = 0;
            elseif isspace(string(Hash_Table_recall(i,2))) == 1 % i don't know
                res(i,t) = 2;
            end
        end
        
        % four possible reponse categories   
        temp=[1,0];% 1: old, 0: new
        for i=1:2
            for j=1:2
                if i == 1
                    trial{i,j,t} = find(res(1:38,t)==temp(j)); 
                elseif i == 2
                    trial{i,j,t} = find(res(39:end,t)==temp(j))+38;
                end
            end
        end

        TTrial = [trial{1,1,t}]; % Recall
%% Location memory
        for i = 1:size(TTrial,1)
            if string(Hash_Table_L(TTrial(i),2)) == string(Hash_Table_recall(TTrial(i),3))
                Suc(i) = TTrial(i);
            elseif string(Hash_Table_L(TTrial(i),2)) ~= string(Hash_Table_recall(TTrial(i),3))
                For(i) = TTrial(i);
            end
        end
        
        TTTrial = [trial{2,1}]; % i don't know
        Suc = nonzeros(Suc); For = [nonzeros(For)]; Don = TTTrial;    
        Succ = str2double(Hash_Table_recall(Suc,4)); Succ_1 = ones(size(Suc,1),1); Succeful = [Succ Succ_1]; %% 01
        Forg = str2double(Hash_Table_recall(For,4)); Forg_1 = zeros(size(For,1),1); Forgot = [Forg Forg_1]; %% 00
        Dont = str2double(Hash_Table_recall(Don,4)); Dont_1 = ones(size(Don,1),1)+1; I_Dont = [Dont Dont_1]; %% 02 I don't know

        Order = [cellfun(@(x) sprintf('%02d',x),num2cell(Succeful),'UniformOutput',false) ; 
            cellfun(@(x) sprintf('%02d',x),num2cell(Forgot),'UniformOutput',false) ; 
            cellfun(@(x) sprintf('%02d',x),num2cell(I_Dont),'UniformOutput',false)];
        Order = sortrows(Order,1);
        
        
        if t == 1
            Succesful = find(str2double(Order(:,2)) == 01);
            Succ_BN = str2double(Order(Succesful));
            ORDER = size(Order,1);
            clear Suc For
        elseif t == 2
            Succ_AN = find(str2double(Order(:,2)) == 01);
            Succ_AN = str2double(Order(Succ_AN));
            clear Suc For
        end
    end
%% True label
    label = intersect(Succ_BN, Succ_AN);   %image
    y = double(ismember(Succ_BN, label));
%% EEG file load
    Path = ['H:\1. Journal\NeuroImage\MT_Data\' Time{1} '\Data\'];    
    load([Path 'Sub' num2str(n) '_' Time{1}]); %EEG load
%% Trigger
    cnt = 0; ccnt = 0;
    for i = 1:size(MAT.MT,1)
        if MAT.MT(i, 2) == Trigger_LM(1,1)
            cnt = cnt+1;
            start_time(:,cnt) = round(cell2mat(MAT.MT(i, 1)));
        elseif MAT.MT(i, 2) == Trigger_LM(1,2)
            ccnt = ccnt+1;
            end_time(:,ccnt) = round(cell2mat(MAT.MT(i, 1)));
        end
    end
%% PSD Analysis
    for i = 1:ORDER
        Data = MAT.data(:, start_time(1,i):end_time(1,i)); 
        for j = 1:size(Data,1)
            xx = Data(j,:)';
            [X1, f]= periodogram(xx,rectwin(length(xx)),length(xx), fs);
            for r = 1:size(range,2)
                MT_temp(j,i,r) = 10*log10(bandpower(X1, f, range{r}, 'psd'));
            end
        end
    end 
    MT_temp_succ = MT_temp(:,Succesful,:); % Succ
    x = double(permute(MT_temp_succ, [1 3 2])); % channel x range x successful
%% Save
    save(['H:\Conference\Winter conference\GH_Memory\LM_PSD_24H\x_data\' 'Sub' num2str(n) '_x'],'x');
    save(['H:\Conference\Winter conference\GH_Memory\LM_PSD_24H\y_data\', 'Sub' num2str(n) '_y'],'y');

end 

