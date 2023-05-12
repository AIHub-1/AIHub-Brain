%% MT analysis - PSD
clear; close all; clc;

addpath('E:\Users\SHIN\eeglab14_1_2b\');
eeglab;
%% 
format long g
fs = 250;

% Delta, theta, alpha, spindle, beta, gamma
range = {[0.5 4],[4 7],[7 12],[12 15],[15 30],[30 50]};

% Before Nap(BN), After Nap(AN)
Time = {'BN', '24'};
% Time = {'BN', 'AN'};

Trigger_PM = ["S 53", "S 54" , "S 52", "S 57"]; %Pocation Memory
% Time = {'BN', 'AN', '24H'};
% Trigger_PM = ["S 53", "S 54" , "S 52", "S 57" ; 
%     "S 63", "S 64", "S 62", "S 67" ;
%     "S 63", "S 64", "S 62", "S 67"]; %Pocation Memory
%% MT_textfile load
Path_M = 'H:\1. Journal\NeuroImage\SLEEP_DATA_24H\';
DirGroup_M = dir(fullfile(Path_M,'*'));
FileNamesGroup_M = {DirGroup_M.name};
FileNamesGroup_M = FileNamesGroup_M(1,3:end);
Time_M = {'BN','24'};
removal_word = [".jpg"];
%% 
for n = 1:size(FileNamesGroup_M,2)
%% PM Task (Succeful)           
    % Recall
    for t = 1:size(Time,2)
        Data_recall=textscan(fopen([FileNamesGroup_M{n} '_' Time_M{t} '_recall_visuo.txt']), '%s %s %s %s %s %s %s');   
        Data_recall=[Data_recall{:}];
        Hash_Table_recall = str2double(erase(Data_recall([2:end],3),removal_word));
        Hash_Table_recall = [cellfun(@(x) sprintf('%02d',x),num2cell(Hash_Table_recall),'UniformOutput',false)...
            Data_recall([2:end],4) Data_recall([2:end],6) Data_recall([2:end],2)];
        Hash_Table_recall = sortrows(Hash_Table_recall,1);

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
        if t == 1
            Suc = [trial{1,1,t}]; 
            Succ = sortrows(str2double(Hash_Table_recall(Suc,4)));
        end    
    end
    
%% True label    
    label = intersect(trial{1,1,1}, trial{1,1,2});
  
    % Before nap
    Data_recall=textscan(fopen([FileNamesGroup_M{n} '_' Time_M{1} '_recall_visuo.txt']), '%s %s %s %s %s %s %s');   
    Data_recall=[Data_recall{:}];
    Hash_Table_recall = str2double(erase(Data_recall([2:end],3),removal_word));
    Hash_Table_recall = [cellfun(@(x) sprintf('%02d',x),num2cell(Hash_Table_recall),'UniformOutput',false)...
        Data_recall([2:end],4) Data_recall([2:end],6) Data_recall([2:end],2)];
    Hash_Table_recall = sortrows(Hash_Table_recall,1);
    
    Label = sortrows(str2double(Hash_Table_recall(label,4))); 
    y = double(ismember(Succ, Label));
%% EEG file load
    Path = ['H:\1. Journal\NeuroImage\MT_Data\' Time{1} '\Data\'];
    load([Path 'Sub' num2str(n) '_' Time{1}]); %EEG load
%% Trigger
    cnt = 0;
    for j = 1:size(MAT.MT,1)
        if MAT.MT(j,2) == Trigger_PM(1,1)
            cnt = cnt+1;
            A{cnt} = [MAT.MT(j,1) MAT.MT(j,2)
                MAT.MT(j+1,1) MAT.MT(j+1,2)];
        end
    end

    AA=A'; AAA = strings([152,2]);
    cnt = 1; ccnt = 2;
    for j = 1:size(AA,1)
        AAA([cnt:ccnt],:)  = [AA{j}];
        cnt = cnt+2;
        ccnt = ccnt+2;
    end            
    TTime = double(AAA(:,1));
    Tri = AAA(:,2);

    PM_start = find(cellfun(@(x) x(end - 3:end) == Trigger_PM(1,1), Tri) == 1);  
    PM_end_1 = find(cellfun(@(x) x(end - 3:end) == Trigger_PM(1,2), Tri) == 1);
    PM_end_2 = find(cellfun(@(x) x(end - 3:end) == Trigger_PM(1,3), Tri) == 1);
    PM_end_3 = find(cellfun(@(x) x(end - 3:end) == Trigger_PM(1,4), Tri) == 1);

    jj = 1; jjj = 1;
    for j = 1:size(PM_start) % 1-76
        if PM_end_3 == 152
            Data = MAT.data(:, round(TTime(PM_start(j)):round(TTime(PM_end_3))));
        elseif jjj > length(PM_end_2) %
            Data = MAT.data(:, round(TTime(PM_start(j))):round(TTime(PM_end_1(jj))));
            jj = jj+1;
        elseif jj > length(PM_end_1)
            Data = MAT.data(:, round(TTime(PM_start(j))):round(TTime(PM_end_2(jjj))));
            jjj = jjj+1;               
        else
            if PM_end_1(jj) < PM_end_2(jjj) %
                Data = MAT.data(:, round(TTime(PM_start(j))):round(TTime(PM_end_1(jj))));
                jj = jj + 1;
            elseif PM_end_1(jj) > PM_end_2(jjj) %
                Data = MAT.data(:, round(TTime(PM_start(j))):round(TTime(PM_end_2(jjj))));
                jjj = jjj + 1;
            end
        end             
%% PSD Analysis
        for k = 1:size(Data,1)
            xx = Data(k,:)';
            [X1, f]= periodogram(xx,rectwin(length(xx)),length(xx), fs);
            for r = 1:size(range,2)
                MT_temp(k,j,r) = 10*log10(bandpower(X1, f, range{r}, 'psd'));
            end
        end
    end         
    MT_temp_succ = MT_temp(:,Succ,:); % Succ
    x = double(permute(MT_temp_succ, [1 3 2])); % channel x range x successful
%% Save
    save(['H:\Conference\Winter conference\GH_Memory\PM_PSD_24H\x_data\' 'Sub' num2str(n) '_x'],'x');
    save(['H:\Conference\Winter conference\GH_Memory\PM_PSD_24H\y_data\', 'Sub' num2str(n) '_y'],'y');
end