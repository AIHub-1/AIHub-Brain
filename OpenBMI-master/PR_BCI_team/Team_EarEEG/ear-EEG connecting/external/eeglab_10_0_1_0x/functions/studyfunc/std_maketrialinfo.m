% std_maketrialinfo() - create trial information structure using the 
%                       .epoch structure of EEGLAB datasets
%
% Usage: 
%   >> STUDY = std_maketrialinfo(STUDY, ALLEEG);  
%
% Inputs:
%   STUDY      - EEGLAB STUDY set
%   ALLEEG     - vector of the EEG datasets included in the STUDY structure 
%
% Inputs:
%   STUDY      - EEGLAB STUDY set updated. The fields which is created or
%                updated is STUDY.datasetinfo.trialinfo
%
% Authors: Arnaud Delorme, SCCN/INC/UCSD, April 2010

% Copyright (C) Arnaud Delorme arno@ucsd.edu
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

function STUDY = std_maketrialinfo(STUDY, ALLEEG);

%% test if .epoch field exist in ALLEEG structure
epochfield = cellfun(@isempty, { ALLEEG.epoch });
if any(epochfield)
    fprintf('Warning: some datasets are continuous and trial information cannot be created\n');
    return;
end;

%% Make trial info
for index = 1:length(ALLEEG)
    eventlat = abs(eeg_point2lat( [ ALLEEG(index).event.latency ], [ ALLEEG(index).event.epoch ], ALLEEG(index).srate, [ALLEEG(index).xmin ALLEEG(index).xmax]));
    events   = ALLEEG(index).event;
    ff = fieldnames(events);
    ff = setdiff(ff, { 'latency' 'urevent' 'epoch' });
    trialinfo = [];
    
    % process time locking event fields
    % ---------------------------------
    indtle    = find(eventlat < 0.02);
    epochs    = [ events(indtle).epoch ];
    extractepoch = true;
    if length(epochs) ~= ALLEEG(index).trials
        
        % special case where there are not the same number of time-locking
        % event as there are data epochs
        if length(unique(epochs)) ~= ALLEEG(index).trials
            extractepoch = false;
            disp('std_maketrialinfo: not the same number of time-locking events as trials, trial info ignored');
        else
            % pick one event per epoch
            [tmp tmpind] = unique(epochs(end:-1:1)); % reversing the array ensures the first event gets picked
            tmpind = length(epochs)+1-tmpind;
            indtle = indtle(tmpind);
            if length(indtle) ~= ALLEEG(index).trials
                extractepoch = false;
                disp('std_maketrialinfo: not the same number of time-locking events as trials, trial info ignored');
            end;
        end;
    end;
    if extractepoch
        commands = {};
        for f = 1:length(ff)
            eval( [ 'eventvals = {events(indtle).' ff{f} '};' ]);
            %if isnumeric(eventvals{1})
            %    eventvals = cellfun(@num2str, eventvals, 'uniformoutput', false);
            %end;
            commands = { commands{:} ff{f} eventvals };
        end;
        trialinfo = struct(commands{:});
        STUDY.datasetinfo(index).trialinfo = trialinfo;
    end;
    
%    % same as above but 10 times slower
%     for e = 1:length(ALLEEG(index).event)
%         if eventlat(e) < 0.0005 % time locking event only
%             epoch = events(e).epoch;
%             for f = 1:length(ff)
%                 fieldval  = getfield(events, {e}, ff{f});
%                 trialinfo = setfield(trialinfo, {epoch}, ff{f}, fieldval);
%             end;
%         end;
%     end;
end;

    
