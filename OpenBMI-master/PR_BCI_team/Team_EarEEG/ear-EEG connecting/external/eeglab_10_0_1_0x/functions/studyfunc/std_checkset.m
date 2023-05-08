% std_checkset() - check STUDY set consistency
%
% Usage: >> [STUDY, ALLEEG] = std_checkset(STUDY, ALLEEG);  
%
% Input:
%   STUDY      - EEGLAB STUDY set
%   ALLEEG     - vector of EEG datasets included in the STUDY structure 
%
% Output:
%   STUDY      - a new STUDY set containing some or all of the datasets in ALLEEG, 
%                plus additional information from the optional inputs above. 
%   ALLEEG     - an EEGLAB vector of EEG sets included in the STUDY structure 
%
% Authors:  Arnaud Delorme & Hilit Serby, SCCN, INC, UCSD, November 2005

% Copyright (C) Arnaud Delorme, SCCN, INC, UCSD, arno@sccn.ucsd.edu
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

function [STUDY, ALLEEG] = std_checkset(STUDY, ALLEEG, option);

if nargin < 2
    help std_checkset;
    return;
end;
    
studywasempty = 0;
if isempty(STUDY), studywasempty = 1; end;

modif = 0;
if ~isfield(STUDY, 'name'),  STUDY.name  = ''; modif = 1; end;
if ~isfield(STUDY, 'task'),  STUDY.task  = ''; modif = 1; end;
if ~isfield(STUDY, 'notes'), STUDY.notes = ''; modif = 1; end;
if ~isfield(STUDY, 'filename'),  STUDY.filename  = ''; modif = 1; end;
if ~isfield(STUDY, 'filepath'),  STUDY.filepath  = ''; modif = 1; end;
if ~isfield(STUDY, 'history'),   STUDY.history   = ''; modif = 1; end;
if ~isfield(STUDY, 'subject'),   STUDY.subject   = {}; modif = 1; end;
if ~isfield(STUDY, 'group'),     STUDY.group     = {}; modif = 1; end;
if ~isfield(STUDY, 'session'),   STUDY.session   = {}; modif = 1; end;
if ~isfield(STUDY, 'condition'), STUDY.condition = {}; modif = 1; end;
if ~isfield(STUDY, 'setind'),    STUDY.setind    = {}; modif = 1; end;
if ~isfield(STUDY, 'etc'),       STUDY.etc       = []; modif = 1; end;
if ~isfield(STUDY, 'etc.warnmemory'), STUDY.etc.warnmemory = 1; modif = 1; end;
if ~isfield(STUDY, 'preclust'),  STUDY.preclust  = []; modif = 1; end;
if ~isfield(STUDY, 'datasetinfo'), STUDY.datasetinfo = []; modif = 1; end;
if ~isfield(STUDY.etc, 'version'), STUDY.etc.version = []; modif = 1; end;
if ~isfield(STUDY.preclust, 'erpclusttimes' ),  STUDY.preclust.erpclusttimes = []; modif = 1; end;
if ~isfield(STUDY.preclust, 'specclustfreqs' ), STUDY.preclust.specclustfreqs = []; modif = 1; end;
if ~isfield(STUDY.preclust, 'erspclustfreqs' ), STUDY.preclust.erspclustfreqs = []; modif = 1; end;
if ~isfield(STUDY.preclust, 'erspclusttimes' ), STUDY.preclust.erspclusttimes = []; modif = 1; end;
if ~isfield(STUDY.datasetinfo, 'comps') & ~isempty(STUDY.datasetinfo), STUDY.datasetinfo(1).comps = []; modif = 1; end;
if ~isfield(STUDY.datasetinfo, 'index') & ~isempty(STUDY.datasetinfo), STUDY.datasetinfo(1).index = []; modif = 1; end;

% all summary fields
% ------------------
try, subject = unique({ STUDY.datasetinfo.subject });
catch, 
     subject = ''; 
     disp('Important warning: some datasets do not have subject codes; some functions may crash!');
end;
try, group = unique({ STUDY.datasetinfo.group });
catch, 
     group = {}; 
     % disp('Important warning: some datasets do not have group codes; some functions may crash!');
end;
try, condition = unique({ STUDY.datasetinfo.condition });
catch, 
     condition = {}; 
     disp('Important warning: some datasets do not have condition codes; some functions may crash!');
end;
try, session = unique([STUDY.datasetinfo.session]);
catch, 
     session = ''; 
     % disp('Important warning: some datasets do not have session numbers; some functions may crash!');
end;
if ~isequal(STUDY.subject,   subject  ), STUDY.subject   = subject;   modif = 1; end;  
if ~isequal(STUDY.group,     group    ), STUDY.group     = group;     modif = 1; end;  
if ~isequal(STUDY.condition, condition), STUDY.condition = condition; modif = 1; end;  
if ~isequal(STUDY.session,   session  ), STUDY.session   = session;   modif = 1; end;  

% check dataset info consistency 
% ------------------------------
for k = 1:length(STUDY.datasetinfo)
    if ~strcmpi(STUDY.datasetinfo(k).filename, ALLEEG(k).filename)
        STUDY.datasetinfo(k).filename = ALLEEG(k).filename; modif = 1;
        fprintf('Warning: file name has changed for dataset %d and the study has been updated\n', k);
        fprintf('         to discard this change in the study, reload it from disk\n');
    end;
end;

% recompute setind array (setind is deprecated but we keep it anyway)
% -------------------------------------------------------------------
setind  = [];
sameica = std_findsameica(ALLEEG);
for index = 1:length(sameica)
    setind(length(sameica{index}):-1:1,index) = sameica{index}';
end;
setind(find(setind == 0)) = NaN;
if any(isnan(setind))
    warndlg('Warning: non-uniform set of dataset, some function might not work');
end
if ~isequal(setind, STUDY.setind), STUDY.setind = setind; modif = 1; end;

% check that dipfit is present in all datasets
% --------------------------------------------
for cc = 1:length(sameica)
    idat = [];
    for tmpi = 1:length(sameica{cc})
        if isfield(ALLEEG(sameica{cc}(tmpi)).dipfit, 'model')
            idat = sameica{cc}(tmpi);
        end;
    end;
    if ~isempty(idat)
        for tmpi = 1:length(sameica{cc})
            if ~isfield(ALLEEG(sameica{cc}(tmpi)).dipfit, 'model')
                ALLEEG(sameica{cc}(tmpi)).dipfit = ALLEEG(idat).dipfit;
                ALLEEG(sameica{cc}(tmpi)).saved  = 'no';
                fprintf('Warning: no ICA dipoles for dataset %d, using dipoles from dataset %d (same ICA)\n', sameica{cc}(tmpi), idat);
            end;
        end;
    end;
end;

% put in fake channels if channel labels are missing
% --------------------------------------------------
chanlabels = { ALLEEG.chanlocs };
if any(cellfun(@isempty, chanlabels))
    if any(~cellfun(@isempty, chanlabels))
        disp('********************************************************************');
        disp(' IMPORTANT WARNING: SOME DATASETS DO NOT HAVE CHANNEL LABELS AND ');
        disp(' SOME OTHERs HAVE CHANNEL LABELS. GENERATING CHANNEL LABELS FOR ');
        disp(' THE FORMER DATASETS (THIS SHOULD PROBABLY BE FIXED BY THE USER)');
        disp('********************************************************************');
    end;
    disp('Generating channel labels for all datasets...');
    for currentind = 1:length(ALLEEG)
        for ind = 1:ALLEEG(currentind).nbchan
            ALLEEG(currentind).chanlocs(ind).labels = int2str(ind);
        end;
    end;
    ALLEEG(currentind).saved = 'no';
end;

if length( unique( [ ALLEEG.srate ] )) > 1
    disp('********************************************************************');
    disp(' IMPORTANT WARNING: SOME DATASETS DO NOT HAVE THE SAME SAMPLING ');
    disp(' RATE AND THIS WILL MAKE MOST SOME STUDY FUNCTION CRASH. THIS');
    disp(' SHOULD PROBABLY BE FIXED BY THE USER).');
    disp('********************************************************************');
end;

% check cluster array
% -------------------
if ~isfield(STUDY, 'cluster'), STUDY.cluster = []; modif = 1; end;
if isempty(STUDY.cluster)
    modif = 1; 
    [STUDY] = std_createclust(STUDY, ALLEEG, 'parentcluster', 'on');
end;
if ~isfield(STUDY, 'changrp')
    STUDY.changrp = [];
end;

% create STUDY design if it is not present
% ----------------------------------------
if isfield(STUDY.datasetinfo, 'trialinfo')
    alltrialinfo = { STUDY.datasetinfo.trialinfo };
    if any(cellfun(@isempty, alltrialinfo)) && any(~cellfun(@isempty, alltrialinfo))
        disp('Rebuilding trial information structure for STUDY');
        STUDY  = std_maketrialinfo(STUDY, ALLEEG); % some dataset do not have trialinfo and
                                                   % some other have it, remake it for everybody
    end;
end;
if ~isfield(STUDY, 'design') || isempty(STUDY.design) || ~isfield(STUDY.design, 'name')
    STUDY  = std_maketrialinfo(STUDY, ALLEEG); 
    STUDY  = std_makedesign(STUDY, ALLEEG);
    STUDY  = std_selectdesign(STUDY, ALLEEG,1);
else
    if isfield(STUDY.design, 'indvar1')
        STUDY  = std_convertdesign(STUDY, ALLEEG);
    end;
    
    % convert combined independent variable values
    % between dash to cell array of strings
    % -------------------------------------
    for inddes = 1:length(STUDY.design)
        for indvar = 1:length(STUDY.design(inddes).variable)
            for indval = 1:length(STUDY.design(inddes).variable(indvar).value)
                STUDY.design(inddes).variable(indvar).value{indval} = convertindvarval(STUDY.design(inddes).variable(indvar).value{indval});
            end;
        end;
        for indcell = 1:length(STUDY.design(inddes).cell)
            for indval = 1:length(STUDY.design(inddes).cell(indcell).value)
                STUDY.design(inddes).cell(indcell).value{indval} = convertindvarval(STUDY.design(inddes).cell(indcell).value{indval});
            end;
        end;
        for indinclude = 1:length(STUDY.design(inddes).include)
            if iscell(STUDY.design(inddes).include{indinclude})
                for indval = 1:length(STUDY.design(inddes).include{indinclude})
                    STUDY.design(inddes).include{indinclude}{indval} = convertindvarval(STUDY.design(inddes).include{indinclude}{indval});
                end;
            end;
        end;
    end;
end;

% scan design to fix old paring format
% ------------------------------------
for design = 1:length(STUDY.design)
    for var = 1:length(STUDY.design(design).variable)
        if isstr(STUDY.design(design).variable(1).pairing)
            if strcmpi(STUDY.design(design).variable(1).pairing, 'paired')
                STUDY.design(design).variable(1).pairing = 'on';
            elseif strcmpi(STUDY.design(design).variable(1).pairing, 'unpaired')
                STUDY.design(design).variable(1).pairing = 'off';
            end;
        end;
        if isstr(STUDY.design(design).variable(2).pairing)
            if strcmpi(STUDY.design(design).variable(2).pairing, 'paired')
                STUDY.design(design).variable(2).pairing = 'on';
            elseif strcmpi(STUDY.design(design).variable(2).pairing, 'unpaired')
                STUDY.design(design).variable(2).pairing = 'off';
            end;
        end;
    end;
end;

% check that ICA is present and if it is update STUDY.datasetinfo
allcompsSTUDY  = { STUDY.datasetinfo.comps };
allcompsALLEEG = { ALLEEG.icaweights };
if all(cellfun(@isempty, allcompsSTUDY)) && ~all(cellfun(@isempty, allcompsALLEEG))
    for index = 1:length(STUDY.datasetinfo)
        STUDY.datasetinfo(index).comps = [1:size(ALLEEG(index).icaweights,1)];
    end;
end;

% make channel groups
% -------------------
if ~isfield(STUDY, 'changrp') || isempty(STUDY.changrp)
    STUDY = std_changroup(STUDY, ALLEEG);
    modif = 1;
end;

% determine if there has been any change
% --------------------------------------
if modif;
    STUDY.saved = 'no';
    if studywasempty
        command = '[STUDY, ALLEEG] = std_checkset([], ALLEEG);';
    else
        command = '[STUDY, ALLEEG] = std_checkset(STUDY, ALLEEG);';
    end;
    eegh(command);
    STUDY.history =  sprintf('%s\n%s',  STUDY.history, command);
end;

% convert combined independent variables
% --------------------------------------
function val = convertindvarval(val);
    if isstr(val)
        inddash = findstr(' - ', val);
        if isempty(inddash), return; end;
        inddash = [ -2 inddash length(val)+1];
        for ind = 1:length(inddash)-1
            newval{ind} = val(inddash(ind)+3:inddash(ind+1)-1);
        end;
        val = newval;
    end;


