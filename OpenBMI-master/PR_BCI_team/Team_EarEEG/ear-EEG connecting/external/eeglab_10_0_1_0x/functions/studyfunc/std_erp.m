% std_erp() -   Constructs and returns channel or ICA activation ERPs for a dataset. 
%               Saves the ERPs into a Matlab file, [dataset_name].icaerp, for
%               data channels or [dataset_name].icaerp for ICA components, 
%               in the same directory as the dataset file.  If such a file 
%               already exists, loads its information. 
% Usage:    
%            >> [erp, times] = std_erp(EEG, 'key', 'val', ...);
% Inputs:
%   EEG          - a loaded epoched EEG dataset structure. 
%
% Optional inputs:
%   'components' - [numeric vector] components of the EEG structure for which 
%                  activation ERPs will be computed. Note that because 
%                  computation of ERP is so fast, all components ERP are
%                  computed and saved. Only selected component 
%                  are returned by the function to Matlab
%                  {default|[] -> all}
%   'channels'   - [cell array] channels of the EEG structure for which 
%                  activation ERPs will be computed. Note that because 
%                  computation of ERP is so fast, all channels ERP are
%                  computed and saved. Only selected channels 
%                  are returned by the function to Matlab
%                  {default|[] -> none}
%   'recompute'  - ['on'|'off'] force recomputing ERP file even if it is 
%                  already on disk.
% Outputs:
%   erp          - ERP for the requested ICA components in the selected 
%                  latency window. ERPs are scaled by the RMS over of the
%                  component scalp map projection over all data channels.
%   times        - vector of times (epoch latencies in ms) for the ERP
%
% File output:     
%    [dataset_file].icaerp     % component erp file
% OR
%    [dataset_file].daterp     % channel erp file
%
% See also: std_spec(), std_ersp(), std_topo(), std_preclust()
%
% Authors: Arnaud Delorme, SCCN, INC, UCSD, January, 2005

% Copyright (C) Arnaud Delorme, SCCN, INC, UCSD, October 11, 2004, arno@sccn.ucsd.edu
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

function [X, t] = std_erp(EEG, varargin); %comps, timerange)

if nargin < 1
    help std_erp;
    return;
end;

% decode inputs
% -------------
if ~isempty(varargin) 
    if ~isstr(varargin{1})
        varargin = { varargin{:} [] [] };
        if all(varargin{1} > 0) 
            options = { 'components' varargin{1} 'timerange' varargin{2} };
        else
            options = { 'channels' -varargin{1} 'timerange' varargin{2} };
        end;
    else
        options = varargin;
    end;
else
    options = varargin;
end;

g = finputcheck(options, { 'components' 'integer' []         [];
                           'channels'   'cell'    {}         {};
                           'rmbase'     'real'    []         [];
                           'trialindices' { 'integer' 'cell' } []         [];
                           'rmcomps'    'cell'    []         cell(1,length(EEG));
                           'fileout'    'string'  []         '';
                           'savetrials' 'string'  { 'on' 'off' } 'off';
                           'interp'     'struct'  { }        struct([]);
                           'recompute'  'string'  { 'on' 'off' } 'off';
                           'timerange'  'real'    []         [] }, 'std_erp');
if isstr(g), error(g); end;
if isempty(g.trialindices), g.trialindices = cell(length(EEG)); end;
if ~iscell(g.trialindices), g.trialindices = { g.trialindices }; end;
if isfield(EEG,'icaweights')
   numc = size(EEG(1).icaweights,1);
else
   error('EEG.icaweights not found');
end
if isempty(g.components)
    g.components = 1:numc;
end

EEG_etc = [];

% % THIS SECTION WOULD NEED TO TEST THAT THE PARAMETERS ON DISK ARE CONSISTENT
%
% % filename 
% % --------
if isempty(g.fileout), g.fileout = fullfile(EEG(1).filepath, EEG(1).filename(1:end-4)); end;
if ~isempty(g.channels)
    filenameshort = [ g.fileout '.daterp'];
    prefix = 'chan';
else    
    filenameshort = [ g.fileout '.icaerp'];
    prefix = 'comp';
end;
%filename = fullfile( EEG(1).filepath, filenameshort);
filename = filenameshort;

% ERP information found in datasets
% ---------------------------------
if exist(filename) & strcmpi(g.recompute, 'off')

    fprintf('File "%s" found on disk, no need to recompute\n', filenameshort);
    setinfo.filebase = g.fileout;
    if strcmpi(prefix, 'comp')
        [X tmp t] = std_readfile(setinfo, 'components', g.components, 'timelimits', g.timerange, 'measure', 'erp');
    else
        [X tmp t] = std_readfile(setinfo, 'channels', g.channels,  'timelimits', g.timerange, 'measure', 'erp');
    end;
    if ~isempty(X), return; end;
    
end 
   
% No ERP information found
% ------------------------
% if isstr(EEG.data)
%     TMP = eeg_checkset( EEG, 'loaddata' ); % load EEG.data and EEG.icaact
% else
%     TMP = EEG;
% end
%    & isempty(TMP.icaact)
%    TMP.icaact = (TMP.icaweights*TMP.icasphere)* ...
%        reshape(TMP.data(TMP.icachansind,:,:), [ length(TMP.icachansind) size(TMP.data,2)*size(TMP.data,3) ]);
%    TMP.icaact = reshape(TMP.icaact, [ size(TMP.icaact,1) size(TMP.data,2) size(TMP.data,3) ]);
%end;
%if strcmpi(prefix, 'comp'), X = TMP.icaact;
%else                        X = TMP.data;
%end;
options = {};
if ~isempty(g.rmcomps), options = { options{:} 'rmcomps' g.rmcomps }; end;
if ~isempty(g.interp),  options = { options{:} 'interp' g.interp }; end;
X       = [];
for dat = 1:length(EEG)
    if strcmpi(prefix, 'comp')
        tmpdata = eeg_getdatact(EEG(dat), 'component', [1:size(EEG(dat).icaweights,1)], 'trialindices', g.trialindices{dat} );
    else
        EEG(dat).data = eeg_getdatact(EEG(dat), 'channel', [1:EEG(dat).nbchan], 'rmcomps', g.rmcomps{dat}, 'trialindices', g.trialindices{dat});
        EEG(dat).trials = size(EEG(dat).data,3);
        EEG(dat).event  = [];
        EEG(dat).epoch  = [];
        if ~isempty(g.interp), 
            EEG(dat) = eeg_interp(EEG(dat), g.interp, 'spherical'); 
        end;
        tmpdata = EEG(dat).data;
    end;
    if isempty(X), X = tmpdata;
    else
        if size(X,1) ~= size(tmpdata,1), error('Datasets to be concatenated do not have the same number of channels'); end;
        if size(X,2) ~= size(tmpdata,2), error('Datasets to be concatenated do not have the same number of time points'); end;
        X(:,:,end+1:end+size(tmpdata,3)) = tmpdata; % concatenating trials
    end;
end;

% Remove baseline mean
% --------------------
pnts     = EEG(1).pnts;
trials   = size(X,3);
timevals = EEG(1).times;
if ~isempty(g.timerange)
    disp('Warning: the ''timerange'' option is deprecated and has no effect');
end;
if ~isempty(g.rmbase)
    disp('Removing baseline...');
    options = { options{:} 'rmbase' g.rmbase };
    [tmp timebeg] = min(abs(timevals - g.rmbase(1)));
    [tmp timeend] = min(abs(timevals - g.rmbase(2)));
    if ~isempty(timebeg)
        X = rmbase(X,pnts, [timebeg:timeend]);
    else
        X = rmbase(X,pnts);
    end
end
X = reshape(X, [ size(X,1) pnts trials ]);
if strcmpi(prefix, 'comp')
    if strcmpi(g.savetrials, 'on')
        X = repmat(sqrt(mean(EEG(1).icawinv.^2))', [1 EEG(1).pnts size(X,3)]) .* X;
    else
        X = repmat(sqrt(mean(EEG(1).icawinv.^2))', [1 EEG(1).pnts]) .* mean(X,3); % calculate ERP
    end;
elseif strcmpi(g.savetrials, 'off')
    X = mean(X, 3);
end;

% Save ERPs in file (all components or channels)
% ----------------------------------------------
if isempty(timevals), timevals = linspace(EEG(1).xmin, EEG(1).xmax, EEG(1).pnts)*1000; end; % continuous data
if strcmpi(prefix, 'comp')
    savetofile( filename, timevals, X, 'comp', 1:size(X,1), options);
    %[X,t] = std_readerp( EEG, 1, g.components, g.timerange);
else
    if ~isempty(g.interp)
        savetofile( filename, timevals, X, 'chan', 1:size(X,1), options, { g.interp.labels });
    else
        tmpchanlocs = EEG(1).chanlocs;
        savetofile( filename, timevals, X, 'chan', 1:size(X,1), options, { tmpchanlocs.labels });
    end;
    %[X,t] = std_readerp( EEG, 1, g.channels, g.timerange);
end;

% -------------------------------------
% saving ERP information to Matlab file
% -------------------------------------
function savetofile(filename, t, X, prefix, comps, params, labels);
    
    disp([ 'Saving ERP file ''' filename '''' ]);
    allerp = [];
    for k = 1:length(comps)
        allerp = setfield( allerp, [ prefix int2str(comps(k)) ], squeeze(X(k,:,:)));
    end;
    if nargin > 6
        allerp.labels = labels;
    end;
    allerp.times      = t;
    allerp.datatype   = 'ERP';
    allerp.parameters = params;
    std_savedat(filename, allerp);
