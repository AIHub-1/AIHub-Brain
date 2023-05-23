% eeg_getdatact() - get EEG data from a specified dataset or
%                  component activity
%
% Usage:
%       >> signal = eeg_getdatact( EEG );
%       >> signal = eeg_getdatact( EEG, 'key', 'val');
%
% Inputs:
%   EEG       - Input dataset
%
% Optional input:
%   'channel'   - [integer array] read only specific channels.
%                 Default is to read all data channels.
%   'component' - [integer array] read only specific components
%   'rmcomps'   - [integer array] remove selected components from data
%                 channels. This is only to be used with channel data not
%                 when selecting components.
%   'trialindices' - [integer array] only read specific trials. Default is
%                 to read all trials.
%   'samples'   - [integer array] only read specific samples. Default is
%                 to read all samples.
%   'reshape'   - ['2d'|'3d'] reshape data. Default is '3d' when possible.
%   'verbose'   - ['on'|'off'] verbose mode. Default is 'on'.
%
% Outputs:
%   signal      - EEG data or component activity
%
% Author: Arnaud Delorme, SCCN & CERCO, CNRS, 2008-
%
% See also: eeg_checkset()

% Copyright (C) 15 Feb 2002 Arnaud Delorme, Salk Institute, arno@salk.edu
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

function data = eeg_getdatact( EEG, varargin);
    
    data = [];
    if nargin < 1
        help eeg_getdatact;
        return;
    end;
    
    opt = finputcheck(varargin, { ...
        'channel'   'integer' {} [1:EEG.nbchan];
        'verbose'   'string'  { 'on' 'off' } 'on';
        'reshape'   'string'  { '2d' '3d' }  '3d';
        'component' 'integer' {} [];        
        'samples'   'integer' {} [];        
        'trialindices' 'integer' {} [];        
        'rmcomps'   'integer' {} [] }, 'eeg_getdatact');
    
    if isstr(opt), error(opt); end;
    if (~isempty(opt.rmcomps) | ~isempty(opt.component)) & isempty(EEG.icaweights)
        error('No ICA weight in dataset');
    end;
    
    if strcmpi(EEG.data, 'in set file')
        EEG = pop_loadset('filename', EEG.filename, 'filepath', EEG.filepath);
    end;
    
    % getting channel or component activation
    % ---------------------------------------
    filename = fullfile(EEG.filepath, [ EEG.filename(1:end-4) '.icaact' ] );
    if ~isempty(opt.component) & ~isempty(EEG.icaact)

        data = EEG.icaact(opt.component,:,:);

    elseif ~isempty(opt.component) & exist(filename)

        % reading ICA file
        % ----------------
        data = repmat(single(0), [ length(opt.component) EEG.pnts EEG.trials ]);
        fid = fopen( filename, 'r', 'ieee-le'); %little endian (see also pop_saveset)
        if fid == -1, error( ['file ' filename ' could not be open' ]); end;
        for ind = 1:length(opt.component)
            fseek(fid, (opt.component(ind)-1)*EEG.pnts*EEG.trials*4, -1);
            data(ind,:) = fread(fid, [EEG.trials*EEG.pnts 1], 'float32')';
        end;
        fclose(fid);

    elseif ~isempty(opt.component)

        if isempty(EEG.icaact)
            data = eeg_getdatact( EEG );
            data = (EEG.icaweights(opt.component,:)*EEG.icasphere)*data(EEG.icachansind,:);
        else
            data = EEG.icaact(opt.component,:,:);
        end;

    elseif isnumeric(EEG.data) % channel

        data = EEG.data(opt.channel,:,:);

    else % channel but no data loaded

        filename = fullfile(EEG.filepath, EEG.data);
        fid = fopen( filename, 'r', 'ieee-le'); %little endian (see also pop_saveset)
        if fid == -1
            error( ['file ' filename ' not found. If you have renamed/moved' 10 ...
                    'the .set file, you must also rename/move the associated data file.' ]);
        else 
            if strcmpi(opt.verbose, 'on')
                fprintf('Reading float file ''%s''...\n', filename);
            end;
        end;
        
        % old format = .fdt; new format = .dat (transposed)
        % -------------------------------------------------
        datformat = 0;
        if length(filename) > 3
            if strcmpi(filename(end-2:end), 'dat')
                datformat = 1;
            end;
        end;
        EEG.datfile = EEG.data;

        % reading data file
        % -----------------
        eeglab_options;
        if length(opt.channel) == EEG.nbchan & option_memmapdata
            fclose(fid);
            data = memmapdata(filename, [EEG.nbchan EEG.pnts EEG.trials]);
        else
            if datformat
                if length(opt.channel) == EEG.nbchan
                    data = fread(fid, [EEG.trials*EEG.pnts EEG.nbchan], 'float32')';
                else
                    data = repmat(single(0), [ length(opt.channel) EEG.pnts EEG.trials ]);
                    for ind = 1:length(opt.channel)
                        fseek(fid, (opt.channel(ind)-1)*EEG.pnts*EEG.trials*4, -1);
                        data(ind,:) = fread(fid, [EEG.trials*EEG.pnts 1], 'float32')';
                    end;
                end;
            else
                data = fread(fid, [EEG.nbchan Inf], 'float32');
                data = data(opt.channel,:,:);
            end;
            fclose(fid);
        end;

    end;
 
    % subracting components from data
    % -------------------------------
    if ~isempty(opt.rmcomps)
        if strcmpi(opt.verbose, 'on')
            fprintf('Removing %d artifactual components\n', length(opt.rmcomps));
        end;
        rmcomps = eeg_getdatact( EEG, 'component', opt.rmcomps); % loaded from file        
        rmchan    = [];
        rmchanica = [];
        for index = 1:length(opt.channel)
            tmpicaind = find(opt.channel(index) == EEG.icachansind);
            if ~isempty(tmpicaind)
                rmchan    = [ rmchan    index ];
                rmchanica = [ rmchanica tmpicaind ];
            end;
        end;
        data(rmchan,:) = data(rmchan,:) - EEG.icawinv(rmchanica,opt.rmcomps)*rmcomps(:,:);

        %EEG = eeg_checkset(EEG, 'loaddata');
        %EEG = pop_subcomp(EEG, opt.rmcomps);
        %data = EEG.data(opt.channel,:,:);
        
        %if strcmpi(EEG.subject, 'julien') & strcmpi(EEG.condition, 'oddball') & strcmpi(EEG.group, 'after')
        %    jjjjf
        %end;
    end;
    
    try,
        if  strcmpi(opt.reshape, '3d')
             data = reshape(data, size(data,1), EEG.pnts, EEG.trials);
        else data = reshape(data, size(data,1), EEG.pnts*EEG.trials);
        end;
    catch
        error('The file size on disk does not correspond to the dataset information.');
    end;
    
    if ~isempty(opt.trialindices)
        data = data(:,:,opt.trialindices);
    end;
    if ~isempty(opt.samples)
        data = data(:,opt.samples,:);
    end;
    
