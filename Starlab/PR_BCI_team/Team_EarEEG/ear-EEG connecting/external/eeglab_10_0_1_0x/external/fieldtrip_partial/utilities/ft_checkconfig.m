function [cfg] = ft_checkconfig(cfg, varargin)

% FT_CHECKCONFIG checks the input cfg of the main FieldTrip functions.
%
% 1: It checks whether the cfg contains all the required options, it gives
% a warning when renamed or deprecated options are used, and it makes sure
% no forbidden options are used. If necessary and possible, this function
% will adjust the cfg to the input requirements. If the input cfg does NOT
% correspond to the requirements, this function gives an elaborate warning
% message.
%
% 2: It controls the relevant cfg options that are being passed on to other
% functions, by putting them into substructures or converting them into the
% required format.
%
% 3: It controls the output cfg (data.cfg) such that it only contains
% relevant and used fields. The size of fields in the output cfg is also
% controlled: fields exceeding a certain maximum size are emptied.
% This part of the functionality is still under construction!
%
% Use as
%   [cfg] = ft_checkconfig(cfg, ...)
%
% The behaviour of checkconfig can be controlled by the following cfg options,
% which can be set as global fieldtrip defaults (see FT_DEFAULTS):
%   cfg.checkconfig = 'pedantic', 'loose' or 'silent' (control the feedback behaviour of checkconfig)
%   cfg.trackconfig = 'cleanup', 'report' or 'off'
%   cfg.checksize   = number in bytes, can be inf (set max size allowed for output cfg fields)
%
% Optional input arguments should be specified as key-value pairs and can include
%   renamed         = {'old',  'new'}        % list the old and new option
%   renamedval      = {'opt',  'old', 'new'} % list option and old and new value
%   required        = {'opt1', 'opt2', etc.} % list the required options
%   deprecated      = {'opt1', 'opt2', etc.} % list the deprecated options
%   unused          = {'opt1', 'opt2', etc.} % list the unused options, these will be removed and a warning is issued
%   forbidden       = {'opt1', 'opt2', etc.} % list the forbidden options, these result in an error
%   createsubcfg    = {'subname', etc.}      % list the names of the subcfg
%   dataset2files   = 'yes', 'no'            % converts dataset into headerfile and datafile
%   checksize       = 'yes', 'no'            % remove large fields from the cfg
%   trackconfig     = 'on', 'off'            % start/end config tracking
%
% See also FT_CHECKDATA, FT_DEFAULTS

% Copyright (C) 2007-2008, Robert Oostenveld, Saskia Haegens
%
% This file is part of FieldTrip, see http://www.ru.nl/neuroimaging/fieldtrip
% for the documentation and details.
%
%    FieldTrip is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    FieldTrip is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with FieldTrip. If not, see <http://www.gnu.org/licenses/>.
%
% $Id: ft_checkconfig.m 2865 2011-02-12 19:24:57Z roboos $

if isempty(cfg)
  cfg = struct; % ensure that it is an empty struct, not empty double
end

global ft_default
if isempty(ft_default)
  ft_default = struct;
end
fieldsused = fieldnames(ft_default);
for i=1:length(fieldsused)
  fn = fieldsused{i};
  if ~isfield(cfg, fn),
    cfg.(fn) = ft_default.(fn);
  end
end

renamed         = keyval('renamed',         varargin);
renamedval      = keyval('renamedval',      varargin);
required        = keyval('required',        varargin);
deprecated      = keyval('deprecated',      varargin);
unused          = keyval('unused',          varargin);
forbidden       = keyval('forbidden',       varargin);
createsubcfg    = keyval('createsubcfg',    varargin);
dataset2files   = keyval('dataset2files',   varargin);
checksize       = keyval('checksize',       varargin); if isempty(checksize), checksize = 'off';  end
trackconfig     = keyval('trackconfig',     varargin);

if ~isempty(trackconfig) && strcmp(trackconfig, 'on')
  % infer from the user configuration whether tracking should be enabled
  if isfield(cfg, 'trackconfig') && (strcmp(cfg.trackconfig, 'report') || strcmp(cfg.trackconfig, 'cleanup'))
    trackconfig = 'on'; % turn on configtracking if user requests report/cleanup
  else
    trackconfig = []; % disable configtracking if user doesn't request report/cleanup
  end
end

% these should be cell arrays and not strings
if ischar(required),   required   = {required};   end
if ischar(deprecated), deprecated = {deprecated}; end
if ischar(unused),     unused     = {unused};     end
if ischar(forbidden),  forbidden  = {forbidden};  end

if isfield(cfg, 'checkconfig')
  silent   = strcmp(cfg.checkconfig, 'silent');
  loose    = strcmp(cfg.checkconfig, 'loose');
  pedantic = strcmp(cfg.checkconfig, 'pedantic');
else
  silent   = false;
  loose    = true;
  pedantic = false;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rename old to new options, give warning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(renamed)
  fieldsused = fieldnames(cfg);
  if any(strcmp(renamed{1}, fieldsused))
    cfg = setfield(cfg, renamed{2}, (getfield(cfg, renamed{1})));
    cfg = rmfield(cfg, renamed{1});
    if silent
      % don't mention it
    elseif loose
      warning('use cfg.%s instead of cfg.%s', renamed{2}, renamed{1});
    elseif pedantic
      error(sprintf('use cfg.%s instead of cfg.%s', renamed{2}, renamed{1}));
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rename old to new value, give warning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(renamedval) && isfield(cfg, renamedval{1})
  if strcmpi(getfield(cfg, renamedval{1}), renamedval{2})
    cfg = setfield(cfg, renamedval{1}, renamedval{3});
    if silent
      % don't mention it
    elseif loose
      warning('use cfg.%s=''%s'' instead of cfg.%s=''%s''', renamedval{1}, renamedval{3}, renamedval{1}, renamedval{2});
    elseif pedantic
      error(sprintf('use cfg.%s=''%s'' instead of cfg.%s=''%s''', renamedval{1}, renamedval{3}, renamedval{1}, renamedval{2}));
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check for required fields, give error when missing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(required)
  fieldsused = fieldnames(cfg);
  [c, ia, ib] = setxor(required, fieldsused);
  if ~isempty(ia)
    error(sprintf('The field cfg.%s is required\n', required{ia}));
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check for deprecated fields, give warning when present
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(deprecated)
  fieldsused = fieldnames(cfg);
  if any(ismember(deprecated, fieldsused))
    if silent
      % don't mention it
    elseif loose
      warning('The option cfg.%s is deprecated, support is no longer guaranteed\n', deprecated{ismember(deprecated, fieldsused)});
    elseif pedantic
      error(sprintf('The option cfg.%s is deprecated, support is no longer guaranteed\n', deprecated{ismember(deprecated, fieldsused)}));
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check for unused fields, give warning when present and remove them
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(unused)
  fieldsused = fieldnames(cfg);
  if any(ismember(unused, fieldsused))
    cfg = rmfield(cfg, unused(ismember(unused, fieldsused)));
    if silent
      % don't mention it
    elseif loose
      warning('The field cfg.%s is unused, it will be removed from your configuration\n', unused{ismember(unused, fieldsused)});
    elseif pedantic
      error(sprintf('The field cfg.%s is unused\n', unused{ismember(unused, fieldsused)}));
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check for forbidden fields, give error when present
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(forbidden)
  fieldsused = fieldnames(cfg);
  if any(ismember(forbidden, fieldsused))
    cfg = rmfield(cfg, forbidden(ismember(forbidden, fieldsused)));
    if silent
      % don't mention it
    elseif loose
      warning('The field cfg.%s is forbidden, it will be removed from your configuration\n', forbidden{ismember(forbidden, fieldsused)});
    elseif pedantic
      error(sprintf('The field cfg.%s is forbidden\n', forbidden{ismember(forbidden, fieldsused)}));
    end
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% createsubcfg
%
% This collects the optional arguments for some of the low-level
% functions and puts them in a separate substructure. This function is to
% ensure backward compatibility of end-user scripts, fieldtrip functions
% and documentation that do not use the nested detailled configuration
% but that use a flat configuration.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(createsubcfg)
  for j=1:length(createsubcfg)
    subname = createsubcfg{j};

    if isfield(cfg, subname)
      % get the options that are already specified in the substructure
      subcfg = getfield(cfg, subname);
    else
      % start with an empty substructure
      subcfg = [];
    end

    % add all other relevant options to the substructure
    switch subname
      case 'preproc'
        fieldname = {
          'reref'
          'refchannel'
          'implicitref'
          'detrend'
          'bpfiltdir'
          'bpfilter'
          'bpfiltord'
          'bpfilttype'
          'bpfreq'
          'bsfiltdir'
          'bsfilter'
          'bsfiltord'
          'bsfilttype'
          'bsfreq'
          'demean'
          'baselinewindow'
          'denoise'
          'dftfilter'
          'dftfreq'
          'hpfiltdir'
          'hpfilter'
          'hpfiltord'
          'hpfilttype'
          'hpfreq'
          'lpfiltdir'
          'lpfilter'
          'lpfiltord'
          'lpfilttype'
          'lpfreq'
          'medianfilter'
          'medianfiltord'
          'hilbert'
          'derivative'
          'rectify'
          'boxcar'
          'absdiff'
          };

      case 'grid'
        fieldname = {
          'xgrid'
          'ygrid'
          'zgrid'
          'resolution'
          'filter'
          'leadfield'
          'inside'
          'outside'
          'pos'
          'dim'
          'tight'
          };

      case 'dics'
        fieldname = {
          'feedback'
          'fixedori'
          'keepfilter'
          'keepmom'
          'lambda'
          'normalize'
          'normalizeparam'
          'powmethod'
          'projectnoise'
          'reducerank'
          'keepcsd'
          'realfilter'
          'subspace'
          'keepsubspace'
          };

      case 'lcmv'
        fieldname = {
          'feedback'
          'fixedori'
          'keepfilter'
          'keepmom'
          'lambda'
          'normalize'
          'normalizeparam'
          'powmethod'
          'projectnoise'
          'projectmom'
          'reducerank'
          'keepcov'
          'subspace'
          'keepsubspace'
          };

      case 'pcc'
        fieldname = {
          'feedback'
          'keepfilter'
          'keepmom'
          'lambda'
          'normalize'
          'normalizeparam'
          %'powmethod'
          'projectnoise'
          'reducerank'
          'keepcsd'
          'realfilter'
          };

      case {'mne', 'loreta', 'rv'}
        fieldname = {
          'feedback'
          };

      case 'music'
        fieldname = {
          'feedback'
          'numcomponent'
          };

      case 'sam'
        fieldname = {
          'meansphereorigin'
          'spinning'
          'feedback'
          'lambda'
          'normalize'
          'normalizeparam'
          'reducerank'
          };

      case 'mvl'
        fieldname = {};

      otherwise
        error('unexpected name of the subfunction');
        fieldname = {};

    end % switch subname

    for i=1:length(fieldname)
      if ~isfield(subcfg, fieldname{i}) && isfield(cfg, fieldname{i})
        subcfg = setfield(subcfg, fieldname{i}, getfield(cfg, fieldname{i}));  % set it in the subconfiguration
        cfg = rmfield(cfg, fieldname{i});                                      % remove it from the main configuration
      end
    end

    % copy the substructure back into the main configuration structure
    cfg = setfield(cfg, subname, subcfg);
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dataset2files
%
% Converts cfg.dataset into cfg.headerfile and cfg.datafile if neccessary.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(dataset2files) && strcmp(dataset2files, 'yes')

  % start with empty fields if they are not present
  if ~isfield(cfg, 'dataset')
    cfg.dataset = [];
  end
  if ~isfield(cfg, 'datafile')
    cfg.datafile = [];
  end
  if ~isfield(cfg, 'headerfile')
    cfg.headerfile = [];
  end

  if ~isempty(cfg.dataset)
    if strcmp(cfg.dataset, 'gui');
      [f, p] = uigetfile('*.*', 'Select a file');
      if isequal(f, 0)
        error('User pressed cancel');
      else
        d = fullfile(p, f);
      end
      cfg.dataset = d;
    end

    % the following code is shared with fileio read_header/read_data
    % therefore the three local variables are used outside of the cfg
    filename   = cfg.dataset;
    datafile   = [];
    headerfile = [];
    switch ft_filetype(filename)
      case '4d_pdf'
        datafile   = filename;
        headerfile = [datafile '.m4d'];
        sensorfile = [datafile '.xyz'];
      case {'4d_m4d', '4d_xyz'}
        datafile   = filename(1:(end-4)); % remove the extension
        headerfile = [datafile '.m4d'];
        sensorfile = [datafile '.xyz'];
      case '4d'
        [path, file, ext] = fileparts(filename);
        datafile   = fullfile(path, [file,ext]);
        headerfile = fullfile(path, [file,ext]);
        configfile = fullfile(path, 'config');
      case {'ctf_ds', 'ctf_old'}
        % convert CTF filename into filenames
        [path, file, ext] = fileparts(filename);
        if any(strcmp(ext, {'.res4' '.meg4', '.1_meg4' '.2_meg4' '.3_meg4' '.4_meg4' '.5_meg4' '.6_meg4' '.7_meg4' '.8_meg4' '.9_meg4'}))
          filename = path;
          [path, file, ext] = fileparts(filename);
        end
        if isempty(path) && isempty(file)
          % this means that the dataset was specified as the present working directory, i.e. only with '.'
          filename = pwd;
          [path, file, ext] = fileparts(filename);
        end
        headerfile = fullfile(filename, [file '.res4']);
        datafile   = fullfile(filename, [file '.meg4']);
        if length(path)>3 && strcmp(path(end-2:end), '.ds')
          filename = path; % this is the *.ds directory
        end
      case 'ctf_meg4'
        [path, file, ext] = fileparts(filename);
        if isempty(path)
          path = pwd;
        end
        headerfile = fullfile(path, [file '.res4']);
        datafile   = fullfile(path, [file '.meg4']);
        if length(path)>3 && strcmp(path(end-2:end), '.ds')
          filename = path; % this is the *.ds directory
        end
      case 'ctf_res4'
        [path, file, ext] = fileparts(filename);
        if isempty(path)
          path = pwd;
        end
        headerfile = fullfile(path, [file '.res4']);
        datafile   = fullfile(path, [file '.meg4']);
        if length(path)>3 && strcmp(path(end-2:end), '.ds')
          filename = path; % this is the *.ds directory
        end
      case 'brainvision_vhdr'
        [path, file, ext] = fileparts(filename);
        headerfile = fullfile(path, [file '.vhdr']);
        if exist(fullfile(path, [file '.eeg']))
          datafile   = fullfile(path, [file '.eeg']);
        elseif exist(fullfile(path, [file '.seg']))
          datafile   = fullfile(path, [file '.seg']);
        elseif exist(fullfile(path, [file '.dat']))
          datafile   = fullfile(path, [file '.dat']);
        end
      case 'brainvision_eeg'
        [path, file, ext] = fileparts(filename);
        headerfile = fullfile(path, [file '.vhdr']);
        datafile   = fullfile(path, [file '.eeg']);
      case 'brainvision_seg'
        [path, file, ext] = fileparts(filename);
        headerfile = fullfile(path, [file '.vhdr']);
        datafile   = fullfile(path, [file '.seg']);
      case 'brainvision_dat'
        [path, file, ext] = fileparts(filename);
        headerfile = fullfile(path, [file '.vhdr']);
        datafile   = fullfile(path, [file '.dat']);
      case 'itab_raw'
        [path, file, ext] = fileparts(filename);
        headerfile = fullfile(path, [file '.raw.mhd']);
        datafile   = fullfile(path, [file '.raw']);
      case 'fcdc_matbin'
        [path, file, ext] = fileparts(filename);
        headerfile = fullfile(path, [file '.mat']);
        datafile   = fullfile(path, [file '.bin']);
      case {'tdt_tsq' 'tdt_tev'}
        [path, file, ext] = fileparts(filename);
        headerfile = fullfile(path, [file '.tsq']);
        datafile   = fullfile(path, [file '.tev']);
      case 'nmc_archive_k'
        [path, file, ext] = fileparts(filename);
        headerfile = [path '/' file 'newparams.txt'];
        if isempty(headerformat)
          headerformat = 'nmc_archive_k';
        end
        if isempty(hdr)
          hdr = ft_read_header(headerfile, 'headerformat', headerformat);
        end
        datafile = filename;
      otherwise
        % convert filename into filenames, assume that the header and data are the same
        datafile   = filename;
        headerfile = filename;
    end
    % end sharing with fileio read_header/read_data
    % put everything back into the cfg
    cfg.dataset    = filename;
    cfg.datafile   = datafile;
    cfg.headerfile = headerfile;

    % fill dataformat if unspecified
    if ~isfield(cfg,'dataformat') || isempty(cfg.dataformat)
      cfg.dataformat = ft_filetype(datafile);
    end

    % fill dataformat if unspecified
    if ~isfield(cfg,'headerformat') || isempty(cfg.headerformat)
      cfg.headerformat = ft_filetype(headerfile);
    end

  elseif ~isempty(cfg.datafile) && isempty(cfg.headerfile);
    % assume that the datafile also contains the header
    cfg.headerfile = cfg.datafile;
  elseif isempty(cfg.datafile) && ~isempty(cfg.headerfile);
    % assume that the headerfile also contains the data
    cfg.datafile = cfg.headerfile;
  end
  % remove empty fields (otherwise a subsequent check on required fields doesn't make any sense)
  if isempty(cfg.dataset),    cfg=rmfield(cfg, 'dataset');    end
  if isempty(cfg.headerfile), cfg=rmfield(cfg, 'headerfile'); end
  if isempty(cfg.datafile),   cfg=rmfield(cfg, 'datafile');   end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% configtracking
%
% switch configuration tracking on/off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(trackconfig)
  try
    if strcmp(trackconfig, 'on') && isa(cfg, 'struct')
      % turn ON configuration tracking
      cfg = config(cfg);
      % remember that configtracking has been turned on
      cfg.trkcfgcount = 1;
    elseif strcmp(trackconfig, 'on') && isa(cfg, 'config')
      % remember how many times configtracking has been turned on
      cfg.trkcfgcount = cfg.trkcfgcount+1; % count the 'ONs'
    end

    if strcmp(trackconfig, 'off') && isa(cfg, 'config')
      % turn OFF configuration tracking, optionally give report and/or cleanup
      cfg.trkcfgcount=cfg.trkcfgcount-1; % count(down) the 'OFFs'

      if cfg.trkcfgcount==0 % only proceed when number of 'ONs' matches number of 'OFFs'
        cfg=rmfield(cfg, 'trkcfgcount');

        if strcmp(cfg.trackconfig, 'report') || strcmp(cfg.trackconfig, 'cleanup')
          % gather information about the tracked results
          r = access(cfg, 'reference');
          o = access(cfg, 'original');

          key = fieldnames(cfg);
          key = key(:)';

          ignorefields = {'checksize', 'trl', 'trlold', 'event', 'artifact', 'artfctdef', 'previous'}; % these fields should never be removed!
          skipsel      = match_str(key, ignorefields);
          key(skipsel) = [];

          used     = zeros(size(key));
          original = zeros(size(key));

          for i=1:length(key)
            used(i)     = (r.(key{i})>0);
            original(i) = (o.(key{i})>0);
          end

          if ~silent
            % give report on screen
            fprintf('\nThe following config fields were specified by YOU and were USED\n');
            sel = find(used & original);
            if numel(sel)
              fprintf('  cfg.%s\n', key{sel});
            else
              fprintf('  <none>\n');
            end

            fprintf('\nThe following config fields were specified by YOU and were NOT USED\n');
            sel = find(~used & original);
            if numel(sel)
              fprintf('  cfg.%s\n', key{sel});
            else
              fprintf('  <none>\n');
            end

            fprintf('\nThe following config fields were set to DEFAULTS and were USED\n');
            sel = find(used & ~original);
            if numel(sel)
              fprintf('  cfg.%s\n', key{sel});
            else
              fprintf('  <none>\n');
            end

            fprintf('\nThe following config fields were set to DEFAULTS and were NOT USED\n');
            sel = find(~used & ~original);
            if numel(sel)
              fprintf('  cfg.%s\n', key{sel});
            else
              fprintf('  <none>\n');
            end
          end % report
        end % report/cleanup

        if strcmp(cfg.trackconfig, 'cleanup')
          % remove the unused options from the configuration
          unusedkey = key(~used);
          for i=1:length(unusedkey)
            cfg = rmfield(cfg, unusedkey{i});
          end
        end

        % convert the configuration back to a struct
        cfg = struct(cfg);
      end
    end % off

  catch
    disp(lasterr);
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check the size of fields in the cfg, remove large fields
% the max allowed size should be specified in cfg.checksize (this can be
% set with ft_defaults)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(checksize, 'yes') && ~isinf(cfg.checksize)
  cfg = checksizefun(cfg, cfg.checksize);
end

function [cfg] = checksizefun(cfg, max_size)

ignorefields = {'checksize', 'trl', 'trlold', 'event', 'artifact', 'artfctdef', 'previous'}; % these fields should never be removed!

fieldsorig = fieldnames(cfg);
for i=1:numel(fieldsorig)
  for k=1:numel(cfg)
    if ~isstruct(cfg(k).(fieldsorig{i})) && ~any(strcmp(fieldsorig{i}, ignorefields))
      % find large fields and remove them from the cfg, skip fields that should be ignored
      temp = cfg(k).(fieldsorig{i});
      s = whos('temp');
      if s.bytes>max_size
        cfg(k).(fieldsorig{i}) = 'empty - this was cleared by checkconfig';
      end
      %%% cfg(k).(fieldsorig{i})=s.bytes; % remember the size of each field for debugging purposes
    elseif isstruct(cfg(k).(fieldsorig{i}));
      % run recursively on subfields that are structs
      cfg(k).(fieldsorig{i}) = checksizefun(cfg(k).(fieldsorig{i}), max_size);
    elseif iscell(cfg(k).(fieldsorig{i})) && strcmp(fieldsorig{i}, 'previous')
      % run recursively on 'previous' fields that are cells
      for j=1:numel(cfg(k).(fieldsorig{i}))
        cfg(k).(fieldsorig{i}){j} = checksizefun(cfg(k).(fieldsorig{i}){j}, max_size);
      end
    end
  end
end
