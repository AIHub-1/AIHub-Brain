function mvar = ft_datatype_mvar(mvar, varargin)

% FT_DATATYPE_MVAR describes the FieldTrip MATLAB structure for mvar data
%
% The mvar datatype represents multivariate model estimates in the time- or
% in the frequency-domain. This is usually obtained from FT_MVARANALYSIS,
% optionally in combination with FT_FREQANALYSIS.
%
% The following is an example of sensor level MVAR model data in the time
% domain
%        dimord: 'chan_chan_lag'     defines how the numeric data should be interpreted
%         label: {3x1 cell}          the channel labels
%        coeffs: [3x3x5 double]      numeric data (MVAR model coefficients 3 channels x 3 channels x 5 time lags)
%      noisecov: [3x3 double]        more numeric data (covariance matrix of the noise residuals 3 channels x 3 channels)
%           dof: 500
%   fsampleorig: 200
%           cfg: [1x1 struct]
%
% The following is an example of sensor-level MVAR model data in the frequency
% domain
%        dimord: 'chan_chan_freq'    defines how the numeric data should be interpreted
%         label: {3x1 cell}          the channel labels
%          freq: [1x101 double]      the frequencies, expressed in Hz
%      transfer: [3x3x101 double]
%     itransfer: [3x3x101 double]
%      noisecov: [3x3 double]
%     crsspctrm: [3x3x101 double]
%           dof: 500
%           cfg: [1x1 struct]
%
% Required fields:
%   - label, dimord, freq
%
% Optional fields:
%   - too many to mention
%
% Deprecated fields:
%   - <none>
%
% Obsoleted fields:
%   - <none>
%
% Revision history:
%
% (2008/latest) The initial version was defined.
%
% See also FT_DATATYPE and FT_DATATYPE_xxx

% Copyright (C) 2011, Robert Oostenveld
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
% $Id: ft_datatype_mvar.m 3031 2011-03-01 20:57:20Z roboos $

% get the optional input arguments, which should be specified as key-value pairs
version = keyval('version', varargin); if isempty(version), version = 'latest'; end

if strcmp(version, 'latest')
  version = '2008';
end

switch version
  case '2008'
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % there are no known conversions for backward or forward compatibility support

  otherwise
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    error('unsupported version "%s" for mvar datatype', version);
end

