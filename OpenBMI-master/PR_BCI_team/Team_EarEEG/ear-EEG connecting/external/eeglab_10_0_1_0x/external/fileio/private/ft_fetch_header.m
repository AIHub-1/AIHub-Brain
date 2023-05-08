function [hdr] = ft_fetch_header(data)

% FT_FETCH_HEADER mimics the behaviour of FT_READ_HEADER, but for a FieldTrip
% raw data structure instead of a file on disk.
%
% Use as
%   [hdr] = ft_fetch_header(data)
%
% See also FT_READ_HEADER, FT_FETCH_DATA, FT_FETCH_EVENT

% Copyright (C) 2008, Esther Meeuwissen
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
% $Id: ft_fetch_header.m 2885 2011-02-16 09:41:58Z roboos $

% check whether input is data
data = ft_checkdata(data, 'datatype', 'raw', 'hastrialdef', 'yes');

trlnum = length(data.trial);
trllen = zeros(trlnum,1);
for trllop=1:trlnum
  trllen(trllop) = size(data.trial{trllop},2);
end

% try to get trial definition according to original data file
if isfield(data, 'sampleinfo')
  trl = data.sampleinfo;
else
  trl = [1 sum(trllen)];
end

% fill in hdr.nChans 
hdr.nChans = length(data.label);

% fill in hdr.label 
hdr.label = data.label;

% fill in hdr.Fs (sample frequency)
hdr.Fs = data.fsample;

% determine hdr.nSamples, hdr.nSamplesPre, hdr.nTrials
% always pretend that it is continuous data
hdr.nSamples    = max(trl(:,2));
hdr.nSamplesPre = 0;
hdr.nTrials     = 1;

% fill in hdr.grad or hdr.elec
if isfield(data, 'grad')
  hdr.grad=data.grad;
elseif isfield(data, 'elec')
  hdr.elec=data.elec;
end
