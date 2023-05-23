function write_brainvision_eeg(filename, hdr, dat);

% WRITE_BRAINVISION_EEG exports continuous EEG data to a BrainVision *.eeg
% and corresponding *.vhdr file. The samples in the exported file are
% multiplexed and stored in ieee-le float32 format.
%
% Use as
%   write_brainvision_eeg(filename, hdr, dat)
%
% See also READ_BRAINVISION_EEG, READ_BRAINVISION_VHDR, READ_BRAINVISION_VMRK

% Copyright (C) 2007, Robert Oostenveld
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
% $Id: write_brainvision_eeg.m 2885 2011-02-16 09:41:58Z roboos $

if length(size(dat))>2
  ntrl  = size(dat,1);
  nchan = size(dat,2);
  nsmp  = size(dat,3);
else
  nchan = size(dat,1);
  nsmp  = size(dat,2);
end

if hdr.nChans~=nchan
  error('number of channels in in header does not match with the data');
end

% this is the only supported data format
hdr.DataFormat      = 'BINARY';
hdr.DataOrientation = 'MULTIPLEXED';
hdr.BinaryFormat    = 'IEEE_FLOAT_32';
hdr.resolution      = ones(size(hdr.label));  % no additional calibration needed, since float32

% determine the filenames
[p, f, x] = fileparts(filename);
headerfile = fullfile(p, [f '.vhdr']);
datafile   = fullfile(p, [f '.eeg']);
markerfile = '';

% open the data file and write the binary data
fid = fopen(datafile, 'wb', 'ieee-le');
if length(size(dat))>2
  warning('writing segmented data as if it were continuous');
  for i=1:ntrl
    fwrite(fid, squeeze(dat(i,:,:)), 'float32');
  end
else
  fwrite(fid, dat, 'float32');
end

fclose(fid);

% open the header file and write the ascii header information
fid = fopen(headerfile, 'wb');
fprintf(fid, 'Brain Vision Data Exchange Header File Version 1.0\r\n');
fprintf(fid, '; Data created by FieldTrip\r\n');
fprintf(fid, '\r\n');
fprintf(fid, '[Common Infos]\r\n');
fprintf(fid, 'DataFile=%s\r\n',          datafile);
if ~isempty(markerfile)
  fprintf(fid, 'MarkerFile=%s\r\n',      markerfile);
end
fprintf(fid, 'DataFormat=%s\r\n',        hdr.DataFormat);
fprintf(fid, 'DataOrientation=%s\r\n',   hdr.DataOrientation);
fprintf(fid, 'NumberOfChannels=%d\r\n',  hdr.nChans);
% Sampling interval in microseconds
fprintf(fid, 'SamplingInterval=%d\r\n',  round(1e6/hdr.Fs));
fprintf(fid, '\r\n');
fprintf(fid, '[Binary Infos]\r\n');
fprintf(fid, 'BinaryFormat=%s\r\n',      hdr.BinaryFormat);
fprintf(fid, '\r\n');
fprintf(fid, '[Channel Infos]\r\n');
% Each entry: Ch<Channel number>=<Name>,<Reference channel name>,<Resolution in microvolts>,<Future extensions>...
% Fields are delimited by commas, some fields might be omitted (empty).
% Commas in channel names should be coded as "\1", but are not supported here
for i=1:hdr.nChans
  fprintf(fid, 'Ch%d=%s,,%g\r\n', i, hdr.label{i}, hdr.resolution(i));
end
fclose(fid);

