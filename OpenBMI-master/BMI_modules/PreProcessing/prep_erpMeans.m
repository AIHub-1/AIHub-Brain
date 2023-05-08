function [out] = prep_erpMeans(dat,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prep_erpMeans (Pre-processing procedure):
%
% Synopsis:
%   [out] = prep_erpMeans(dat,<var>)
%
% Example :
%         out = prep_erpMeans(dat,{'nMeans',20})
%                 : 20 mean values for each epoch
%         out = prep_erpMeans(dat,{'nSamples',50})
%                 : calculate means with 50 samples each
%         Two or three options can be used together, but when the both 'nMeans' and
%         'nSamples' are used, 'nMeans' will be ignored.
%
% Arguments:
%     dat - Segmented data itself
% Options:
%     nMeans - The number of means you want to calculate in a single epoch
%     nSamples - The number of samples used in calculating a single mean value
%
% Returns:
%     out - Mean values of eeg signal
%
%
% Description:
%     This function calculates the mean in specified time interval for each
%     epoch.
%     continuous data should be [time * channels]
%     epoched data should be [time * channels * trials]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 01-2018
% mh_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ndims(dat)~=3
    warning('OpenBMI: Data must be segmented. Is the number of channel 1?');
end
if isempty(varargin)
    warning('OpenBMI: Whole samples in each trial are averaged')
    out = mean(dat,1);
    return
end
opt = opt_cellToStruct(varargin{:});


if isfield(opt,'nMeans') && isscalar(opt.nMeans)
    m = opt.nMeans;
    s = floor(size(dat,1)/m);
    opt.nSamples = s;
elseif isfield(opt,'nSamples') && isscalar(opt.nSamples)
    s = opt.nSamples;
    m = floor(size(dat,1)/s);
    opt.nMeans = m;
else
    warning('OpenBMI: ''nMeans'' and ''nSamples'' should be a scalar');return
end
x = zeros(m,size(dat,2),size(dat,3));
for j=1:m
    x(j,:,:) = mean(dat([(j-1)*s+1:j*s],:,:),1);
end

out = x;

