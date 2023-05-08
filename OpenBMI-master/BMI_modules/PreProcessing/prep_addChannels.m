function [out] = prep_addChannels(dat1, dat2, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prep_addChannels (Pre-processing procedure):
%
% Synopsis:
%     [out] = prep_addChannels(DAT,DAT2,<OPT>)
%
% Example :
%     out = prep_addChannels(dat1,dat2,{'Name',{'C3','C4'}})
%     out = prep_addChannels(dat1,dat2,{'Index',[25,29]})
%
% Arguments:
%     dat1 - Data structure, continuous or epoched 
%     dat2 - Data structure, continuous or epoched (NOTE: specified channels should be included)
%     varargin - struct or property/value list of optional properties:
%           'Name'- Cell type. channel name in dat2 to be added to dat1
%           'Index'- channel index in dat2 to be added to dat1
%
% Returns:
%     out - Data structure which channels are added (continuous or epoched)
%
%
% Description:
%     Add specific channels to the former data(dat1) from dat2
%     continuous data should be [time * channels]
%     epoched data should be [time * channels * trials]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 12-2017
% mh_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(varargin)
    warning('OpenBMI: Data of all channels from the latter data will be added to the former data')
    opt.Name = dat2.chan;
else
    opt = opt_cellToStruct(varargin{:});
end

if isfield(opt,'Name') && isfield(opt,'Index')
    if find(ismember(dat2.chan,opt.Name))~=opt.Index
        warning('OpenBMI: Mismatch between name and index of channels')
        return
    end
    ch = opt.Name;
    ch_idx = opt.Index;
elseif isfield(opt,'Name') && ~isfield(opt,'Index')
    ch = opt.Name;
    ch_idx = find(ismember(dat2.chan,ch));
    if size(ch_idx)~=size(ch,2)
        warning('OpenBMI: Error in ''Name''')
        return
    end
elseif ~isfield(opt,'Name') && isfield(opt,'Index')
    ch_idx = opt.Index;
    ch = dat2.chan(ch_idx);
else
    warning('OpenBMI: Channels should be specified in a correct form')
    return
end

if ~isfield(dat1,'x') || ~isfield(dat2,'x')
    warning('OpenBMI: Data structure must have a field named ''x''')
    return
end
if ~isfield(dat1,'chan') || ~isfield(dat2,'chan')
    warning('OpenBMI: Data structure must have a field named ''chan''')
    return
end

s1=size(dat1.x);s2=size(dat2.x);
if s1(1:end-1)~=s2(1:end-1)
    warning('OpenBMI: Unmatched data size')
    return
end

out = rmfield(dat1,{'x','chan'});

ch_ori = dat1.chan;
idx = find(ismember(ch,ch_ori));
ch(idx)=[];
ch_idx(idx)=[];

out.chan = cat(2,dat1.chan,ch);
d1 = ndims(dat1.x);
if d1 == 2
    x = dat2.x(:,ch_idx);
elseif d1 == 3
    x = dat2.x(:,:,ch_idx);
else
    warning('Check for the dimension of input data')
end
out.x = cat(d1,dat1.x,x);
