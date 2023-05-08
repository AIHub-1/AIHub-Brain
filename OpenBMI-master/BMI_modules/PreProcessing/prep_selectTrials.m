function [out] = prep_selectTrials(dat,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prep_selectTrials (Pre-processing procedure):
%
% Synopsis:
%     [out] = prep_selectTrials(DAT,<OPT>)
%
% Example :
%     out = prep_selectTrials(dat,{'Index',[20:35]});
%
% Arguments:
%     dat - Structure. epoched data
%         - Data which trials are to be selected   
%     varargin - struct or property/value list of optional properties:
%          :  index - index of trials to be selected
%           
% Returns:
%     out - Data structure which has selected channels from epoched data
%
% Description:
%     This function selects data of specified trials 
%     from  epoched data.
%     continuous data should be [time * channels]
%     epoched data should be [time * channels * trials]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 12-2017
% mh_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(varargin)
    warning('OpenBMI: Trials should be specified');
    out = dat;
    return
end
if iscell(varargin{:})
    opt=opt_cellToStruct(varargin{:});
elseif isstruct(varargin{:}) % already structure(x-validation)
    opt=varargin{:}
end

if ~isfield(dat, 'x')
    warning('OpenBMI: Data structure must have a field named ''x''');
    return
end
if ~isfield(dat, 't')
    warning('OpenBMI: Data structure must have a field named ''t''');
    return
end
if ~isfield(dat, 'y_dec') || ~isfield(dat, 'y_logic') || ~isfield(dat, 'y_class')
    warning('OpenBMI: Data structure must have a field named ''y_dec'',''y_logic'',''y_class''');
    return
end

idx = opt.Index;
nd = ndims(dat.x);
if nd == 3
    x = dat.x(:,idx,:);
elseif nd==2
    if size(dat.chan,2)==1
        x = dat.x(:,idx);
        warning('OpenBMI: just 1 channel data?')
    else
        x=dat.x;
    end
elseif nd ==1
    x = dat.x;
else
    warning('OpenBMI: Check for the data dimensionality')
    return
end
out = rmfield(dat,{'x','t','y_dec','y_logic','y_class'});
out.x = x;
out.t = dat.t(idx);
out.y_dec = dat.y_dec(idx);
out.y_logic = dat.y_logic(:,idx);
out.y_class = dat.y_class(idx);
