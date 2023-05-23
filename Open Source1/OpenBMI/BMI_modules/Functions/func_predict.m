function [ cf_out ] = func_predict( in, varargin )
% func_predict:
%     Predicting the features of data based on trained classifier.
%     In this version, only producing the lda classifier. Other classifier algorithm will be updated.
%     Also, finding func_train.
% 
% Example:
%     func_predict(fv, {'classifier','lda'};
%
% Input:
%     in- Data set of test 
%
% Options:
%     classifier - setting the classifier
%
% Retuns:
%     cf_out - result of testing output

if isstruct(varargin{:})
    opt=varargin{:};    
else
    opt=varargin{:}{:}; % cross-validation procedure
end

if isstruct(in)
dat=in.x;
else
    dat=in;
end
switch lower(opt.classifier)
    case 'lda'
        [ntri ndim]=size(dat);
%         if ntri~=1
%            dat=dat';
%         end
        if ndim ~= size(opt.cf_param.w,1) && ntri == size(opt.cf_param.w, 1)
            dat = dat';
        end
            
        cf_out= real( dat*opt.cf_param.w+repmat(opt.cf_param.b', size(dat,1),1)); %% compatibility for matlab 2014b
end

end

