function [ Acc ] = mi_performance(CNT,params,Niteration)
opt = opt_cellToStruct(params);
SMT=[];
% Pre-processing
for idx=1:2
    CNTch = prep_selectChannels(CNT{idx}, {'Index', opt.channel_index});
    CNTchfilt =prep_filter(CNTch , {'frequency', opt.band});
    all_SMT = prep_segmentation(CNTchfilt, {'interval', opt.time_interval});
    if idx==1
        smt = all_SMT;
        clear all_SMT
    else
        SMT = prep_addTrials(smt, all_SMT);
    end
end

% Feature extracion and Classification
for iter=1:Niteration
    CV.train={
        '[SMT, CSP_W, CSP_D]=func_csp(SMT,{"nPatterns", [2]})'
        'FT=func_featureExtraction(SMT, {"feature","logvar"})'
        '[CF_PARAM]=func_train(FT,{"classifier","LDA"})'
        };
    CV.test={
        'SMT=func_projection(SMT, CSP_W)'
        'FT=func_featureExtraction(SMT, {"feature","logvar"})'
        '[cf_out]=func_predict(FT, CF_PARAM)'
        };
    CV.option={
        'KFold','10'
        };
    [loss]=eval_crossValidation(SMT, CV);
    iter_result(1,iter)=1-loss;
end
Acc = mean(iter_result);
clear iter_result
end

