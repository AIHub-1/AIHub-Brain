% pop_apreclust() - Calculates pairwise similarity matrices for select EEG
%                   measures (erp, ersp..) to be used (later) in MP clustering.
%                   Pre-clustering results are placed under STUDY.preclust.similarity 
%                   field. This functions calls std_apreclust() internally.
%
% Usage:
%     >> STUDY = pop_apreclust(STUDY, ALLEEG) % popup window
%
% See also:  std_apreclust(), std_apcluster(), pop_apcluster()
% 
% Author: Nima Bigdely-Shamlo, SCCN/INC/UCSD, 2009

function [STUDY, ALLEEG, command] = pop_apreclust(STUDY, ALLEEG)
% command is used for keeping a history.
returnedFromGui = inputgui( 'geometry', { 1 1 [1 2 1] [1 2 1] [1 2 1] [1 2 1] [1 2 1] [1  2  1] 1 1 }, ...
    'geomvert', [], 'uilist', { ...
    { 'style', 'text', 'string', [ 'Select measures for Affinity Product pre-clustering:' ] }, {}, ...
    {},{ 'Style', 'checkbox', 'string' 'Equiv. dipoles' 'tag' 'scale' 'value' 1} , {}, ...
    {},{ 'Style', 'checkbox', 'string' 'ERPs' 'tag' 'scale' 'value' 1}, {},...
    {},{ 'Style', 'checkbox', 'string' 'ERSPs' 'tag' 'scale' 'value' 1}, {},...
    {},{ 'Style', 'checkbox', 'string' 'ITCs' 'tag' 'scale' 'value' 0}, {},...
    {},{ 'Style', 'checkbox', 'string' 'Mean spectra' 'tag' 'scale' 'value' 0}, {}, ...
    {}, { 'Style', 'checkbox', 'string' 'Scalp maps' 'tag' 'scale' 'value' 0}, {},...
    {}, { 'Style', 'checkbox', 'string' 'Re-calculate All' 'tag' 'scale' 'value' 0}, ...
    }, 'helpcom','pophelp(''pop_apreclust'');', 'title', 'AF pre-clustering -- pop_apreclust()');

if isempty(returnedFromGui) % an empty returnedFromGui means the Cancel button has been pressed so nothing should be done.
    command = '';
    return; % Cancel button is pressed, so do nothing.
else
    
    answers = cell2mat(returnedFromGui);
    measureNamesInGUIorder = {'dipole', 'erp', 'ersp', 'itc', 'spec', 'map'};
    
    measuresToUseInClustering = measureNamesInGUIorder(find(answers(1:end-1))); %#ok<FNDSB>
    reCalculateAll = answers(end);
    STUDY = std_apreclust(STUDY,ALLEEG, measuresToUseInClustering, reCalculateAll);
    
    % prepare 'command' variable for placing both in eeglab histry (accessible with eegh() ) and also
    % adding to  STUDY.history
    
    measuresInOneString = [];
    for i=1:length(measuresToUseInClustering)
        if i>1
            measuresInOneString = [measuresInOneString ' , ' '''' measuresToUseInClustering{i} ''''];
        else
            measuresInOneString = ['''' measuresToUseInClustering{1} ''''];
        end;
    end;
    
    command = ['[STUDY ALLEEG]= std_apreclust(STUDY,ALLEEG, {' measuresInOneString '} , ' num2str(reCalculateAll) ');'];
    STUDY.history =  sprintf('%s\n%s',  STUDY.history, command);
end;
