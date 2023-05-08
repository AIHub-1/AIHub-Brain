function mrk = WAM_20170814_Imag_Arrow( mrko, varargin )
%% Reaching 23~  GIGA
% stimDef= {'S 11',    'S 21',      'S 31',  'S 41',   'S 51',  'S 61' 'S  8';
%          'Forward', 'Backward', 'Left',  'Right',  'Up',    'Down' 'Rest'};
% %       
% Grasping Previous Trigger - ����̱���
% stimDef= {'S 71',  'S 81';
%           'Grasp', 'Open'};
%% Grasping 23~ Modified trigger - �����̺���
% stimDef= {'S 11',  'S 21','S  8';
%           'Grasp', 'Open','Rest'};
% 
%
%% Grasping 180905~ - multiGrasp ���� GIGA
% stimDef= {'S 11', 'S 21', 'S 61', 'S  8';
%          'Cylindrical', 'Spherical', 'Lumbrical', 'Rest'};

%% Twisting Data 1 ~ 22
%  stimDef= {'S 91',  'S101' 'S 92' ;
%            'Left',  'Right' 'Rest'};

%% Twisting Data 23~ GIGA
 stimDef= {'S 91',  'S101' 'S  8';
           'Left',  'Right' 'Rest' };



%% Default
miscDef= {'S 13',    'S 14';
          'Start',   'End'};

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'stimDef', stimDef, ...
                       'miscDef', miscDef);

mrk= mrk_defineClasses(mrko, opt.stimDef);
mrk.misc= mrk_defineClasses(mrko, opt.miscDef);