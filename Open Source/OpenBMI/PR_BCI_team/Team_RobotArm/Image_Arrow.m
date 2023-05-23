function mrk = Imag_Arrow( mrko, varargin )

%classification trigger from stimulation
%the stifDef should have 4 digit
stimDef={'S  1','S  2','S  3';
          'class1', 'class2', 'class3'};

% Default
miscDef= {'S  4',    'S  5';
          'Start',   'End'};

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'stimDef', stimDef, ...
                       'miscDef', miscDef);

mrk= mrk_defineClasses(mrko, opt.stimDef);
mrk.misc= mrk_defineClasses(mrko, opt.miscDef);
