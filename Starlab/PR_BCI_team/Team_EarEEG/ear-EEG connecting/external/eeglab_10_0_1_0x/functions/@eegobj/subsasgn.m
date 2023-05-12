function this = subsasgn(this,index,c)

    if isempty(this)
        this = eegobj;
    end;
        
    % mulitple dataset
    if strcmpi(index(1).type, '()') && length(index) == 1 % dataset assignement
        if isempty(c) % suppression
            this(index.subs{1}) = [];
        else
            allfields = fieldnames( c );
            this(index.subs{1}) = eeg_emptyset;
            
            % dealing with input structure
            if isa(c, 'eegobj')
                c2 = struct(c);
                c  = c2(1).EEG;
                fieldorder = fieldnames(c);
                for cIndexe = 2:length(c2)
                    c(cIndexe) = orderfields(c2(cIndexe).EEG, fieldorder);
                end;
            end;
            
            for i=1:length( allfields )
                for j = 1:length(index.subs{1})
                    this(index.subs{1}(j)).EEG.(allfields{i}) = c(min(j, length(c))).(allfields{i});
                end;
                %this(index.subs(1)).EEG = setfield(this(index.subs(1)).EEG, getfield(c, allfields{i}), allfields{i});
                %eval( ['this(' int2str(index.subs{1}) ').' allfields{i} ' = c.' allfields{i} ';' ]);
            end;	
            %if ~isfield(c, 'datfile') & isfield(this, 'datfile')
            %    this(index.subs{1}).datfile = '';
            %end;
        end;
    elseif strcmpi(index(1).type, '()')
        if length(index(1).subs{1}) > 1
            error('Unsuported object feature - a.field or a([x y]).field is not supported for object arrays');
        elseif length(this) < index(1).subs{1}
            this(index(1).subs{1}) = eegobj;
        end;
        this(index(1).subs{1}).EEG = builtin('subsasgn', this(index(1).subs{1}).EEG, index(2:end), c);
    elseif strcmpi(index(1).type, '.')
        if length(this) > 1
            error('Unsuported object feature - a.field or a([x y]).field is not supported for object arrays');
        else
            this.EEG = builtin('subsasgn', this.EEG, index, c);
        end;
    end;
