function grp_plots = vis_freqFFTPlot(SMT, varargin)


opt = [varargin{:}];
if ~isstruct(opt) && iscell(opt)
    opt = opt_cellToStruct(opt);
end

%% Options
if ~isfield(opt, 'Channels') opt.Channels = {SMT.chan{1}}; end
if ~isfield(opt, 'Class') opt.Class = {SMT.class{1,2}}; end
if ~isfield(opt, 'SelectTime') opt.SelectTime = [SMT.ival(1) SMT.ival(end)]; end
if ~isfield(opt, 'Plots') opt.Plots = gca; end

grp_plots = opt.Plots;

co = [0 0.4470 0.7410;
    0.8500 0.3250 0.0980;
    0.9290 0.6940 0.1250;
    0.4940 0.1840 0.5560;
    0.4660 0.6740 0.1880;
    0.3010 0.7450 0.9330;
    0.6350 0.0780 0.1840];

%% TODO: ���� ���⼭...? �ٸ� ��� ������?
SMT = prep_selectChannels(SMT, {'Name', opt.Channels});
%%
[YfreqDomain, frequencyRange] = positiveFFT(SMT.x, SMT.fs);

mask = frequencyRange >= 0;

YfreqDomain = abs(YfreqDomain(mask, :));
frequencyRange = frequencyRange(mask);

freqXLIM = [frequencyRange(1) 50];
freqYLIM = [0 max(reshape(YfreqDomain, 1, []))];

idx = 1;
for ch_num = 1:length(opt.Channels)
    for cl = 1:length(opt.Class)
        plot(grp_plots(idx), frequencyRange,YfreqDomain(:,cl),'Color', co(cl, :)); hold on;
        
        legend(grp_plots(idx), SMT.class(cl,2), 'Interpreter', 'none', 'AutoUpdate', 'off');
        
        grid(grp_plots(idx),'on');
        
        xlim(grp_plots(idx), freqXLIM);
        ylim(grp_plots(idx), freqYLIM);
        
        ylabel(grp_plots(idx),SMT.chan{ch_num}, 'Rotation', 90, 'FontWeight', 'normal', 'FontSize', 12);
        idx = idx + 1;
    end
end
end

function [X,freq]=positiveFFT(x,Fs)
N=length(x);    % get the number of points
k=0:N-1;        % create a vector from 0 to N-1
T=N/Fs;         % get the frequency interval
freq=k/T;       % create the frequency range
X=fft(x)/N*2;   % normalize the data
cutOff = ceil(N/2);

% take only the first half of the spectrum
X = X(1:cutOff, :);
freq = freq(1:cutOff);
end
