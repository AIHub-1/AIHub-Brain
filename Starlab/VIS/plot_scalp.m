%% Function plot_scalp
function plot_scalp(ax, w, MNT, p_range, resolution)
line_width = 1;

center = [0 0];
theta = linspace(0,2*pi,360);

x = cos(theta)+center(1);
y = sin(theta)+center(2);

xe = MNT.x'; ye = MNT.y';

maxrad = max(1,max(max(abs(xe)),max(abs(ye))));

xx = linspace(-maxrad, maxrad, resolution);
yy = linspace(-maxrad, maxrad, resolution)';
% ----------------------------------------------------------------------
% contour plot
[xg,yg,zg] = griddata(xe, ye, w, xx, yy, 'v4');
mask = ~(sqrt(xg.^2+yg.^2)<=maxrad);
zg(mask)=NaN;

contourf(ax, xg, yg, zg, 50, 'LineStyle','none');
hold(ax,'on');


%% TODO: patch ����... �̰� �� �ʿ��� �Լ��ΰ�... 2014b ȣȯ���� ����
patch(ax, [0 0], [0 0], [1 2]);
ccc = get(ax, 'children');
set(ccc(1), 'Visible', 'off');
%%
set(ax, 'CLim', p_range);
% ----------------------------------------------------------------------
% contour line
p_min = p_range(1);
p_max = p_range(2);

cl = linspace(p_min, p_max, 7);
cl = cl(2:end-1);

contour(ax, xg, yg, zg, cl, 'k-');
% ----------------------------------------------------------------------
% disp electrodes
plot(ax, xe, ye, 'k.', 'MarkerSize', 1, 'LineWidth', 0.2); hold on;
set(0,'defaultfigurecolor',[1 1 1])

% ----------------------------------------------------------------------
% H = struct('ax', ax);
% set(gcf,'CurrentAxes',ax);
% ----------------------------------------------------------------------
% nose plot
nose = [1 1.2 1];
nosi = [83 90 97]+1;
% nose = plot(nose.*x(nosi), nose.*y(nosi), 'k', 'linewidth', line_width );
plot(ax, nose.*x(nosi), nose.*y(nosi), 'k', 'linewidth', line_width );

hold(ax, 'on');

% ----------------------------------------------------------------------
% ears plot
earw = .08; earh = .3;
% H.ears(1) = plot(x*earw-1-earw, y*earh, 'k', 'linewidth', line_width);
% H.ears(2) = plot(x*earw+1+earw, y*earh, 'k', 'linewidth', line_width);
plot(ax, x*earw-1-earw, y*earh, 'k', 'linewidth', line_width);
plot(ax, x*earw+1+earw, y*earh, 'k', 'linewidth', line_width);

hold(ax, 'on');

% ----------------------------------------------------------------------
% main circle plot
plot(ax, x,y, 'k');
set(ax, 'xTick',[], 'yTick',[]);
axis(ax, 'xy', 'tight', 'equal', 'tight');

hold(ax, 'on');

axis(ax, 'off');
set(get(ax,'XLabel'),'Visible','on')
end