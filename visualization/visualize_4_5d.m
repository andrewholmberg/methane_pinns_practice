clear;

loadedData = load('C:/Users/andyh/OneDrive/Documents/MATLAB');

varName = 'myVariable';
variableNames = fieldnames(loadedData);
for i = 1:length(variableNames)
    varName = variableNames{i};
    eval([varName ' = loadedData.(varName);']);
end

% Create figure and initial plot
fig = figure;
ax = axes('Parent', fig);
h = surf(ax, X2(:,:,1), X3(:,:,1), U(:,:,1));
title(ax, '3D Surface Plot');
xlabel(ax, 'X2');
ylabel(ax, 'X3');
zlabel(ax, 'U');

% Set constant axis limits


xlim(ax, [0,2]);
ylim(ax, [0,2]);
zlim(ax, [0,2]);


% Add a slider
slider = uicontrol('Style', 'slider', 'Min', 1, 'Max', size(U, 3), 'Value', 1, ...
    'SliderStep', [1/(size(U, 3)-1) , 10/(size(U, 3)-1)], 'Units', 'normalized', 'Position', [0.2 0.02 0.6 0.03]);

% Add a label to display the current slice index
sliderLabel = uicontrol('Style', 'text', 'Units', 'normalized', ...
    'Position', [0.82 0.02 0.1 0.03], 'String', 'Time: 0');

% Add listener to the slider
addlistener(slider, 'ContinuousValueChange', @(src, event) updatePlot(src, event,X1, X2, X3, U, h, ax, sliderLabel,fig));

% Define the updatePlot function
function updatePlot(src, event, X1, X2, X3, U, h, ax, sliderLabel,fig)
    un = unique(X1)
    val = round(src.Value);
    U_data = U(:,:,val);
    set(h, 'ZData', U_data);
    % Ensure the axes limits are constant
    xlim(ax, [min(X2(:)), max(X2(:))]);
    ylim(ax, [min(X3(:)), max(X3(:))]);
    zlim(ax, [min(U(:)), max(U(:))]);
    % Update the slider label
    set(sliderLabel, 'String', ['Time: ',num2str(un(val)) ]);
    saveas(fig,['D:\andyh\Documents\Projects\mines\methane_project\pinns_practice',num2str(val),'.jpg'])
    drawnow;
end


% Keep the figure open
un=unique(X1)
un(1)
uiwait(fig);