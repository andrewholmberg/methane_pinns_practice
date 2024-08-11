clear;

% Load data
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

% Add slider for X1
sliderX1 = uicontrol('Style', 'slider', 'Min', 1, 'Max', size(U, 4), 'Value', 1, ...
    'SliderStep', [1/(size(U, 4)-1) , 10/(size(U, 4)-1)], 'Units', 'normalized', 'Position', [0.2 0.02 0.6 0.03]);

% Add a label to display the current slice index for X1
sliderLabelX1 = uicontrol('Style', 'text', 'Units', 'normalized', ...
    'Position', [0.82 0.02 0.1 0.03], 'String', 'Time: 0');

% Add slider for X4
sliderX4 = uicontrol('Style', 'slider', 'Min', 1, 'Max', size(U, 4), 'Value', 1, ...
    'SliderStep', [1/(size(U,4)-1) , 10/(size(U, 4)-1)], 'Units', 'normalized', 'Position', [0.2 0.06 0.6 0.03]);

% Add a label to display the current slice index for X4
sliderLabelX4 = uicontrol('Style', 'text', 'Units', 'normalized', ...
    'Position', [0.82 0.06 0.1 0.03], 'String', 'X4: 0');

% Add listeners to the sliders
addlistener(sliderX1, 'ContinuousValueChange', @(src, event) updatePlot(src, sliderX4, X1, X2, X3, X4, U, h, ax, sliderLabelX1, sliderLabelX4, fig));
addlistener(sliderX4, 'ContinuousValueChange', @(src, event) updatePlot(sliderX1, src, X1, X2, X3, X4, U, h, ax, sliderLabelX1, sliderLabelX4, fig));

% Define the updatePlot function
function updatePlot(srcX1, srcX4, X1, X2, X3, X4, U, h, ax, sliderLabelX1, sliderLabelX4, fig)
    valX1 = round(srcX1.Value);
    valX4 = round(srcX4.Value);
    U_data = U(:,:,valX1); % Adjust as per the dimension you want to vary
    set(h, 'ZData', U_data);
    % Ensure the axes limits are constant
    xlim(ax, [min(X2(:)), max(X2(:))]);
    ylim(ax, [min(X3(:)), max(X3(:))]);
    zlim(ax, [min(U(:)), max(U(:))]);
    % Update the slider labels
    set(sliderLabelX1, 'String', ['Time: ', num2str(X1(valX1))]);
    set(sliderLabelX4, 'String', ['X4: ', num2str(X4(valX4))]);
    % Save the figure as an image
    saveas(fig,['D:\andyh\Documents\Projects\mines\methane_project\pinns_practice\visualization\20240810_methane_gif\images\figure_X1_',num2str(valX1),'_X4_',num2str(valX4),'.jpg'])
    drawnow;
end

% Keep the figure open
uiwait(fig);
