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
fig = uifigure;  % Use uifigure to allow UI components
ax = uiaxes('Parent', fig);

% Set the position of the graph within the figure
ax.Position = [100, 150, 400, 300];  % [left, bottom, width, height]


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
sliderX1 = uislider(fig, 'Limits', [1, size(U, 4)], 'Value', 1, ...
    'Position', [100, 50, 300, 3]);

% Add a label to display the current slice index for X1
sliderLabelX1 = uilabel(fig, 'Position', [410, 50, 100, 22], 'Text', 'Time: 0');

% Add slider for X4
sliderX4 = uislider(fig, 'Limits', [1, size(U, 3)], 'Value', 1, ...
    'Position', [100, 100, 300, 3]);

% Add a label to display the current slice index for X4
sliderLabelX4 = uilabel(fig, 'Position', [410, 100, 100, 22], 'Text', 'X4: 0');

% Add listeners to the sliders
sliderX1.ValueChangedFcn = @(src, event) updatePlot(sliderX1, sliderX4, X1, X2, X3, X4, U, h, ax, sliderLabelX1, sliderLabelX4, fig);
sliderX4.ValueChangedFcn = @(src, event) updatePlot(sliderX1, sliderX4, X1, X2, X3, X4, U, h, ax, sliderLabelX1, sliderLabelX4, fig);

% Define the updatePlot function
function updatePlot(sliderX1, sliderX4, X1, X2, X3, X4, U, h, ax, sliderLabelX1, sliderLabelX4, fig)
    valX1 = round(sliderX1.Value);
    valX4 = round(sliderX4.Value);
    U_data = U(:,:,valX4, valX1); % Adjust as per the dimension you want to vary
    set(h, 'ZData', U_data);
    % Ensure the axes limits are constant
    xlim(ax, [min(X2(:)), max(X2(:))]);
    ylim(ax, [min(X3(:)), max(X3(:))]);
    zlim(ax, [min(U(:)), max(U(:))]);
    % Update the slider labels
    set(sliderLabelX1, 'Text', ['Time: ', num2str(X1(1,1,1,valX1))]);
    set(sliderLabelX4, 'Text', ['X4: ', num2str(X4(1,1,valX4,1))]);
    
    % Save the captured image
    %frame = getframe(fig);
    %img = frame.cdata;
    
    % Save the captured image as a JPG file
    %imwrite(img, ['D:\andyh\Documents\Projects\mines\methane_project\pinns_practice\visualization\20240810_methane_3d_gif\images\figure_X1_',num2str(valX1),'_X4_',num2str(valX4),'.jpg']);
    
    drawnow;
end

% Keep the figure open
uiwait(fig);
