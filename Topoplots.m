
clc; clear; close all;

Font_size=22;

% Load the values (Note: the LIME values were normalized between 0 and 1.)
load('Figures_values.mat');

% Set up a figure for the Differences between Bipolar and Normal (Activity_beta)
figure;

% Create a topographic plot
topoplot(Difference_Act, loc, ...
         'electrodes', 'ptslabels', 'style', 'both', 'shading', 'interp');

% Configure the color axis and add a colorbar
caxis([min(Difference_Act) max(Difference_Act)]);
colormap(); % Use default colormap
colorbar();
ylabel(colorbar, 'Values (a.u.)'); % Label for colorbar
title('Differences (Bipolar-Normal)'); % Add a descriptive title
set(gca, 'FontSize', Font_size); % Adjust font for publication

% Set up a figure for the Z-values data
figure;

% Create a topographic plot
topoplot(z_values, loc, ...
         'electrodes', 'ptslabels', 'style', 'both', 'shading', 'interp');

% Configure the color axis and add a colorbar
caxis([min(z_values) max(z_values)]);
colormap(); % Use default colormap
colorbar();
ylabel(colorbar, 'Values (a.u.)'); % Label for colorbar
title('Z Values'); % Add a descriptive title
set(gca, 'FontSize', Font_size); % Adjust font for publication

% Set up a figure for the LIME Beta Normalized data (between 0 and 1)
figure;

% Create a topographic plot
topoplot(lime_val_Act_beta, loc, ...
         'electrodes', 'ptslabels', 'style', 'both', 'shading', 'interp');

% Configure the color axis and add a colorbar
caxis([min(lime_val_Act_beta) max(lime_val_Act_beta)]);
colormap(flipud(hot)); % Use a flipped 'hot' colormap for better visibility
title('LIME Beta'); % Add a descriptive title
set(gca, 'FontSize', Font_size); % Adjust font for publication

% Set up a figure for the LIME Gamma Normalized data (between 0 and 1)
figure;

% Create a topographic plot
topoplot(lime_val_Act_gama, loc, ...
         'electrodes', 'ptslabels', 'style', 'both', 'shading', 'interp');

% Configure the color axis and add a colorbar
caxis([min(lime_val_Act_gama) max(lime_val_Act_gama)]);
colormap(flipud(hot)); % Use a flipped 'hot' colormap for better visibility
title('LIME Gamma'); % Add a descriptive title
set(gca, 'FontSize', Font_size); % Adjust font for publication
