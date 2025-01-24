function Boxplot_Statistical_Hjorth(data, jitter, scatter_size, font_size)
% Boxplot_Statistical_Hjorth - Plots Hjorth features with statistical analysis.
%
% Description:
%   This function generates a box plot for Hjorth features of Bipolar and 
%   Normal groups with overlaid scatter points. Statistical analysis 
%   (Wilcoxon rank-sum test) is performed to compare the groups.
%
% INPUTS:
%   data          - Structure containing Hjorth features:
%                   data.Bipolar: Bipolar Hjorth features (3D matrix (subjects x channels x epochs))
%                   data.Normal: Normal Hjorth features (3D matrix (subjects x channels x epochs))
%   jitter        - Jitter for scatter plot points
%   scatter_size  - Size of scatter points
%   font_size     - Font size for labels, title, and axes
%
% OUTPUT:
%   A box plot with overlaid scatter points and statistical analysis.
%
% Usage:
%   Boxplot_Statistical_Hjorth(data, 0.1, 20, 22)
%
%
% Author: Ahmad Zandbagleh
% Email: ahmad.zand.elec@gmail.com
% 23-Sep-2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Validate and preprocess input data
    Bipolar = mean(data.Bipolar, 3); % Average across epochs (3rd dimension)
    Normal = mean(data.Normal, 3); % Average across epochs (3rd dimension)

    % Calculate channel means for each subject
    Bipolar_avg_ch = mean(Bipolar, 2); % Mean across all channels for Bipolar group
    Normal_avg_ch = mean(Normal, 2); % Mean across all channels for Normal group

    % Perform statistical tests (Wilcoxon rank-sum)
    num_columns = size(Bipolar, 2); % Number of channels
    p_values = zeros(1, num_columns); % Preallocate for p-values
    z_scores = zeros(1, num_columns); % Preallocate for z-scores

    for i = 1:num_columns
        [p_values(i), ~, stats] = ranksum(Bipolar(:, i), Normal(:, i), 'alpha', 0.05);
        z_scores(i) = stats.zval; % Store z-scores
    end

    % Overall statistical test for group averages
    [overall_p, ~, ~] = ranksum(Bipolar_avg_ch, Normal_avg_ch, 'alpha', 0.05);

    % Combine group averages for plotting
    combined_data = [Bipolar_avg_ch, Normal_avg_ch];
    group_labels = {'Bipolar', 'Normal'};

    % Create the box plot
    figure;
    boxplot(combined_data, 'symbol', '', 'Widths', 0.5);
    hold on;

    % Customize box plot appearance
    boxes = findobj(gca, 'Tag', 'Box'); % Get box handles
    colors = [0.0, 0.45, 0.74;  % Blue (for Normal group)
              0.85, 0.33, 0.1]; % Orange (for Bipolar group)

    % Apply colors to boxes with transparency
    patch(get(boxes(1), 'XData'), get(boxes(1), 'YData'), colors(1, :), 'FaceAlpha', 0.5); % Normal group
    patch(get(boxes(2), 'XData'), get(boxes(2), 'YData'), colors(2, :), 'FaceAlpha', 0.5); % Bipolar group

    % Add scatter points with jitter
    scatter(ones(1, length(Bipolar_avg_ch)), Bipolar_avg_ch, ...
            'MarkerEdgeColor', colors(2, :), ...
            'MarkerFaceColor', [0.9, 0.75, 0.5], ...
            'jitter', 'on', 'jitterAmount', jitter, ...
            'SizeData', scatter_size);

    scatter(2 * ones(1, length(Normal_avg_ch)), Normal_avg_ch, ...
            'MarkerEdgeColor', colors(1, :), ...
            'MarkerFaceColor', [0.8, 0.8, 0.9], ...
            'jitter', 'on', 'jitterAmount', jitter, ...
            'SizeData', scatter_size);

    % Labels and formatting
    set(gca, 'XTickLabel', group_labels, 'FontSize', font_size);
    ylabel('Values (a.u.)');
    box on;
    grid minor;
    hold off;
end
