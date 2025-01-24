function lime_val = SVM_Classification_LIME(class1_data, class2_data, num_important_predictors, subject_samples)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function for Feature Selection & Classification (SVM) with LIME Explanation
% This function performs classification using SVM, evaluates its performance using Leave-One-Subject-Out Cross-Validation (LOSOCV),
% and uses LIME to explain the model's predictions.
%
% Inputs:
% - class1_data: Feature matrix for class 1 (Channels x Instances).
% - class2_data: Feature matrix for class 2 (Channels x Instances).
% - num_important_predictors: Number of predictors to explain with LIME.
% - subject_samples: Number of samples (epochs) per subject.
%
% Outputs:
% - lime_val: Importance values across all channels from LIME.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Combine data and labels
    combined_data = [class1_data, class2_data];
    labels = [ones(1, size(class1_data, 2)), 2 * ones(1, size(class2_data, 2))];

    % Parameters
    num_channels = size(combined_data, 1); % Number of channels
    num_subjects = size(combined_data, 2) / subject_samples; % Number of subjects

    % Initialize LIME importance accumulator
    lime_val = zeros(1, num_channels);

    % LOSOCV loop
    for subject = 1:num_subjects
        % Define test and train indices
        test_indices = (subject - 1) * subject_samples + 1 : subject * subject_samples;
        train_indices = setdiff(1:size(combined_data, 2), test_indices);

        % Split data into training and test sets
        train_data = combined_data(:, train_indices);
        test_data = combined_data(:, test_indices);
        train_labels = labels(train_indices);

        % Train SVM model
        mdl = fitcsvm(train_data', train_labels);

        % Initialize LIME explainer
        explainer = lime(mdl, train_data', 'NumImportantPredictors', num_important_predictors);

        % Compute LIME importance for each test sample
        for i = 1:size(test_data, 2)
            test_sample = test_data(:, i)'; % Current test sample

            % Fit LIME to the test sample
            lime_fitted = fit(explainer, test_sample);

            % Extract important predictor indices and their importance values
            important_indices = lime_fitted.ImportantPredictors;

            % Assign LIME importance values based on model type
            if isa(lime_fitted.SimpleModel, 'ClassificationLinear')
                lime_importance = abs(lime_fitted.SimpleModel.Beta);
            elseif isa(lime_fitted.SimpleModel, 'ClassificationTree')
                lime_importance = abs(lime_fitted.SimpleModel.predictorImportance);
            else
                error('Unsupported model type for LIME SimpleModel');
            end

            % Map LIME importance values to the full channel space
            lime_importance_full = zeros(1, num_channels);
            lime_importance_full(important_indices) = lime_importance;

            % Accumulate LIME importance values
            lime_val = lime_val + lime_importance_full;
        end
    end

    % Normalize LIME importance values
    lime_val = lime_val / (num_subjects * subject_samples);
end
