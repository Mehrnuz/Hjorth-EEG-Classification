function [acc, sens, spec] = FeatureSelection_Classification_ROC(featuresclass1, featuresclass2, num_subj_class1, num_subj_class2, num_epochs, num_selectedfeatures, NumNeighbors_knn)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function for Feature Selection & Multi-Classifier Evaluation with ROC Curves
% This function performs classification using multiple classifiers (SVM, Random Forest, KNN, and LDA) 
% and evaluates their performance based on Leave-One-Subject-Out Cross-Validation (LOSOCV). It also plots the ROC curves.
%
% Inputs:
% - featuresclass1: Feature matrix for class 1. (Features x Instances (Includes both subjects and epochs)) 
% - featuresclass2: Feature matrix for class 2. (Features x Instances(Includes both subjects and epochs))
% - label: Label vector indicating the true class for each instance (binary class labels, e.g., 0 and 1).
% - num_subj_class1: Number of subjects in the class1. 
% - num_subj_class2: Number of subjects in the class2. 
% - num_epochs: Number of epochs per subject. (should be equal for all subjects)
% - num_selectedfeatures: Number of features to be selected for classification.
% - NumNeighbors_knn: A positive integer specifying the number of nearest neighbors to use in the KNN classifier.
%
% Outputs:
% - acc: Average accuracy for each classifier (SVM, RF, LDA, KNN).
% - sens: Average sensitivity for each classifier (SVM, RF, LDA, KNN).
% - spec: Average specificity for each classifier (SVM, RF, LDA, KNN).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Ref.:
%  [1]  Saghab Torbati, M.; Zandbagleh, A.; Daliri, M.R.; Ahmadi, A.; Rostami, R.; Kazemi, R. 
%       Explainable AI for Bipolar Disorder Diagnosis Using Hjorth Parameters. Diagnostics 2025, 15, 316. 
%       https://doi.org/10.3390/diagnostics15030316 
% 
% If you use the code, please make sure that you cite Reference [1]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Authors:  Mehrnaz Saghab Torbati and Ahmad Zandbagleh
% Emails: mehrnaz.s.torbati@ieee.org and ahmad.zand.elec@gmail.com
% 23-Sep-2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Initialize performance metrics
    acc = zeros(4, 1); % Accuracy
    sens = zeros(4, 1); % Sensitivity
    spec = zeros(4, 1); % Specificity

    % Combine data from both classes for cross-validation
    combined_data = [featuresclass1, featuresclass2];
    labels = [ones(1, size(featuresclass1,2)), 2*ones(1, size(featuresclass2,2))]; % Labels for the combined data

    % Initialize arrays to store predictions and scores for ROC curves
    all_scores_SVM = [];
    all_labels = [];
    all_scores_RF = [];
    all_scores_LDA = [];
    all_scores_KNN = [];

    % LOSOCV loop
    for i = 1:num_subj_class1+num_subj_class2
        % Define test and training indices
        test_indices = (i-1) * num_epochs + 1 : i * num_epochs;
        train_indices = setdiff(1:size(combined_data, 2), test_indices);

        % Split data into training and test sets
        test_data = combined_data(:, test_indices);
        test_labels = labels(test_indices);
        train_data = combined_data(:, train_indices);
        train_labels = labels(train_indices);

        % Feature selection using MRMR (Minimum Redundancy Maximum Relevance)
        selected_features = fscmrmr(train_data', train_labels); % Rank features
        top_features = selected_features(1:num_selectedfeatures); % Select top features
        train_data_selected = train_data(top_features, :);
        test_data_selected = test_data(top_features, :);

        % SVM classification
        svm_model = fitcsvm(train_data_selected', train_labels, 'KernelFunction', 'linear');
        [svm_predictions, svm_scores] = predict(svm_model, test_data_selected');
        all_scores_SVM = [all_scores_SVM; svm_scores(:, 2)]; % Use the second column (score for class 2)
        accuracies_SVM(i) = mean(svm_predictions == test_labels') * 100;

        % Random Forest classification
        rf_model = fitcensemble(train_data_selected', train_labels, 'Method', 'RobustBoost');
        [rf_predictions, rf_scores] = predict(rf_model, test_data_selected');
        all_scores_RF = [all_scores_RF; rf_scores(:, 2)]; % Use the second column if applicable
        accuracies_RF(i) = mean(rf_predictions == test_labels') * 100;

        % LDA classification
        lda_model = fitcdiscr(train_data_selected', train_labels, 'DiscrimType', 'quadratic');
        [lda_predictions, lda_scores] = predict(lda_model, test_data_selected');
        all_scores_LDA = [all_scores_LDA; lda_scores(:, 2)]; % Use the second column (score for class 2)
        accuracies_LDA(i) = mean(lda_predictions == test_labels') * 100;

        % KNN classification
        knn_model = fitcknn(train_data_selected', train_labels, 'NumNeighbors', NumNeighbors_knn);
        [knn_predictions, knn_scores] = predict(knn_model, test_data_selected');
        knn_probabilities = knn_scores(:, 2); % Convert predicted labels to probabilities (if possible)
        all_scores_KNN = [all_scores_KNN; knn_probabilities];
        accuracies_KNN(i) = mean(knn_predictions == test_labels') * 100;

        % Collect all true labels
        all_labels = [all_labels; test_labels'];
    end

    % Compute average accuracy, sensitivity, and specificity
    acc = [mean(accuracies_SVM); mean(accuracies_RF); mean(accuracies_LDA); mean(accuracies_KNN)];
    sens = [mean(accuracies_SVM(1:num_subj_class1)); mean(accuracies_RF(1:num_subj_class1)); mean(accuracies_LDA(1:num_subj_class1)); mean(accuracies_KNN(1:num_subj_class1))];
    spec = [mean(accuracies_SVM(num_subj_class1+1:num_subj_class1+num_subj_class2)); mean(accuracies_RF(num_subj_class1+1:num_subj_class1+num_subj_class2)); mean(accuracies_LDA(num_subj_class1+1:num_subj_class1+num_subj_class2)); mean(accuracies_KNN(num_subj_class1+1:num_subj_class1+num_subj_class2))];

    % Plot ROC Curves
    figure;
    hold on;
    [fpr, tpr, ~, auc] = perfcurve(all_labels, all_scores_SVM, 2);
    plot(fpr, tpr, 'DisplayName', sprintf('SVM (AUC = %.4f)', auc), 'LineWidth', 4);

    [fpr, tpr, ~, auc] = perfcurve(all_labels, all_scores_RF, 2);
    plot(fpr, tpr, 'DisplayName', sprintf('RF (AUC = %.4f)', auc), 'LineWidth', 4);

    [fpr, tpr, ~, auc] = perfcurve(all_labels, all_scores_LDA, 2);
    plot(fpr, tpr, 'DisplayName', sprintf('LDA (AUC = %.4f)', auc), 'LineWidth', 4);

    [fpr, tpr, ~, auc] = perfcurve(all_labels, all_scores_KNN, 2);
    plot(fpr, tpr, 'DisplayName', sprintf('KNN (AUC = %.4f)', auc), 'LineWidth', 4);

    % Increase font sizes for axes, labels, title, and legend
    xlabel('False Positive Rate', 'FontSize', 20, 'FontWeight', 'bold');
    ylabel('True Positive Rate', 'FontSize', 20, 'FontWeight', 'bold');
    title('ROC Curves', 'FontSize', 20, 'FontWeight', 'bold');
    legend_handle = legend('show', 'FontSize', 14, 'FontWeight', 'bold');
    set(legend_handle, 'EdgeColor', 'black', 'LineWidth', 3); % Bold boundary
    grid on
    hold off;
end
