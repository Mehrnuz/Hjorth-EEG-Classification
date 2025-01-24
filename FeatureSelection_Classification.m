function [acc, sens, spec] = FeatureSelection_Classification(featuresclass1, featuresclass2, num_subj_class1, num_subj_class2, num_epochs, num_selectedfeatures, NumNeighbors_knn)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function for Feature Selection & Multi-Classifier Evaluation
% This function performs classification using multiple classifiers (SVM, Random Forest, KNN, and LDA) 
% and evaluates their performance based on Leave-One-Out Cross-Validation (LOOCV).
%
% Inputs:
% - featuresclass1: Feature matrix for class 1. (Features x Instances (Includes both subjects and epochs)) 
% - featuresclass2: Feature matrix for class 2. (Features x Instances(Includes both subjects and epochs))
% - num_subj_class1: Number of subjects in the class1. 
% - num_subj_class2: Number of subjects in the class2. 
% - num_epochs: Number of epochs per subject. (should be equal for all subjects)
% - num_selectedfeatures: Number of features to be selected for classification.
% - NumNeighbors_knn: A positive integer specifying the number of nearest neighbors to use in the KNN classifier.
%                        
%                        
% Outputs:
% - acc: Average accuracy for each classifier (SVM, RF, LDA, KNN).
% - sens: Average sensitivity for each classifier (SVM, RF, LDA, KNN).
% - spec: Average specificity for each classifier (SVM, RF, LDA, KNN).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    % Initialize performance metrics
    acc = zeros(4, 1); % Accuracy for each classifier
    sens = zeros(4, 1); % Sensitivity for each classifier
    spec = zeros(4, 1); % Specificity for each classifier

    % Combine data from both classes for cross-validation
    combined_data = [featuresclass1, featuresclass2];
    labels = [ones(1, size(featuresclass1,2)), 2*ones(1, size(featuresclass2,2))]; % Labels for the combined data

    % Initialize arrays to store accuracy for each fold
    accuracies_SVM = zeros(num_subj_class1+num_subj_class2, 1);
    accuracies_RF = zeros(num_subj_class1+num_subj_class2, 1);
    accuracies_LDA = zeros(num_subj_class1+num_subj_class2, 1);
    accuracies_KNN = zeros(num_subj_class1+num_subj_class2, 1);

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
        svm_predictions = predict(svm_model, test_data_selected');
        accuracies_SVM(i) = mean(svm_predictions == test_labels') * 100;

        % Random Forest classification
        rf_model = fitcensemble(train_data_selected', train_labels, 'Method', 'RobustBoost');
        rf_predictions = predict(rf_model, test_data_selected');
        accuracies_RF(i) = mean(rf_predictions == test_labels') * 100;

        % LDA classification
        lda_model = fitcdiscr(train_data_selected', train_labels, 'DiscrimType', 'quadratic');
        lda_predictions = predict(lda_model, test_data_selected');
        accuracies_LDA(i) = mean(lda_predictions == test_labels') * 100;

        % KNN classification
        knn_model = fitcknn(train_data_selected', train_labels, 'NumNeighbors', NumNeighbors_knn);
        knn_predictions = predict(knn_model, test_data_selected');
        accuracies_KNN(i) = mean(knn_predictions == test_labels') * 100;
    end

    % Compute average accuracy, sensitivity, and specificity
    acc = [mean(accuracies_SVM); mean(accuracies_RF); mean(accuracies_LDA); mean(accuracies_KNN)];
    sens = [mean(accuracies_SVM(1:num_subj_class1)); mean(accuracies_RF(1:num_subj_class1)); mean(accuracies_LDA(1:num_subj_class1)); mean(accuracies_KNN(1:num_subj_class1))];
    spec = [mean(accuracies_SVM(num_subj_class1+1:num_subj_class1+num_subj_class2)); mean(accuracies_RF(num_subj_class1+1:num_subj_class1+num_subj_class2)); mean(accuracies_LDA(num_subj_class1+1:num_subj_class1+num_subj_class2)); mean(accuracies_KNN(num_subj_class1+1:num_subj_class1+num_subj_class2))];
end