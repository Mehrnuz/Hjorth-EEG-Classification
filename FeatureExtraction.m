function [HjorthActivity, HjorthMobility, HjorthComplexity] = FeatureExtraction(EEG_cell, fs, filt_order, num_frequencybands, num_channels, num_epochs, num_subjects)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function for feature extraction
% This function performs feature extraction from EEG data.
% Inputs:
% - EEG_cell: EEG dataset for all subjects. It should be a cell array containing all subjects for each group.
%              Each element contains a 3D matrix of size (time points x channels x epochs (should be equal for all subjects)).
% - fs: Sampling frequency in Hz.
% - filt_order: Order of the Butterworth filter used for subband extraction.
% - num_frequencybands: Number of frequency bands to process. (default: 5)
% - num_channels: Number of EEG channels in the dataset.
% - num_epochs: Number of epochs per subject. (should be equal for all subjects)
% - num_subjects: Number of subjects in the dataset.
%
% Outputs:
% - HjorthActivity: Matrix storing the Hjorth Activity parameter for each band, channel, epoch, and subject. (bands x channels x [epochs x subjects])
% - HjorthMobility: Matrix storing the Hjorth Mobility parameter for each band, channel, epoch, and subject. (bands x channels x [epochs x subjects])
% - HjorthComplexity: Matrix storing the Hjorth Complexity parameter for each band, channel, epoch, and subject. (bands x channels x [epochs x subjects])
% Output Dimensions:
% - The output matrices (HjorthActivity, HjorthMobility, HjorthComplexity) are structured as:
%   (bands x channels x [epochs x subjects]).
%   For example:
%   - If there are 5 bands, 19 channels, 50 epochs, and 20 subjects:
%     - Total epochs across all subjects = 50 x 20 = 1000.
%     - The resulting matrix dimensions will be (5 x 19 x 1000).
% Processing Logic:
% - The function concatenates the epochs for each subject sequentially.
% - For each subject, all epochs are processed and appended.
% - After completing one subject, the next subject's epochs are concatenated to the first.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Initialize Hjorth parameters
    HjorthActivity = zeros(num_frequencybands, num_channels, num_epochs * num_subjects);
    HjorthMobility = zeros(num_frequencybands, num_channels, num_epochs * num_subjects);
    HjorthComplexity = zeros(num_frequencybands, num_channels, num_epochs * num_subjects);

    % Frequency band ranges (in Hz)
    frequencyBands = [
        0.5, 4;   % Delta
        4, 8;     % Theta
        8, 13;    % Alpha
        13, 30;   % Beta
        30, 45    % Gamma
    ];

    % Precompute filter coefficients for each band
    filters = cell(num_frequencybands, 2); % Store b and a coefficients in a cell array
    for i = 1:num_frequencybands
        [filters{i, 1}, filters{i, 2}] = butter(filt_order, frequencyBands(i, :) / (fs / 2), 'bandpass');
    end

    % Filter EEG data for each subject and extract features
    for subj = 1:num_subjects
        EEG_data = EEG_cell{subj}; % Get data for the current subject

        % Initialize filtered EEG data
        filteredEEG = zeros(size(EEG_data));

        % Apply filters for each band and epoch
        for band = 1:num_frequencybands
            b = filters{band, 1}; % Extract b coefficients
            a = filters{band, 2}; % Extract a coefficients

            % Filter each epoch and channel
            for epoch = 1:num_epochs
                for ch = 1:num_channels
                    signal_raw = EEG_data(:, ch, epoch);
                    filteredEEG(:, ch, epoch) = filtfilt(b, a, signal_raw);
                end
            end

            % Calculate Hjorth parameters for each epoch and channel
            for epoch = 1:num_epochs
                for ch = 1:num_channels
                    signal_filt = filteredEEG(:, ch, epoch);
                    HjorthActivity(band, ch, epoch + (subj - 1) * num_epochs) = var(signal_filt);
                    HjorthMobility(band, ch, epoch + (subj - 1) * num_epochs) = HjorthMobility_func(signal_filt);
                    HjorthComplexity(band, ch, epoch + (subj - 1) * num_epochs) = HjorthMobility_func(diff([0; signal_filt])) / HjorthMobility_func(signal_filt);
                end
        
            end
        end
    end
end

% Function to calculate Hjorth Mobility
% This function computes the Hjorth Mobility parameter for a given signal.
% Inputs:
% - signal: A vector representing the EEG signal for one channel.
% Outputs:
% - HjorthMobility: A scalar value representing the mobility of the signal.
function HjorthMobility = HjorthMobility_func(signal)
    HjorthMobility = sqrt(var(diff([0; signal])) / var(signal));
end
