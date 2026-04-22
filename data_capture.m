clear;

% Adjust to match RX position
x_lab = 11.5;
y_lab = 9.5;
orientation = 135;

% Adjust to match TX position, ALSO UPDATE THE FILENAME TO TX#

tx_num = 1;
tx_pos = [-1, 10, 0.875];

%tx_num = 2;
%tx_pos = [13, 8, 0.875];

%tx_num = 3;
%tx_pos = [6, 5, 1.021];

if x_lab == round(x_lab) && y_lab == round(y_lab)
    data_folder = "mimo_data_f2/";
elseif x_lab == round(x_lab) || y_lab == round(y_lab)
    disp("Something is mismatched check x y")
else
    data_folder = "mimo_test_data_f2/";
end


%% -------------------------------------------------------------------------
% CONFIGURATION
% -------------------------------------------------------------------------
Fs          = 500e3;
CaptureTime = 2;           % seconds
TotalSamples = round(Fs * CaptureTime);

%% -------------------------------------------------------------------------
% SDR CONFIGURATION
% -------------------------------------------------------------------------
rxObj = comm.SDRuReceiver( ...
    'Platform',          'B210', ...
    'SerialNum',         '34C78FD', ...
    'MasterClockRate',   30e6, ...
    'DecimationFactor',  60, ...
    'SamplesPerFrame',   TotalSamples, ...
    'CenterFrequency',   5.4e9, ...
    'ChannelMapping',    [1 2], ...    
    'Gain',              [70 70], ...
    'OutputDataType',    'double');

%% -------------------------------------------------------------------------
% CAPTURE
% -------------------------------------------------------------------------
disp(['Capturing ', num2str(CaptureTime), ' seconds of data...']);

[rxBuffer, ~, overrun] = rxObj();

if overrun
    warning('Overrun detected during capture.');
end
if size(rxBuffer,1) < TotalSamples
    warning('Incomplete capture.');
end

disp('Capture complete. Saving raw IQ buffer...');

%% -------------------------------------------------------------------------
% SAVE
% -------------------------------------------------------------------------
raw_data.iq          = rxBuffer;        % [TotalSamples x 2] complex double
raw_data.Fs          = Fs;
raw_data.CaptureTime = CaptureTime;
raw_data.freq        = 5.4e9;
raw_data.rx_pos      = [x_lab, y_lab, 0.571];
raw_data.tx_pos      = tx_pos;
raw_data.mimo        = false;
raw_data.rx_orient   = orientation;
    
if x_lab >= 0
    filename = data_folder + 'mimo_tx' + string(tx_num) + '_' + string(x_lab) + '_' + string(y_lab) + '_' + string(orientation) + '.mat';
else
    filename = data_folder + 'mimo_tx' + string(tx_num) + '_n1_' + string(y_lab) + '_' + string(orientation) + '.mat';
end

% if you f-up a file then just uncomment this and ignore the error
% generated
%save(filename, 'raw_data');
%disp('Fixer File saved!');

if not(isfile(filename))
    save(filename, 'raw_data'); % dev filename
    disp('File saved!');
else

    error("duplicate file name")
end

release(rxObj);
