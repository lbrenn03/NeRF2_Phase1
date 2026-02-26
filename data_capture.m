clear;

% Adjust to match RX position
x_lab = 5;
y_lab = ;

% Adjust to match TX position
tx_pos = [6, 5, 1.021];


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
    'CenterFrequency',   915e6, ...
    'Gain',              40, ...
    'ChannelMapping',    [1 2], ...
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
raw_data.freq        = 915e6;
raw_data.rx_pos      = [x_lab, y_lab, 0.571];
raw_data.tx_pos      = tx_pos;
raw_data.mimo        = false;
raw_data.bf_angle    = NaN;
raw_data.rx_orient   = 0;

if x_lab >= 0
    filename = 'pilot_data/tx_6_5/p1_tx3_' + string(x_lab) + '_' + string(y_lab) + '.mat';
else
    filename = 'pilot_data/tx_6_5/p1_tx3_n1_' + string(y_lab) + '.mat';
end

save(filename, 'raw_data');

release(rxObj);
