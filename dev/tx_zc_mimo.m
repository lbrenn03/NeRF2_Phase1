clear all; close all; clc;

%% 1. CONFIGURATION
mimo_config = struct();
mimo_config.fc = 915e6;              
mimo_config.Gain = 60;               
mimo_config.MasterClock = 30e6;      
mimo_config.Interp = 60;             
Fs = mimo_config.MasterClock / mimo_config.Interp; 

% Zadoff-Chu Parameters (Length 1023)
N_zc = 1021; 
u1 = 31;    
u2 = 37;    

%% 2. SIGNAL GENERATION
n = (0:N_zc-1)';
% Standard ZC Formula for odd N: exp(-j * pi * u * n * (n+1) / N_zc)
zc_S1 = exp(-1j * pi * u1 * n .* (n + 1) / N_zc);
zc_S2 = exp(-1j * pi * u2 * n .* (n + 1) / N_zc);

% Pulse Shaping
rrcTx = comm.RaisedCosineTransmitFilter(...
    'RolloffFactor', 0.25, ...
    'FilterSpanInSymbols', 8, ...
    'OutputSamplesPerSymbol', 4);

waveform_S1 = rrcTx(zc_S1);
waveform_S2 = rrcTx(zc_S2);

% Normalize
waveform_S1 = waveform_S1 / max(abs(waveform_S1)) * 0.8;
waveform_S2 = waveform_S2 / max(abs(waveform_S2)) * 0.8;

%% 3. SAVE FOR PYTHON (.mat)
% Saving variables for scipy.io.loadmat
save('nerf2_mimo_refs.mat', 'waveform_S1', 'waveform_S2', 'u1', 'u2', 'Fs', 'N_zc');
fprintf('Success: nerf2_mimo_refs.mat created.\n');

%% 4. HARDWARE & TRANSMISSION
tx = comm.SDRuTransmitter(...
    'Platform',             'B210', ...
    'SerialNum',            '34C78EF', ...
    'ChannelMapping',       [1 2], ... 
    'MasterClockRate',      mimo_config.MasterClock, ...
    'InterpolationFactor',  mimo_config.Interp, ...
    'CenterFrequency',      mimo_config.fc, ...
    'Gain',                 mimo_config.Gain, ...
    'TransportDataType',    'int16');

silence_gap = complex(zeros(20000, 1)); 
tx_frame = [[waveform_S1; silence_gap], [waveform_S2; silence_gap]];

disp("Transmitting Simultaneous ZC-1023... Press Ctrl+C to stop.");
while true
    tx(tx_frame);
end
