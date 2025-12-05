clear all;
%% ==============================================================================
%  NeRF2 TRANSMITTER (Sounding Signal Beacon)
% ==============================================================================

% 1. HARDWARE CONFIG
tx = comm.SDRuTransmitter(...
    'Platform',             'B210', ...
    'SerialNum',            '34C78EF', ... 
    'MasterClockRate',      30e6, ...
    'InterpolationFactor',  60, ...          % Fs = 500 kS/s
    'CenterFrequency',      915e6, ...
    'Gain',                 60, ...          % FIXED GAIN (Do not change during data collection)
    'TransportDataType',    'int16');

Fs = 30e6 / 60; % 500 ksps

% 2. SIGNAL GENERATION (PN Sequence 1023)
% A longer sequence (1023 vs 127) gives better noise immunity
pn = comm.PNSequence(...
    'Polynomial', [10 3 0], ...  % x^10 + x^3 + 1
    'InitialConditions', ones(1,10), ...
    'SamplesPerFrame', 1023);

sounding_bits = pn();

% QPSK Modulate
qpskMod = comm.QPSKModulator('BitInput',true, 'PhaseOffset', pi/4);
% Padding for QPSK (even number of bits)
if mod(length(sounding_bits), 2) ~= 0
    sounding_bits = [sounding_bits; 0];
end
sounding_syms = qpskMod(sounding_bits);

% RRC Pulse Shaping
rrcTx = comm.RaisedCosineTransmitFilter(...
    'RolloffFactor', 0.25, ...
    'FilterSpanInSymbols', 6, ...
    'OutputSamplesPerSymbol', 5); % 5 samples per symbol

tx_waveform = rrcTx(sounding_syms);

% 3. NORMALIZE (Max amplitude 0.8 to avoid DAC clipping)
tx_waveform = tx_waveform / max(abs(tx_waveform)) * 0.8;

% 4. FRAME CONSTRUCTION
% We add a small gap of silence between packets so the RX can distinguish them
silence = complex(zeros(2000, 1)); 
tx_frame = [tx_waveform; silence];

disp("Transmitting Sounding Signal... Press Ctrl+C to stop.");

% 5. TRANSMISSION LOOP
while true
    tx(tx_frame);
end