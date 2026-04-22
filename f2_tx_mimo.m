clear all;
%% ==============================================================================
% NeRF2 TRANSMITTER - MIMO MODE (turn-taking antennas)
% ==============================================================================
%% MIMO CONFIGURATION
mimo_config = struct();
mimo_config.fc = 5.40e9; % Center frequency (Hz)
sps = 5;

%% 1. HARDWARE CONFIG
tx = comm.SDRuTransmitter(...
    'Platform', 'B210', ...
    'SerialNum', '34C78EF', ...
    'ChannelMapping', [1 2], ...
    'MasterClockRate', 30e6, ...
    'InterpolationFactor', 60, ...
    'CenterFrequency', mimo_config.fc, ...
    'Gain', 80, ...
    'TransportDataType', 'int16');

Fs = 30e6 / 60; % 500 ksps

%% 2. SIGNAL GENERATION - STREAM S

pn_S1 = comm.PNSequence(...
    'Polynomial', [10 3 0], ...
    'InitialConditions', ones(1,10), ...
    'SamplesPerFrame', 1032);

sounding_bits_S1 = pn_S1();

sounding_bits_S2 = sounding_bits_S1;
for ii = 1:1032/4
    sounding_bits_S2(4*ii-3) = - (sounding_bits_S1(4*ii-1) - 1);
    sounding_bits_S2(4*ii-2) = sounding_bits_S1(4*ii);
    sounding_bits_S2(4*ii-1) = sounding_bits_S1(4*ii-3);
    sounding_bits_S2(4*ii) = - (sounding_bits_S1(4*ii-2) - 1);
end

% QPSK Modulate
qpskMod = comm.QPSKModulator('BitInput',true, 'PhaseOffset', pi/4);

sounding_syms_S1 = qpskMod(sounding_bits_S1);
sounding_syms_S2 = qpskMod(sounding_bits_S2);

% RRC Pulse Shaping
rrcTx = comm.RaisedCosineTransmitFilter(...
'RolloffFactor', 0.25, ...
'FilterSpanInSymbols', 6, ...
'OutputSamplesPerSymbol', sps);

waveform_S1 = rrcTx(sounding_syms_S1);

rrcTx = comm.RaisedCosineTransmitFilter(...
'RolloffFactor', 0.25, ...
'FilterSpanInSymbols', 6, ...
'OutputSamplesPerSymbol', sps);

waveform_S2 = rrcTx(sounding_syms_S2);

waveform_S1 = waveform_S1(3*sps + 1 : 3*sps + 2560);
waveform_S2 = waveform_S2(3*sps + 1 : 3*sps + 2560);

%% 3. NORMALIZE BOTH STREAMS
waveform_S1 = waveform_S1 / max(abs(waveform_S1)) * 0.8;
waveform_S2 = waveform_S2 / max(abs(waveform_S2)) * 0.8;

bits_S1 = sounding_bits_S1(1:1024);
bits_S2 = sounding_bits_S2(1:1024);

syms_S1 = sounding_syms_S1(1:512);
syms_S2 = sounding_syms_S2(1:512);

save('waveform_STTD.mat', 'waveform_S1', 'waveform_S2', 'bits_S1', 'bits_S2', 'syms_S1', 'syms_S2');


disp("Alternating between MIMO Streams Continuously... Press Ctrl+C to stop.");

silence = complex(zeros(5000, 1));
silence2 = complex(zeros(7560, 1));

tx_frame_S1 = [silence; waveform_S1; silence2];
tx_frame_S2 = [silence2; silence; waveform_S2];

% %% 5. CONTINUOUS TRANSMISSION LOOP
while true
    tx([tx_frame_S1, tx_frame_S2]);
end
