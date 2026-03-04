clear all;
%% ==============================================================================
%  NeRF2 TRANSMITTER - MIMO MODE (turn-taking antennas)
% ==============================================================================

%% MIMO CONFIGURATION
mimo_config = struct();
mimo_config.stream_mode = 'inverted';  % Options: 'inverted' (S and S̄) or 'independent'
mimo_config.fc = 915e6;                % Center frequency (Hz)

%% 1. HARDWARE CONFIG
tx = comm.SDRuTransmitter(...
    'Platform',             'B210', ...
    'SerialNum',            '34C78EF', ...
    'ChannelMapping',       [1 2], ...
    'MasterClockRate',      30e6, ...
    'InterpolationFactor',  60, ...
    'CenterFrequency',      mimo_config.fc, ...
    'Gain',                 60, ...
    'TransportDataType',    'int16');

Fs = 30e6 / 60; % 500 ksps


%% 2. SIGNAL GENERATION - STREAM S
pn_S1 = comm.PNSequence(...
    'Polynomial', [10 3 0], ...
    'InitialConditions', ones(1,10), ...
    'SamplesPerFrame', 1024);

pn_S2 = comm.PNSequence(...
    'Polynomial', [10 3 0], ...
    'InitialConditions', [1 0 1 0 1 0 1 0 1 0], ... % different initial conditions so we don't mix the signals up during timing sync
    'SamplesPerFrame', 1024);

sounding_bits_S1 = pn_S1();
sounding_bits_S2 = pn_S2();

% QPSK Modulate
qpskMod = comm.QPSKModulator('BitInput',true, 'PhaseOffset', pi/4);

sounding_syms_S1 = qpskMod(sounding_bits_S1);
sounding_syms_S2 = qpskMod(sounding_bits_S2);

% RRC Pulse Shaping
rrcTx = comm.RaisedCosineTransmitFilter(...
    'RolloffFactor', 0.25, ...
    'FilterSpanInSymbols', 6, ...
    'OutputSamplesPerSymbol', 5);

waveform_S1 = rrcTx(sounding_syms_S1);
waveform_S2 = rrcTx(sounding_syms_S2);

%% 3. NORMALIZE BOTH STREAMS
waveform_S1 = waveform_S1 / max(abs(waveform_S1)) * 0.8;
waveform_S2 = waveform_S2 / max(abs(waveform_S2)) * 0.8;

fprintf('Correlation S vs S2: %.4f\n', abs(mean(waveform_S1 .* conj(waveform_S2))));

disp("Alternating between MIMO Streams Continuously... Press Ctrl+C to stop.");
silence = complex(zeros(20000, 1)); 
tx_frame_S1 = [silence; waveform_S1];
tx_frame_S2 = [silence; waveform_S2];

waiting_for_turn = complex(zeros(len(tx_frame_S1), 1)); 

%% 5. CONTINUOUS TRANSMISSION LOOP
select = 1;
while true
    if (select == 1)
        tx([waveform_S1, waiting_for_turn]);
        select = 2;
    else
        tx([waiting_for_turn, waveform_S2]);
        select = 1;
    
end