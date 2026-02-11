clear all;
%% ==============================================================================
%  NeRF2 TRANSMITTER - MIMO MODE (2 Independent Streams)
% ==============================================================================

%% MIMO CONFIGURATION
mimo_config = struct();
mimo_config.stream_mode = 'inverted';  % Options: 'inverted' (S and S̄) or 'independent'
mimo_config.fc = 915e6;                % Center frequency (Hz)

%% 1. HARDWARE CONFIG - ANTENNA 1 (Stream S)
tx1 = comm.SDRuTransmitter(...
    'Platform',             'B210', ...
    'SerialNum',            '34C78EF', ...
    'ChannelMapping',       1, ...
    'MasterClockRate',      30e6, ...
    'InterpolationFactor',  60, ...
    'CenterFrequency',      mimo_config.fc, ...
    'Gain',                 60, ...
    'TransportDataType',    'int16');

%% HARDWARE CONFIG - ANTENNA 2 (Stream S̄)
tx2 = comm.SDRuTransmitter(...
    'Platform',             'B210', ...
    'SerialNum',            '34C78EF', ...
    'ChannelMapping',       2, ...
    'MasterClockRate',      30e6, ...
    'InterpolationFactor',  60, ...
    'CenterFrequency',      mimo_config.fc, ...
    'Gain',                 60, ...
    'TransportDataType',    'int16');

Fs = 30e6 / 60; % 500 ksps

fprintf('\n=== MIMO PARAMETERS ===\n');
fprintf('Mode: %s\n', mimo_config.stream_mode);
fprintf('Antenna 1: Transmitting S\n');
fprintf('Antenna 2: Transmitting S̄ (inverted)\n');

%% 2. SIGNAL GENERATION - STREAM S
pn = comm.PNSequence(...
    'Polynomial', [10 3 0], ...
    'InitialConditions', ones(1,10), ...
    'SamplesPerFrame', 1023);

sounding_bits = pn();

% QPSK Modulate
qpskMod = comm.QPSKModulator('BitInput',true, 'PhaseOffset', pi/4);

if mod(length(sounding_bits), 2) ~= 0
    sounding_bits = [sounding_bits; 0];
end

sounding_syms = qpskMod(sounding_bits);

% RRC Pulse Shaping
rrcTx = comm.RaisedCosineTransmitFilter(...
    'RolloffFactor', 0.25, ...
    'FilterSpanInSymbols', 6, ...
    'OutputSamplesPerSymbol', 5);

waveform_S = rrcTx(sounding_syms);

%% 3. CREATE INVERTED STREAM S̄
% Complex conjugate creates inverted signal
waveform_S_bar = -waveform_S;  % Simple inversion (can also use conj() for conjugate)

%% 4. NORMALIZE BOTH STREAMS
waveform_S = waveform_S / max(abs(waveform_S)) * 0.8;
waveform_S_bar = waveform_S_bar / max(abs(waveform_S_bar)) * 0.8;

fprintf('\n=== TRANSMISSION PARAMETERS ===\n');
fprintf('Sample Rate: %.0f kS/s\n', Fs/1e3);
fprintf('Waveform Length: %d samples (%.2f ms)\n', length(waveform_S), length(waveform_S)/Fs*1000);
fprintf('Stream S power: %.4f\n', mean(abs(waveform_S).^2));
fprintf('Stream S̄ power: %.4f\n', mean(abs(waveform_S_bar).^2));
fprintf('Correlation S vs S̄: %.4f\n', abs(mean(waveform_S .* conj(waveform_S_bar))));

disp("Transmitting MIMO Streams Continuously... Press Ctrl+C to stop.");

%% 5. CONTINUOUS TRANSMISSION LOOP
% No packet spacing - continuous transmission for MIMO spatial multiplexing
tx_count = 0;
start_time = tic;

while true
    % Simultaneous transmission of S and S̄
    tx1(waveform_S);
    tx2(waveform_S_bar);
    
    tx_count = tx_count + 1;
    
    % Display stats every 1000 transmissions
    if mod(tx_count, 1000) == 0
        elapsed = toc(start_time);
        data_rate = (tx_count * length(waveform_S)) / elapsed / 1e3; % kS/s
        fprintf('Transmissions: %d | Effective rate: %.1f kS/s | Runtime: %.1f s\n', ...
                tx_count, data_rate, elapsed);
    end
end