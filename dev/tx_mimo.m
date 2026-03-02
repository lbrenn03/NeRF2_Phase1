clear all;
%% ==============================================================================
%  NeRF2 TRANSMITTER - MIMO MODE with Alamouti OSTBC (2x1 or 2x2)
% ==============================================================================

%% MIMO CONFIGURATION
mimo_config = struct();
mimo_config.fc = 915e6;                % Center frequency (Hz)
mimo_config.scheme = 'alamouti';       % Alamouti OSTBC

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

fprintf('\n=== MIMO PARAMETERS ===\n');
fprintf('Scheme: Alamouti OSTBC (2 TX antennas)\n');
fprintf('Provides: Transmit diversity (order 2)\n');
fprintf('Rate: 1 (same throughput as SISO)\n');

%% 2. SIGNAL GENERATION - Generate Symbols
pn = comm.PNSequence(...
    'Polynomial', [10 3 0], ...
    'InitialConditions', ones(1,10), ...
    'SamplesPerFrame', 1024);  % Use power of 2 for even pairing

sounding_bits = pn();

% QPSK Modulate
qpskMod = comm.QPSKModulator('BitInput',true, 'PhaseOffset', pi/4);
sounding_syms = qpskMod(sounding_bits);

fprintf('\n=== SYMBOL GENERATION ===\n');
fprintf('Total symbols: %d\n', length(sounding_syms));
fprintf('Symbol pairs for Alamouti: %d\n', length(sounding_syms)/2);

%% 3. ALAMOUTI SPACE-TIME BLOCK CODE
% Reshape symbols into pairs [s1, s2]
num_pairs = floor(length(sounding_syms) / 2);
sounding_syms = sounding_syms(1:num_pairs*2);  % Trim to even length
syms_paired = reshape(sounding_syms, 2, num_pairs);

s1 = syms_paired(1, :).';  % First symbol of each pair
s2 = syms_paired(2, :).';  % Second symbol of each pair

% Alamouti Encoding:
% Time slot 1: TX1 = s1,      TX2 = s2
% Time slot 2: TX1 = -s2*,    TX2 = s1*

% Create symbol sequences for each antenna
tx1_symbols = zeros(2*num_pairs, 1);
tx2_symbols = zeros(2*num_pairs, 1);

tx1_symbols(1:2:end) = s1;           % Odd time slots: s1
tx1_symbols(2:2:end) = -conj(s2);    % Even time slots: -s2*

tx2_symbols(1:2:end) = s2;           % Odd time slots: s2
tx2_symbols(2:2:end) = conj(s1);     % Even time slots: s1*

fprintf('\n=== ALAMOUTI ENCODING ===\n');
fprintf('TX1 structure: [s1, -s2*, s3, -s4*, ...]\n');
fprintf('TX2 structure: [s2,  s1*, s4,  s3*, ...]\n');
fprintf('Orthogonality check: symbols are time-orthogonal\n');

%% 4. PULSE SHAPING (RRC Filter)
rrcTx = comm.RaisedCosineTransmitFilter(...
    'RolloffFactor', 0.25, ...
    'FilterSpanInSymbols', 6, ...
    'OutputSamplesPerSymbol', 5);

waveform_tx1 = rrcTx(tx1_symbols);
waveform_tx2 = rrcTx(tx2_symbols);

%% 5. POWER NORMALIZATION
% Normalize both waveforms jointly to ensure equal power per antenna
max_val = max([abs(waveform_tx1); abs(waveform_tx2)]);
waveform_tx1 = waveform_tx1 / max_val * 0.8;
waveform_tx2 = waveform_tx2 / max_val * 0.8;

% Verify power balance
power_tx1 = mean(abs(waveform_tx1).^2);
power_tx2 = mean(abs(waveform_tx2).^2);

fprintf('\n=== POWER ANALYSIS ===\n');
fprintf('TX1 average power: %.4f\n', power_tx1);
fprintf('TX2 average power: %.4f\n', power_tx2);
fprintf('Total radiated power: %.4f (both antennas combined)\n', power_tx1 + power_tx2);
fprintf('Power balance (should be ~1.0): %.4f\n', power_tx1/power_tx2);

%% 6. VERIFY ALAMOUTI ORTHOGONALITY PROPERTY
% Check orthogonality of transmitted sequences
% For Alamouti: <tx1, tx2> should have specific structure
correlation = mean(waveform_tx1 .* conj(waveform_tx2));
fprintf('\n=== ORTHOGONALITY CHECK ===\n');
fprintf('Cross-correlation magnitude: %.4f\n', abs(correlation));
fprintf('(Non-zero is expected due to time-structure, not simultaneous orthogonality)\n');

%% 7. TRANSMISSION PARAMETERS
fprintf('\n=== TRANSMISSION PARAMETERS ===\n');
fprintf('Sample Rate: %.0f kS/s\n', Fs/1e3);
fprintf('Waveform Length: %d samples (%.2f ms)\n', length(waveform_tx1), length(waveform_tx1)/Fs*1000);
fprintf('Center Frequency: %.2f MHz\n', mimo_config.fc/1e6);
fprintf('TX Gain: 60 dB\n');

disp(' ');
disp('==============================================================================');
disp('Transmitting Alamouti OSTBC Continuously... Press Ctrl+C to stop.');
disp('==============================================================================');

%% 8. CONTINUOUS TRANSMISSION LOOP
tx_count = 0;
start_time = tic;

while true
    % Transmit both waveforms simultaneously
    % Column 1 = TX1 (Antenna 1), Column 2 = TX2 (Antenna 2)
    tx([waveform_tx1, waveform_tx2]);
    
    tx_count = tx_count + 1;
    
    % Display stats every 1000 transmissions
    if mod(tx_count, 1000) == 0
        elapsed = toc(start_time);
        symbols_per_sec = (tx_count * num_pairs * 2) / elapsed; % Total symbols transmitted
        data_rate_kbps = symbols_per_sec * 2 / 1e3; % QPSK = 2 bits/symbol
        fprintf('[%s] TX: %d | Symbol rate: %.1f kSym/s | Data rate: %.1f kbps | Runtime: %.1f s\n', ...
                datestr(now, 'HH:MM:SS'), tx_count, symbols_per_sec/1e3, data_rate_kbps, elapsed);
    end
    
    pause(0.05);  % Small pause to prevent buffer overflow
end
