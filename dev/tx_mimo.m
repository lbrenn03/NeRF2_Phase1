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

disp("Interleaving MIMO Streams Continuously... Press Ctrl+C to stop.");
% Interleave S1: place symbols at odd indices, zeros at even
% replace below line
% waveform_S1_interleaved = zeros(2*length(waveform_S1), 1);
% waveform_S1_interleaved(1:2:end) = waveform_S1;
waveform_S1_interleaved = zeros(2*length(waveform_S1), 1);
num_blocks = ceil(length(waveform_S1) / 5);
for block = 0:num_blocks-1
    src_start = block*5 + 1;
    src_end = min(src_start + 4, length(waveform_S1));
    dst_start = block*10 + 1;
    
    waveform_S1_interleaved(dst_start:dst_start+(src_end-src_start)) = waveform_S1(src_start:src_end);
end

% Interleave S2: place symbols at even indices, zeros at odd  
% waveform_S2_interleaved = zeros(2*length(waveform_S2), 1);
% waveform_S2_interleaved(2:2:end) = waveform_S2;
waveform_S2_interleaved = zeros(2*length(waveform_S2), 1);
num_blocks = ceil(length(waveform_S2) / 5);
for block = 0:num_blocks-1
    src_start = block*5 + 1;
    src_end = min(src_start + 4, length(waveform_S2));
    dst_start = block*10 + 6;  % Start at position 6 (after 5 zeros)
    
    waveform_S2_interleaved(dst_start:dst_start+(src_end-src_start)) = waveform_S2(src_start:src_end);
end

silence = complex(zeros(20000, 1)); 
tx_frame_S1 = [silence; waveform_S1_interleaved];
tx_frame_S2 = [silence; waveform_S2_interleaved];

% Zip S1 and S2: fill all zeros with data from the other stream
waveform_combined = waveform_S1_interleaved + waveform_S2_interleaved;

% Verify no overlaps (optional check)
overlap_check = (waveform_S1_interleaved ~= 0) & (waveform_S2_interleaved ~= 0);
if any(overlap_check)
    warning('Overlap detected between S1 and S2!');
else
    disp('Successfully zipped waveforms with no overlap.');
end

% Display stats
fprintf('\n=== COMBINED WAVEFORM STATS ===\n');
fprintf('Total length: %d samples\n', length(waveform_combined));
fprintf('S1 non-zero samples: %d\n', sum(waveform_S1_interleaved ~= 0));
fprintf('S2 non-zero samples: %d\n', sum(waveform_S2_interleaved ~= 0));
fprintf('Combined non-zero samples: %d\n', sum(waveform_combined ~= 0));
fprintf('Zero samples remaining: %d\n', sum(waveform_combined == 0));

% Save to file
save('mimo_waveforms.mat', 'waveform_combined', 'waveform_S1_interleaved', ...
     'waveform_S2_interleaved', 'waveform_S1', 'waveform_S2', 'Fs', 'mimo_config');

disp('Waveforms saved to mimo_waveforms.mat');

%% 5. CONTINUOUS TRANSMISSION LOOP
while true
    tx([tx_frame_S1, tx_frame_S2]);
end
