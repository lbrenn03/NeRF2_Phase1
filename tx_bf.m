clear all;
%% ==============================================================================
%  NeRF2 TRANSMITTER - BEAMFORMING MODE (2 Antennas)
% ==============================================================================

%% BEAMFORMING CONFIGURATION
beam_config = struct();
beam_config.theta_deg = 30;                       % Beam steering angle (degrees)
beam_config.antenna_spacing_lambda = 0.5;         % Antenna spacing in wavelengths
beam_config.fc = 915e6;                           % Center frequency (Hz)
beam_config.lambda = 3e8 / beam_config.fc;        % Wavelength (m)

%% PACKET TIMING CONFIGURATION
packet_config = struct();
packet_config.packet_interval_ms = 100;           % Time between packet starts (ms)
packet_config.silence_duration_ms = 20;           % Silent gap between packets (ms)
packet_config.enable_precise_timing = true;       % Use timer for precise intervals

%% 1. HARDWARE CONFIG - ANTENNA 1
tx1 = comm.SDRuTransmitter(...
    'Platform',             'B210', ...
    'SerialNum',            '34C78EF', ...
    'ChannelMapping',       1, ...               % TX Channel 1
    'MasterClockRate',      30e6, ...
    'InterpolationFactor',  60, ...              % Fs = 500 kS/s
    'CenterFrequency',      beam_config.fc, ...
    'Gain',                 60, ...              % FIXED GAIN
    'TransportDataType',    'int16');

%% HARDWARE CONFIG - ANTENNA 2
tx2 = comm.SDRuTransmitter(...
    'Platform',             'B210', ...
    'SerialNum',            '34C78EF', ...
    'ChannelMapping',       2, ...               % TX Channel 2
    'MasterClockRate',      30e6, ...
    'InterpolationFactor',  60, ...
    'CenterFrequency',      beam_config.fc, ...
    'Gain',                 60, ...
    'TransportDataType',    'int16');

Fs = 30e6 / 60; % 500 ksps

%% 2. CALCULATE BEAMFORMING WEIGHTS
% Phase shift for steering angle theta
d = beam_config.antenna_spacing_lambda * beam_config.lambda; % Physical spacing
theta_rad = beam_config.theta_deg * pi / 180;
phase_shift = 2 * pi * d * sin(theta_rad) / beam_config.lambda;

% Beamforming weights (antenna 1 = reference, antenna 2 = phase-shifted)
w1 = 1;                          % Weight for antenna 1
w2 = exp(1j * phase_shift);      % Weight for antenna 2 (phase steering)

fprintf('\n=== BEAMFORMING PARAMETERS ===\n');
fprintf('Steering Angle: %.1f degrees\n', beam_config.theta_deg);
fprintf('Antenna Spacing: %.2f lambda (%.3f m)\n', beam_config.antenna_spacing_lambda, d);
fprintf('Phase Shift: %.2f degrees\n', phase_shift * 180/pi);
fprintf('Weight 1: %.3f ∠ %.1f°\n', abs(w1), angle(w1)*180/pi);
fprintf('Weight 2: %.3f ∠ %.1f°\n', abs(w2), angle(w2)*180/pi);

%% 3. SIGNAL GENERATION (PN Sequence 1023)
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

base_waveform = rrcTx(sounding_syms);

%% 4. APPLY BEAMFORMING WEIGHTS
tx_waveform_ant1 = w1 * base_waveform;
tx_waveform_ant2 = w2 * base_waveform;

% Normalize both
max_val = max([abs(tx_waveform_ant1); abs(tx_waveform_ant2)]);
tx_waveform_ant1 = tx_waveform_ant1 / max_val * 0.8;
tx_waveform_ant2 = tx_waveform_ant2 / max_val * 0.8;

%% 5. FRAME CONSTRUCTION WITH CONFIGURABLE SPACING
silence_samples = round((packet_config.silence_duration_ms / 1000) * Fs);
silence = complex(zeros(silence_samples, 1));

packet_interval_samples = round((packet_config.packet_interval_ms / 1000) * Fs);
waveform_length = length(tx_waveform_ant1) + silence_samples;

if packet_interval_samples > waveform_length
    additional_silence = packet_interval_samples - waveform_length;
    silence = complex(zeros(silence_samples + additional_silence, 1));
end

tx_frame_ant1 = [tx_waveform_ant1; silence];
tx_frame_ant2 = [tx_waveform_ant2; silence];

fprintf('\n=== TRANSMISSION PARAMETERS ===\n');
fprintf('Sample Rate: %.0f kS/s\n', Fs/1e3);
fprintf('Packet Duration: %.2f ms\n', length(tx_frame_ant1)/Fs*1000);
fprintf('Packet Rate: %.2f packets/sec\n', 1000/packet_config.packet_interval_ms);

disp("Transmitting Beamformed Signal... Press Ctrl+C to stop.");

%% 6. SYNCHRONIZED TRANSMISSION LOOP
if packet_config.enable_precise_timing
    packet_timer = tic;
    packet_count = 0;
    
    while true
        % Transmit simultaneously from both antennas
        tx1(tx_frame_ant1);
        tx2(tx_frame_ant2);
        
        packet_count = packet_count + 1;
        
        elapsed = toc(packet_timer);
        target_time = packet_count * (packet_config.packet_interval_ms / 1000);
        wait_time = target_time - elapsed;
        
        if wait_time > 0
            pause(wait_time);
        end
        
        if mod(packet_count, 100) == 0
            actual_rate = packet_count / toc(packet_timer);
            fprintf('Packets sent: %d | Actual rate: %.2f pkt/s\n', packet_count, actual_rate);
        end
    end
else
    while true
        tx1(tx_frame_ant1);
        tx2(tx_frame_ant2);
    end
end