clear all;

%% ==============================================================================
% NeRF2 BEAMFORMING TRANSMITTER (Fast Circular Sweep - 1 pkt/angle)
%% ==============================================================================

% 1. HARDWARE CONFIG
fc = 915e6;                    % Center frequency
c = 3e8;                       % Speed of light
lambda = c / fc;               % Wavelength
d = 0.5 * lambda;              % Antenna spacing (0.5λ)

tx = comm.SDRuTransmitter(...
    'Platform',             'B210', ...
    'SerialNum',            '34C78EF', ...
    'MasterClockRate',      30e6, ...
    'InterpolationFactor',  60, ...          % Fs = 500 kS/s
    'CenterFrequency',      fc, ...
    'Gain',                 60, ...
    'ChannelMapping',       [1, 2], ...      % Enable both TX channels
    'TransportDataType',    'int16');

tx.EnableBurstMode = true;
tx.NumFramesInBurst = 1;

Fs = 30e6 / 60; % 500 ksps

% 2. SIGNAL GENERATION (PN Sequence 1023)
pn = comm.PNSequence(...
    'Polynomial', [10 3 0], ...
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
    'OutputSamplesPerSymbol', 5);

tx_waveform_base = rrcTx(sounding_syms);

% Normalize (Max amplitude 0.8 to avoid DAC clipping)
tx_waveform_base = tx_waveform_base / max(abs(tx_waveform_base)) * 0.8;

% 3. BEAMFORMING PARAMETERS
angles = 0:60:299;             % 6 beam positions (0°, 60°, 120°, 180°, 240°, 300°)
num_antennas = 2;              % B210 has 2 TX channels

% Timing: 1 packet per angle
INTER_PACKET_GAP = 0.05;       % 50ms between packets (matches receiver)

% Frame construction
silence = complex(zeros(20000, 1));  % Pre-packet silence
inter_packet_silence = complex(zeros(round(INTER_PACKET_GAP * Fs), 1));

disp("==========================================================");
disp("Transmitting Beamformed Signal - 1 Packet per Angle");
disp("==========================================================");
fprintf("Center Frequency: %.2f MHz\n", fc/1e6);
fprintf("Antenna Spacing: %.4f m (0.5λ)\n", d);
fprintf("Beam angles: %s\n", mat2str(angles));
fprintf("Packets per cycle: %d\n", length(angles));
disp("Press Ctrl+C to stop.");
disp("==========================================================");

% 4. CONTINUOUS CIRCULAR SWEEP
angle_idx = 1;
cycle_count = 0;

while true
    % Current beam angle
    theta = deg2rad(angles(angle_idx));
    
    % Calculate phase shifts for 2-element ULA
    phase_shifts = zeros(1, num_antennas);
    for n = 1:num_antennas
        phase_shifts(n) = -2*pi/lambda * d * (n-1) * sin(theta);
    end
    
    % Apply phase shifts to create beamformed signals
    tx_antenna1 = tx_waveform_base * exp(1j * phase_shifts(1));
    tx_antenna2 = tx_waveform_base * exp(1j * phase_shifts(2));
    
    % Construct single-packet frames for both antennas
    tx_frame1 = [silence; tx_antenna1; inter_packet_silence];
    tx_frame2 = [silence; tx_antenna2; inter_packet_silence];
    
    % Create matrix with columns for each antenna (correct format)
    tx_frame = [tx_frame1, tx_frame2];
    
    % Transmit single packet at this angle
    tx(tx_frame);
    
    % Display progress (every complete cycle)
    if angle_idx == 1
        cycle_count = cycle_count + 1;
        fprintf("Cycle %d complete (angles: %s)\n", ...
                cycle_count, mat2str(angles));
    end
    
    % Move to next angle
    angle_idx = angle_idx + 1;
    if angle_idx > length(angles)
        angle_idx = 1;  % Loop back to 0 degrees
    end
end
