clear all;

%% ==============================================================================
% NeRF2 BEAMFORMING TRANSMITTER (Circular Sweep)
%% ==============================================================================

% 1. HARDWARE CONFIG
fc = 915e6;                    % Center frequency
c = 3e8;                       % Speed of light
lambda = c / fc;               % Wavelength
d = 0.5 * lambda;              % Antenna spacing (0.5λ)

% Assuming 2 antennas on B210 (MIMO capable)
tx = comm.SDRuTransmitter(...
    'Platform',             'B210', ...
    'SerialNum',            '34C78EF', ...
    'MasterClockRate',      30e6, ...
    'InterpolationFactor',  60, ...          % Fs = 500 kS/s
    'CenterFrequency',      fc, ...
    'Gain',                 60, ...          % FIXED GAIN
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
qpskMod = comm.QPSKModulator('BitInput', true, 'PhaseOffset', pi/4);

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
sweep_duration = 0.1;          % Time per beam direction (100ms)
angles = 0:60:359;             % Sweep in 10-degree increments (full circle)
num_antennas = 2;              % B210 has 2 TX channels

% Frame construction base
silence = complex(zeros(20000, 1));

disp("==========================================================");
disp("Transmitting Beamformed Signal with Circular Sweep");
disp("==========================================================");
fprintf("Center Frequency: %.2f MHz\n", fc/1e6);
fprintf("Antenna Spacing: %.4f m (0.5λ)\n", d);
fprintf("Sweep Rate: %.1f degrees every %.0f ms\n", 10, sweep_duration*1000);
disp("Press Ctrl+C to stop.");
disp("==========================================================");

% 4. CONTINUOUS CIRCULAR SWEEP
angle_idx = 1;
while true
    % Current beam angle
    theta = deg2rad(angles(angle_idx));
    
    % Calculate phase shifts for 2-element uniform linear array (ULA)
    % Phase delay for antenna n: -(2π/λ) * d * (n-1) * sin(θ)
    phase_shifts = zeros(1, num_antennas);
    for n = 1:num_antennas
        phase_shifts(n) = -2*pi/lambda * d * (n-1) * sin(theta);
    end
    
    % Apply phase shifts to create beamformed signals
    tx_antenna1 = tx_waveform_base * exp(1j * phase_shifts(1));
    tx_antenna2 = tx_waveform_base * exp(1j * phase_shifts(2));
    
    % Construct frames for both antennas
    tx_frame1 = [silence; tx_antenna1];
    tx_frame2 = [silence; tx_antenna2];
    
    % Interleave samples for both channels (B210 expects interleaved format)
    tx_frame = zeros(2*length(tx_frame1), 1);
    tx_frame(1:2:end) = tx_frame1;  % Channel 1 (odd indices)
    tx_frame(2:2:end) = tx_frame2;  % Channel 2 (even indices)
    
    % Transmit
    tx(tx_frame);
    
    % Display current beam direction (update every 5 angles)
    if mod(angle_idx, 5) == 1
        fprintf("Beam Direction: %3d degrees\n", angles(angle_idx));
    end
    
    % Move to next angle
    angle_idx = angle_idx + 1;
    if angle_idx > length(angles)
        angle_idx = 1;  % Loop back to 0 degrees
    end
    
    % Small pause to control sweep rate
    pause(sweep_duration);
end
