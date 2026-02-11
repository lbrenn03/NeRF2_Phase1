clear;
%% =========================================================================
%  NeRF2 RECEIVER — MIMO MODE (2 TX × 2 RX Correlation Collection)
%  Records: ant1_sig1, ant1_sig2, ant2_sig1, ant2_sig2 for each packet
% =========================================================================

%% -------------------------------------------------------------------------
% 1. CONFIGURATION
% -------------------------------------------------------------------------
Fs          = 500e3;
CaptureTime = 2;                    % seconds
TotalSamples = round(Fs * CaptureTime);

sps     = 5;
rolloff = 0.25;
span    = 6;

NUM_PACKETS = 10;  % Process top 10 correlation peaks

%% -------------------------------------------------------------------------
% 2. REFERENCE SIGNAL GENERATION (MATCH TX)
% -------------------------------------------------------------------------
pn = comm.PNSequence( ...
    'Polynomial', [10 3 0], ...
    'InitialConditions', ones(1,10), ...
    'SamplesPerFrame', 1023);

ref_bits = pn();
if mod(numel(ref_bits),2)
    ref_bits(end+1) = 0;
end

qpskMod = comm.QPSKModulator( ...
    'BitInput', true, ...
    'PhaseOffset', pi/4);

rrcTx = comm.RaisedCosineTransmitFilter( ...
    'RolloffFactor', rolloff, ...
    'FilterSpanInSymbols', span, ...
    'OutputSamplesPerSymbol', sps);

ref_syms = qpskMod(ref_bits);
reference_sig = rrcTx(ref_syms);
reference_sig = reference_sig / max(abs(reference_sig));

% Generate orthogonal reference for MIMO (signal 2)
ref_syms_orthog = -conj(ref_syms);
reference_sig_orthog = rrcTx(ref_syms_orthog);
reference_sig_orthog = reference_sig_orthog / max(abs(reference_sig_orthog));

N = numel(reference_sig);   % packet length (samples)

%% -------------------------------------------------------------------------
% 3. SDR CONFIGURATION (SINGLE LARGE CAPTURE)
% -------------------------------------------------------------------------
rxObj = comm.SDRuReceiver( ...
    'Platform',         'B210', ...
    'SerialNum',        '34C78FD', ...
    'MasterClockRate',  30e6, ...
    'DecimationFactor', 60, ...
    'SamplesPerFrame',  TotalSamples, ...
    'CenterFrequency',  915e6, ...
    'Gain',             40, ...
    'ChannelMapping',   [1 2], ...
    'OutputDataType',   'double');

cfoObj = comm.CoarseFrequencyCompensator( ...
    'Modulation', 'QPSK', ...
    'SampleRate', Fs, ...
    'FrequencyResolution', 50);

%% -------------------------------------------------------------------------
% 4. CAPTURE
% -------------------------------------------------------------------------
disp(['Capturing ', num2str(CaptureTime), ' seconds of data...']);

[rxBuffer, ~, overrun] = rxObj();
if overrun
    warning('Overrun detected during capture.');
end
if size(rxBuffer,1) < TotalSamples
    warning('Incomplete capture.');
end

disp('Capture complete. Starting offline processing.');

% Check for saturation
fprintf('Max amplitude: %.4f\n', max(abs(rxBuffer(:))));
fprintf('Mean power ant1: %.4f\n', mean(abs(rxBuffer(:,1)).^2));
fprintf('Mean power ant2: %.4f\n', mean(abs(rxBuffer(:,2)).^2));

%% -------------------------------------------------------------------------
% 5. PACKET DETECTION (XCORR) - Local Peaks with Time-Based Spacing
% -------------------------------------------------------------------------
% Use primary signal for packet detection
[corrVals, lags] = xcorr(rxBuffer(:,1), reference_sig);
corrMag = abs(corrVals);

% --- Known TX behavior ---
interPacketTime = 0.05;                    % seconds (TX wait)
minPeakDist = round(interPacketTime * Fs); % samples between packets

% --- Robust prominence threshold ---
minProm = 0.75 * max(corrMag);

% --- Find true LOCAL peaks (one per transmission) ---
[pks, locs] = findpeaks( ...
    corrMag, ...
    'MinPeakDistance', minPeakDist, ...
    'MinPeakProminence', minProm ...
);

% --- Sort by strength AFTER enforcing spacing ---
[~, sortIdx] = sort(pks, 'descend');
pks  = pks(sortIdx);
locs = locs(sortIdx);

% -------------------------------------------------------------------------
% Filter valid packets
% -------------------------------------------------------------------------
valid_packets = [];
threshold = 0;

for k = 1:numel(locs)
    lag = lags(locs(k));
    s   = lag + 1;
    e   = s + N - 1;

    if s > 0 && e <= size(rxBuffer,1) && pks(k) > threshold
        valid_packets(end+1).startIdx = s; %#ok<SAGROW>
        valid_packets(end).peakVal  = pks(k);
        valid_packets(end).peakTime = s / Fs;
    end
    
    if numel(valid_packets) >= NUM_PACKETS
        break;
    end
end

if isempty(valid_packets)
    disp('❌ No valid packets found above threshold.');
    release(rxObj);
    return
end

fprintf('✅ Found %d valid packets\n\n', numel(valid_packets));

%% -------------------------------------------------------------------------
% 6. PROCESS EACH PACKET - COLLECT 4 CORRELATIONS
% -------------------------------------------------------------------------
rrcRx = comm.RaisedCosineReceiveFilter( ...
    'RolloffFactor', rolloff, ...
    'FilterSpanInSymbols', span, ...
    'InputSamplesPerSymbol', sps, ...
    'DecimationFactor', sps);

qpskDemod = comm.QPSKDemodulator( ...
    'BitOutput', true, ...
    'PhaseOffset', pi/4);

num_ref_syms = length(ref_syms);

% Storage for results
% ant1_sig1 = RX antenna 1 correlation with reference signal 1 (primary)
% ant1_sig2 = RX antenna 1 correlation with reference signal 2 (orthogonal)
% ant2_sig1 = RX antenna 2 correlation with reference signal 1 (primary)
% ant2_sig2 = RX antenna 2 correlation with reference signal 2 (orthogonal)
results = struct('ber1', [], 'ber2', [], 'numErr1', [], 'numErr2', [], ...
                 'cfo', [], 'peakVal', [], 'peakTime', [], ...
                 'ant1_sig1', [], 'ant1_sig2', [], 'ant2_sig1', [], 'ant2_sig2', [], ...
                 'rx_power_ant1', [], 'rx_power_ant2', [], 'power_ratio', []);

% Storage for constellation analysis
cluster_centers_ant1 = zeros(4, numel(valid_packets));
cluster_centers_ant2 = zeros(4, numel(valid_packets));

% QPSK ideal constellation points (for reference)
qpsk_ideal = exp(1j * (pi/4 + [0 pi/2 pi 3*pi/2]));

figure(1); clf;
all_syms_ant1 = zeros(500, numel(valid_packets));
all_syms_ant2 = zeros(500, numel(valid_packets));

for p = 1:numel(valid_packets)
    startIdx = valid_packets(p).startIdx;
    
    %% Extract packets
    packet_ant1 = rxBuffer(startIdx:startIdx+N-1,1);
    packet_ant2 = rxBuffer(startIdx:startIdx+N-1,2);
    
    %% CFO Correction
    [~, estCFO] = cfoObj(packet_ant1);
    t = (0:N-1).' / Fs;
    cfoVec = exp(-1j*2*pi*estCFO*t);
    
    clean_ant1 = packet_ant1 .* cfoVec;
    clean_ant2 = packet_ant2 .* cfoVec;
    
    %% Matched Filter
    mf_ant1 = rrcRx(clean_ant1);
    mf_ant2 = rrcRx(clean_ant2);
    
    %% Fine Synchronization with Signal 1 (primary)
    [fine_corr1_sig1, fine_lags] = xcorr(mf_ant1, ref_syms, 'normalized');
    [~, fine_idx] = max(abs(fine_corr1_sig1));
    
    fine_timing_offset = fine_lags(fine_idx);
    fine_phase_ant1 = angle(fine_corr1_sig1(fine_idx));
    
    [fine_corr2_sig1, ~] = xcorr(mf_ant2, ref_syms, 'normalized');
    fine_phase_ant2 = angle(fine_corr2_sig1(fine_idx));
    
    %% Correlate with Signal 2 (orthogonal)
    [fine_corr1_sig2, ~] = xcorr(mf_ant1, ref_syms_orthog, 'normalized');
    [fine_corr2_sig2, ~] = xcorr(mf_ant2, ref_syms_orthog, 'normalized');
    
    %% Store all 4 correlation values at the same timing index
    ant1_sig1 = fine_corr1_sig1(fine_idx);
    ant1_sig2 = fine_corr1_sig2(fine_idx);
    ant2_sig1 = fine_corr2_sig1(fine_idx);
    ant2_sig2 = fine_corr2_sig2(fine_idx);
    
    %% Extract and align symbols
    if fine_timing_offset > 0 && fine_timing_offset + num_ref_syms <= length(mf_ant1)
        syms_ant1 = mf_ant1(fine_timing_offset : fine_timing_offset + num_ref_syms - 1);
        syms_ant2 = mf_ant2(fine_timing_offset : fine_timing_offset + num_ref_syms - 1);
    elseif fine_timing_offset <= 0 && abs(fine_timing_offset) + num_ref_syms <= length(mf_ant1)
        idx_start = abs(fine_timing_offset) + 1;
        syms_ant1 = mf_ant1(idx_start : idx_start + num_ref_syms - 1);
        syms_ant2 = mf_ant2(idx_start : idx_start + num_ref_syms - 1);
    else
        syms_ant1 = mf_ant1(span+1:end-span);
        syms_ant2 = mf_ant2(span+1:end-span);
    end
    
    %% Apply phase correction (using signal 1 phase)
    syms_ant1_corrected = syms_ant1 * exp(-1j * fine_phase_ant1);
    syms_ant2_corrected = syms_ant2 * exp(-1j * fine_phase_ant2);
    
    %% Store symbols for constellation analysis
    all_syms_ant1(:, p) = syms_ant1;
    all_syms_ant2(:, p) = syms_ant2;
    
    %% Calculate cluster centers for this packet using k-means
    try
        [~, centers1] = kmeans([real(syms_ant1), imag(syms_ant1)], 4, ...
            'MaxIter', 100, 'Replicates', 3);
        cluster_centers_ant1(:, p) = centers1(:,1) + 1j*centers1(:,2);
    catch
        cluster_centers_ant1(:, p) = NaN(4,1);
    end
    
    try
        [~, centers2] = kmeans([real(syms_ant2), imag(syms_ant2)], 4, ...
            'MaxIter', 100, 'Replicates', 3);
        cluster_centers_ant2(:, p) = centers2(:,1) + 1j*centers2(:,2);
    catch
        cluster_centers_ant2(:, p) = NaN(4,1);
    end
    
    %% Demodulate
    rx_bits_ant1 = qpskDemod(syms_ant1_corrected);
    rx_bits_ant2 = qpskDemod(syms_ant2_corrected);
    
    %% Calculate BER
    L = min(numel(ref_bits), numel(rx_bits_ant1));
    ref_bits_trimmed = ref_bits(1:L);
    
    [numErr1, ber1] = biterr(ref_bits_trimmed, rx_bits_ant1(1:L));
    [numErr2, ber2] = biterr(ref_bits_trimmed, rx_bits_ant2(1:L));
    
    %% Power measurements
    rx_power_ant1 = mean(abs(mf_ant1).^2);
    rx_power_ant2 = mean(abs(mf_ant2).^2);
    power_ratio = rx_power_ant1 / (rx_power_ant2 + eps);
    
    %% Store results
    results(p).ber1 = ber1;
    results(p).ber2 = ber2;
    results(p).numErr1 = numErr1;
    results(p).numErr2 = numErr2;
    results(p).cfo = estCFO;
    results(p).peakVal = valid_packets(p).peakVal;
    results(p).peakTime = valid_packets(p).peakTime;
    results(p).ant1_sig1 = ant1_sig1;
    results(p).ant1_sig2 = ant1_sig2;
    results(p).ant2_sig1 = ant2_sig1;
    results(p).ant2_sig2 = ant2_sig2;
    results(p).rx_power_ant1 = rx_power_ant1;
    results(p).rx_power_ant2 = rx_power_ant2;
    results(p).power_ratio = power_ratio;
    
    if mod(p, 10) == 0
        fprintf('Processed %d / %d packets\n', p, numel(valid_packets));
    end
end

%% -------------------------------------------------------------------------
% 7. DATA QUALITY VISUALIZATION
% -------------------------------------------------------------------------
figure(2); clf;

% Plot 1: EVM (Error Vector Magnitude) per packet
subplot(2,3,1)
evm_ant1 = zeros(1, numel(valid_packets));
evm_ant2 = zeros(1, numel(valid_packets));
for p = 1:numel(valid_packets)
    % Rotate to nearest QPSK point
    [~, nearest_idx] = min(abs(all_syms_ant1(:,p) - qpsk_ideal.'), [], 2);
    nearest_syms = qpsk_ideal(nearest_idx);
    evm_ant1(p) = sqrt(mean(abs(all_syms_ant1(:,p) - nearest_syms.').^2));
    
    [~, nearest_idx] = min(abs(all_syms_ant2(:,p) - qpsk_ideal.'), [], 2);
    nearest_syms = qpsk_ideal(nearest_idx);
    evm_ant2(p) = sqrt(mean(abs(all_syms_ant2(:,p) - nearest_syms.').^2));
end
plot(1:numel(valid_packets), evm_ant1, 'b-o', 'LineWidth', 1.5); hold on;
plot(1:numel(valid_packets), evm_ant2, 'r-s', 'LineWidth', 1.5);
yline(0.2, 'k--', 'Warning Threshold');
hold off;
grid on;
xlabel('Packet Number');
ylabel('EVM');
title('Error Vector Magnitude per Packet');
legend('Ant 1', 'Ant 2', 'Location', 'best');

% Plot 2: SNR estimate per packet
subplot(2,3,2)
snr_ant1 = -10*log10(evm_ant1.^2);
snr_ant2 = -10*log10(evm_ant2.^2);
plot(1:numel(valid_packets), snr_ant1, 'b-o', 'LineWidth', 1.5); hold on;
plot(1:numel(valid_packets), snr_ant2, 'r-s', 'LineWidth', 1.5);
yline(15, 'k--', 'Good SNR');
hold off;
grid on;
xlabel('Packet Number');
ylabel('SNR (dB)');
title('Estimated SNR per Packet');
legend('Ant 1', 'Ant 2', 'Location', 'best');

% Plot 3: Correlation peak strength (channel consistency)
subplot(2,3,3)
peak_vals = [results.peakVal];
plot(1:numel(valid_packets), peak_vals, 'b-o', 'LineWidth', 1.5);
grid on;
xlabel('Packet Number');
ylabel('Correlation Peak');
title('Correlation Peak Strength (Channel Consistency)');

% Plot 4: Symbol-wise averaged constellation (Ant 1)
subplot(2,3,4)
avg_syms_ant1 = mean(all_syms_ant1, 2);
scatter(real(avg_syms_ant1), imag(avg_syms_ant1), 15, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
plot(real(qpsk_ideal), imag(qpsk_ideal), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
grid on; axis equal;
title(sprintf('Symbol-Averaged Constellation - Ant 1\nEVM=%.3f', mean(evm_ant1)));
xlabel('In-Phase'); ylabel('Quadrature');

% Plot 5: Symbol-wise averaged constellation (Ant 2)
subplot(2,3,5)
avg_syms_ant2 = mean(all_syms_ant2, 2);
scatter(real(avg_syms_ant2), imag(avg_syms_ant2), 15, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
plot(real(qpsk_ideal), imag(qpsk_ideal), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
grid on; axis equal;
title(sprintf('Symbol-Averaged Constellation - Ant 2\nEVM=%.3f', mean(evm_ant2)));
xlabel('In-Phase'); ylabel('Quadrature');

% Plot 6: Pass/Fail summary
subplot(2,3,6)
axis off;
text(0.1, 0.9, 'DATA QUALITY SUMMARY', 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.75, sprintf('Packets: %d', numel(valid_packets)), 'FontSize', 10);
text(0.1, 0.65, sprintf('Mean EVM Ant1: %.3f', mean(evm_ant1)), 'FontSize', 10);
text(0.1, 0.55, sprintf('Mean EVM Ant2: %.3f', mean(evm_ant2)), 'FontSize', 10);
text(0.1, 0.45, sprintf('Mean SNR Ant1: %.1f dB', mean(snr_ant1)), 'FontSize', 10);
text(0.1, 0.35, sprintf('Mean SNR Ant2: %.1f dB', mean(snr_ant2)), 'FontSize', 10);

% Quality decision
if mean(evm_ant1) < 0.2 && mean(evm_ant2) < 0.2 && std(evm_ant1) < 0.1
    text(0.1, 0.15, '✓ GOOD QUALITY', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'g');
elseif mean(evm_ant1) < 0.35 && mean(evm_ant2) < 0.35
    text(0.1, 0.15, '⚠ MARGINAL QUALITY', 'FontSize', 14, 'FontWeight', 'bold', 'Color', [1 0.5 0]);
else
    text(0.1, 0.15, '✗ POOR QUALITY - RETAKE', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'r');
end

%% -------------------------------------------------------------------------
% 8. INTRA-PACKET DENOISING ANALYSIS
% -------------------------------------------------------------------------
fprintf('\nComputing intra-packet denoising...\n');

% For each packet, cluster symbols and compute MSE
intra_packet_mse_ant1 = zeros(1, numel(valid_packets));
intra_packet_mse_ant2 = zeros(1, numel(valid_packets));

for p = 1:numel(valid_packets)
    syms_p_ant1 = all_syms_ant1(:, p);
    syms_p_ant2 = all_syms_ant2(:, p);
    
    % K-means to find 4 clusters
    try
        [idx1, centers1] = kmeans([real(syms_p_ant1), imag(syms_p_ant1)], 4, ...
            'MaxIter', 100, 'Replicates', 3);
        centers1_complex = centers1(:,1) + 1j*centers1(:,2);
        
        % MSE within each cluster (noise within constellation point)
        cluster_mse = zeros(4,1);
        for c = 1:4
            cluster_syms = syms_p_ant1(idx1 == c);
            if ~isempty(cluster_syms)
                cluster_mse(c) = mean(abs(cluster_syms - centers1_complex(c)).^2);
            end
        end
        intra_packet_mse_ant1(p) = mean(cluster_mse, 'omitnan');
    catch
        intra_packet_mse_ant1(p) = NaN;
    end
    
    try
        [idx2, centers2] = kmeans([real(syms_p_ant2), imag(syms_p_ant2)], 4, ...
            'MaxIter', 100, 'Replicates', 3);
        centers2_complex = centers2(:,1) + 1j*centers2(:,2);
        
        cluster_mse = zeros(4,1);
        for c = 1:4
            cluster_syms = syms_p_ant2(idx2 == c);
            if ~isempty(cluster_syms)
                cluster_mse(c) = mean(abs(cluster_syms - centers2_complex(c)).^2);
            end
        end
        intra_packet_mse_ant2(p) = mean(cluster_mse, 'omitnan');
    catch
        intra_packet_mse_ant2(p) = NaN;
    end
end

figure(3); clf;

% Plot 1: Intra-packet MSE (noise floor per packet)
subplot(1,2,1)
semilogy(1:numel(valid_packets), intra_packet_mse_ant1, 'b-o', 'LineWidth', 1.5); hold on;
semilogy(1:numel(valid_packets), intra_packet_mse_ant2, 'r-s', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Packet Number');
ylabel('Intra-Packet MSE');
title('Noise Floor per Packet');
legend('Ant 1', 'Ant 2', 'Location', 'best');

% Plot 2: Inter-packet channel variation
subplot(1,2,2)
% Standard deviation of correlation peaks (channel variation indicator)
corr_std = std([results.peakVal]);
corr_mean = mean([results.peakVal]);
bar([corr_mean, corr_std]);
grid on;
set(gca, 'XTickLabel', {'Mean Peak', 'Std Dev'});
ylabel('Correlation Value');
title('Channel Consistency Metric');
text(1, corr_mean*1.1, sprintf('CV = %.1f%%', 100*corr_std/corr_mean), 'HorizontalAlignment', 'center');

%% -------------------------------------------------------------------------
% 9. MIMO CORRELATION ANALYSIS
% -------------------------------------------------------------------------
figure(4); clf;

% Extract correlation magnitudes
ant1_sig1_mag = abs([results.ant1_sig1]);
ant1_sig2_mag = abs([results.ant1_sig2]);
ant2_sig1_mag = abs([results.ant2_sig1]);
ant2_sig2_mag = abs([results.ant2_sig2]);

% Plot 1: All 4 correlation magnitudes
subplot(2,2,1)
plot(1:numel(valid_packets), ant1_sig1_mag, 'b-o', 'LineWidth', 1.5); hold on;
plot(1:numel(valid_packets), ant1_sig2_mag, 'r-s', 'LineWidth', 1.5);
plot(1:numel(valid_packets), ant2_sig1_mag, 'g-^', 'LineWidth', 1.5);
plot(1:numel(valid_packets), ant2_sig2_mag, 'm-d', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Packet Number');
ylabel('|Correlation|');
title('MIMO Correlation Magnitudes');
legend('Ant1-Sig1', 'Ant1-Sig2', 'Ant2-Sig1', 'Ant2-Sig2', 'Location', 'best');

% Plot 2: Signal ratios (MIMO indicator)
subplot(2,2,2)
mimo_ratio_ant1 = ant1_sig2_mag ./ (ant1_sig1_mag + eps);
mimo_ratio_ant2 = ant2_sig2_mag ./ (ant2_sig1_mag + eps);
plot(1:numel(valid_packets), mimo_ratio_ant1, 'b-o', 'LineWidth', 1.5); hold on;
plot(1:numel(valid_packets), mimo_ratio_ant2, 'r-s', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Packet Number');
ylabel('Sig2/Sig1 Ratio');
title('MIMO Signal Strength Ratio');
legend('Antenna 1', 'Antenna 2', 'Location', 'best');

% Plot 3: Phase relationships
subplot(2,2,3)
phase_ant1_sig1 = angle([results.ant1_sig1]);
phase_ant1_sig2 = angle([results.ant1_sig2]);
phase_diff_ant1 = rad2deg(phase_ant1_sig2 - phase_ant1_sig1);
plot(1:numel(valid_packets), phase_diff_ant1, 'b-o', 'LineWidth', 1.5);
grid on;
xlabel('Packet Number');
ylabel('Phase Difference (degrees)');
title('Ant1: Phase(Sig2) - Phase(Sig1)');
ylim([-180 180]);

% Plot 4: Complex plane visualization
subplot(2,2,4)
scatter(real([results.ant1_sig1]), imag([results.ant1_sig1]), 50, 'b', 'filled'); hold on;
scatter(real([results.ant1_sig2]), imag([results.ant1_sig2]), 50, 'r', 'filled');
scatter(real([results.ant2_sig1]), imag([results.ant2_sig1]), 50, 'g', 'filled');
scatter(real([results.ant2_sig2]), imag([results.ant2_sig2]), 50, 'm', 'filled');
hold off;
grid on; axis equal;
xlabel('Real'); ylabel('Imag');
title('Correlation Complex Values');
legend('Ant1-Sig1', 'Ant1-Sig2', 'Ant2-Sig1', 'Ant2-Sig2', 'Location', 'best');

%% -------------------------------------------------------------------------
% 10. SUMMARY STATISTICS
% -------------------------------------------------------------------------
fprintf("\n================ SUMMARY STATISTICS ================\n");
fprintf("Total packets processed: %d\n", numel(valid_packets));
fprintf("\nAntenna 1:\n");
fprintf("  Mean BER:    %.6f\n", mean([results.ber1]));
fprintf("  Median BER:  %.6f\n", median([results.ber1]));
fprintf("  Min BER:     %.6f\n", min([results.ber1]));
fprintf("  Max BER:     %.6f\n", max([results.ber1]));
fprintf("\nAntenna 2:\n");
fprintf("  Mean BER:    %.6f\n", mean([results.ber2]));
fprintf("  Median BER:  %.6f\n", median([results.ber2]));
fprintf("  Min BER:     %.6f\n", min([results.ber2]));
fprintf("  Max BER:     %.6f\n", max([results.ber2]));
fprintf("\nCFO Statistics:\n");
fprintf("  Mean CFO:    %.1f Hz\n", mean([results.cfo]));
fprintf("  Std CFO:     %.1f Hz\n", std([results.cfo]));
fprintf("\nMIMO Correlation Statistics:\n");
fprintf("  Mean |Ant1-Sig1|: %.3f\n", mean(ant1_sig1_mag));
fprintf("  Mean |Ant1-Sig2|: %.3f\n", mean(ant1_sig2_mag));
fprintf("  Mean |Ant2-Sig1|: %.3f\n", mean(ant2_sig1_mag));
fprintf("  Mean |Ant2-Sig2|: %.3f\n", mean(ant2_sig2_mag));
fprintf("  Mean MIMO Ratio (Ant1): %.3f\n", mean(mimo_ratio_ant1));
fprintf("  Mean MIMO Ratio (Ant2): %.3f\n", mean(mimo_ratio_ant2));
fprintf("====================================================\n");

%% -------------------------------------------------------------------------
% 11. SAVE RESULTS
% -------------------------------------------------------------------------
final_data.results = results;
final_data.num_packets = numel(valid_packets);
final_data.all_syms_ant1 = all_syms_ant1;
final_data.all_syms_ant2 = all_syms_ant2;
final_data.rx_pos = [0, 0, 0.571]; % UPDATE THIS for each capture
final_data.tx_pos = [-1, 10, 0.875];
final_data.freq = 915e6;
final_data.tx_bf_angle = NaN;  % MANUALLY SET THIS for beamforming captures
final_data.rx_orientation = NaN;
final_data.tx_mimo = true;     % This is the MIMO collection script

% UPDATE filename increment for each new capture
save('nerf_mimo_0_0.mat','final_data');

fprintf('\n✅ Results saved to nerf_mimo_1.mat\n');

release(rxObj);
