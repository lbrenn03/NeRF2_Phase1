clear;
%% =========================================================================
%  NeRF2 RECEIVER — 2 s Offline Processing (Top 100 Packets)
% =========================================================================

x_lab = -1;                          
y_lab = 9;  

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

%% -------------------------------------------------------------------------
% 5. PACKET DETECTION (XCORR) - Local Peaks with Time-Based Spacing
% -------------------------------------------------------------------------
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
% 6. PROCESS EACH PACKET
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
results = struct('ber1', [], 'ber2', [], 'numErr1', [], 'numErr2', [], ...
                 'cfo', [], 'peakVal', [], 'peakTime', []);

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
    
    %% Fine Synchronization
    [fine_corr1, fine_lags] = xcorr(mf_ant1, ref_syms, 'normalized');
    [~, fine_idx] = max(abs(fine_corr1));
    
    fine_timing_offset = fine_lags(fine_idx);
    fine_phase_ant1 = angle(fine_corr1(fine_idx));
    
    [fine_corr2, ~] = xcorr(mf_ant2, ref_syms, 'normalized');
    fine_phase_ant2 = angle(fine_corr2(fine_idx));
    
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
    
    %% Apply phase correction
    syms_ant1_corrected = syms_ant1 * exp(-1j * fine_phase_ant1);
    syms_ant2_corrected = syms_ant2 * exp(-1j * fine_phase_ant2);
    
    %% Store symbols for constellation analysis
    all_syms_ant1(:, p) = syms_ant1;
    all_syms_ant2(:, p) = syms_ant2;
    
    %% Calculate cluster centers for this packet using k-means
    % Use k-means to find 4 clusters in the rotated constellation
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
    
    %% Store results
    results(p).ber1 = ber1;
    results(p).ber2 = ber2;
    results(p).numErr1 = numErr1;
    results(p).numErr2 = numErr2;
    results(p).cfo = estCFO;
    results(p).peakVal = valid_packets(p).peakVal;
    results(p).peakTime = valid_packets(p).peakTime;
    
    if mod(p, 10) == 0
        fprintf('Processed %d / %d packets\n', p, numel(valid_packets));
    end
end

%% -------------------------------------------------------------------------
% 7. COMPUTE CLUSTER MEANS PER PACKET (4 points per packet)
% -------------------------------------------------------------------------
% For each packet, separate symbols by QPSK constellation point and average
cluster_means_ant1 = zeros(4, numel(valid_packets));
cluster_means_ant2 = zeros(4, numel(valid_packets));

for p = 1:numel(valid_packets)
    syms_ant1 = all_syms_ant1(:, p);
    syms_ant2 = all_syms_ant2(:, p);
    
    % Use k-means to identify which symbols belong to which cluster
    try
        [idx1, ~] = kmeans([real(syms_ant1), imag(syms_ant1)], 4, ...
            'MaxIter', 100, 'Replicates', 3);
        
        % Average symbols in each cluster
        for c = 1:4
            cluster_syms = syms_ant1(idx1 == c);
            if ~isempty(cluster_syms)
                cluster_means_ant1(c, p) = mean(cluster_syms);
            else
                cluster_means_ant1(c, p) = NaN;
            end
        end
    catch
        cluster_means_ant1(:, p) = NaN;
    end
    
    try
        [idx2, ~] = kmeans([real(syms_ant2), imag(syms_ant2)], 4, ...
            'MaxIter', 100, 'Replicates', 3);
        
        for c = 1:4
            cluster_syms = syms_ant2(idx2 == c);
            if ~isempty(cluster_syms)
                cluster_means_ant2(c, p) = mean(cluster_syms);
            else
                cluster_means_ant2(c, p) = NaN;
            end
        end
    catch
        cluster_means_ant2(:, p) = NaN;
    end
end

%% -------------------------------------------------------------------------
% 8. DIAGNOSTIC CONSTELLATION FIGURE (replaces averaged constellation)
%    Shows 4 individual packets spread across the capture + summary metrics
% -------------------------------------------------------------------------

% Pick 4 representative packet indices spread across the capture
num_p = numel(valid_packets);
sample_idxs = unique(round(linspace(1, num_p, min(4, num_p))));

figure(2); clf;
tiledlayout(3, max(numel(sample_idxs), 2), 'TileSpacing', 'compact', 'Padding', 'compact');

for si = 1:numel(sample_idxs)
    p = sample_idxs(si);
    syms1 = all_syms_ant1(:, p);
    syms2 = all_syms_ant2(:, p);

    % --- Per-packet EVM (relative to k-means cluster centers) ---
    try
        [~, c1] = kmeans([real(syms1), imag(syms1)], 4, 'MaxIter', 100, 'Replicates', 3);
        c1c = c1(:,1) + 1j*c1(:,2);
        % Assign each symbol to nearest center
        dists1 = abs(syms1 - c1c.');   % 500 x 4
        [~, assign1] = min(dists1, [], 2);
        evm1 = mean(abs(syms1 - c1c(assign1)).^2);
    catch
        evm1 = NaN;
    end

    try
        [~, c2] = kmeans([real(syms2), imag(syms2)], 4, 'MaxIter', 100, 'Replicates', 3);
        c2c = c2(:,1) + 1j*c2(:,2);
        dists2 = abs(syms2 - c2c.');
        [~, assign2] = min(dists2, [], 2);
        evm2 = mean(abs(syms2 - c2c(assign2)).^2);
    catch
        evm2 = NaN;
    end

    % --- Row 1: Ant 1 individual constellation ---
    nexttile(si)
    scatter(real(syms1), imag(syms1), 8, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
    grid on; axis equal;
    title(sprintf('Pkt %d | Ant1 | EVM=%.3f', p, evm1), 'FontSize', 8);
    xlabel('I'); ylabel('Q');

    % --- Row 2: Ant 2 individual constellation ---
    nexttile(si + max(numel(sample_idxs), 2))
    scatter(real(syms2), imag(syms2), 8, 'r', 'filled', 'MarkerFaceAlpha', 0.3);
    grid on; axis equal;
    title(sprintf('Pkt %d | Ant2 | EVM=%.3f', p, evm2), 'FontSize', 8);
    xlabel('I'); ylabel('Q');
end

% --- Row 3: Correlation peak strength across all packets (stability indicator) ---
peak_vals = [valid_packets.peakVal];
peak_times = [valid_packets.peakTime];

nexttile([1, max(numel(sample_idxs), 2)])
plot(peak_times, peak_vals / max(peak_vals), 'k.-', 'MarkerSize', 12, 'LineWidth', 1.2);
yline(0.75, 'r--', '75% threshold', 'LabelHorizontalAlignment', 'left');
grid on;
xlabel('Capture Time (s)');
ylabel('Normalized Peak Strength');
title(sprintf('Correlation Peak Stability | CV = %.3f', std(peak_vals)/mean(peak_vals)));
ylim([0 1.1]);


%% -------------------------------------------------------------------------
% 9. DENOISING ANALYSIS - MSE vs Number of Packets Averaged  
% -------------------------------------------------------------------------
max_N = numel(valid_packets);
mse_vs_N_ant1 = zeros(4, max_N);  % 4 clusters x N packets
mse_vs_N_ant2 = zeros(4, max_N);

fprintf('\nComputing denoising curve...\n');
for N = 1:max_N
    % Average over first N packets
    avg_N_ant1 = mean(all_syms_ant1(:, 1:N), 2);
    avg_N_ant2 = mean(all_syms_ant2(:, 1:N), 2);
    
    % K-means clustering on averaged symbols
    try
        [idx1, centers1] = kmeans([real(avg_N_ant1), imag(avg_N_ant1)], 4, ...
            'MaxIter', 100, 'Replicates', 3);
        centers1_complex = centers1(:,1) + 1j*centers1(:,2);
        
        % Compute MSE for each cluster
        for c = 1:4
            cluster_syms = avg_N_ant1(idx1 == c);
            if ~isempty(cluster_syms)
                mse_vs_N_ant1(c, N) = mean(abs(cluster_syms - centers1_complex(c)).^2);
            else
                mse_vs_N_ant1(c, N) = NaN;
            end
        end
    catch
        mse_vs_N_ant1(:, N) = NaN;
    end
    
    try
        [idx2, centers2] = kmeans([real(avg_N_ant2), imag(avg_N_ant2)], 4, ...
            'MaxIter', 100, 'Replicates', 3);
        centers2_complex = centers2(:,1) + 1j*centers2(:,2);
        
        for c = 1:4
            cluster_syms = avg_N_ant2(idx2 == c);
            if ~isempty(cluster_syms)
                mse_vs_N_ant2(c, N) = mean(abs(cluster_syms - centers2_complex(c)).^2);
            else
                mse_vs_N_ant2(c, N) = NaN;
            end
        end
    catch
        mse_vs_N_ant2(:, N) = NaN;
    end
    
    if mod(N, 10) == 0
        fprintf('  N = %d / %d\n', N, max_N);
    end
end

% Plot MSE curves
figure(3); clf;
subplot(1,2,1)
colors = lines(4);
hold on
for c = 1:4
    plot(1:max_N, mse_vs_N_ant1(c, :), 'Color', colors(c,:), 'LineWidth', 1.5);
end
hold off
grid on;
set(gca, 'YScale', 'log');
xlabel('Number of Packets Averaged');
ylabel('Mean Squared Error');
title('Denoising: MSE vs N - Ant 1');
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Location', 'best');

subplot(1,2,2)
hold on
for c = 1:4
    plot(1:max_N, mse_vs_N_ant2(c, :), 'Color', colors(c,:), 'LineWidth', 1.5);
end
hold off
grid on;
set(gca, 'YScale', 'log');
xlabel('Number of Packets Averaged');
ylabel('Mean Squared Error');
title('Denoising: MSE vs N - Ant 2');
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Location', 'best');

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
fprintf("====================================================\n");

%% -------------------------------------------------------------------------
% 11. SAVE RESULTS
% -------------------------------------------------------------------------
final_data.results = results;
final_data.num_packets = numel(valid_packets);
final_data.all_syms_ant1 = all_syms_ant1;
final_data.all_syms_ant2 = all_syms_ant2;
final_data.cluster_means_ant1 = cluster_means_ant1;
final_data.cluster_means_ant2 = cluster_means_ant2;
final_data.rx_pos = [x_lab, y_lab, 0.571];
final_data.tx_pos = [13, 8, 0.875];
final_data.freq = 915e6;
final_data.tx_bf_angle = NaN;
final_data.rx_orientation = NaN;
final_data.tx_mimo = NaN;

if x_lab >= 0
    filename = 'pilot_data/tx_13_8/x-' + string(x_lab) + '/nerf_' + string(x_lab) + '_' + string(y_lab) + '.mat'
else
    filename = 'pilot_data/tx_13_8/x-n1/nerf_n1_' + string(y_lab) + '.mat'
end

save(filename,'final_data'); % increment to save to nerf_(n+1).mat

release(rxObj);
