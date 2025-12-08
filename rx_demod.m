clear all;
%% ==============================================================================
%  NeRF2 RECEIVER (5s Offline Processing)
% ==============================================================================

% 1. REFERENCE AND CONFIG
Fs = 500e3;
CaptureTime = 5; % seconds
TotalSamples = round(Fs * CaptureTime);

% Generate Reference (Must match TX exactly)
pn = comm.PNSequence('Polynomial', [10 3 0], 'InitialConditions', ones(1,10), 'SamplesPerFrame', 1023);
ref_bits = pn();
if mod(length(ref_bits), 2) ~= 0, ref_bits = [ref_bits; 0]; end
qpskMod = comm.QPSKModulator('BitInput',true, 'PhaseOffset', pi/4);
ref_syms = qpskMod(ref_bits);
sps = 5;
rrcTx = comm.RaisedCosineTransmitFilter('RolloffFactor', 0.25, 'FilterSpanInSymbols', 6, 'OutputSamplesPerSymbol', sps);
reference_sig = rrcTx(ref_syms);
% Normalize reference for correlation
reference_sig = reference_sig / max(abs(reference_sig));
N = length(reference_sig); % Length of the known sounding packet

% 2. HARDWARE CONFIG (Single Capture)
% SamplesPerFrame is set to the total capture size to ensure a single, large read.
rxObj = comm.SDRuReceiver(...
    'Platform',             'B210', ...
    'SerialNum',            '34C78FD', ...
    'MasterClockRate',      30e6, ...
    'DecimationFactor',     60, ...
    'SamplesPerFrame',      TotalSamples, ... % <-- Set to 5 seconds worth of samples
    'CenterFrequency',      915e6, ...
    'Gain',                 60, ...
    'ChannelMapping',       [1 2], ...
    'OutputDataType',       'double');

% 3. PROCESSING OBJECTS
cfoObj = comm.CoarseFrequencyCompensator('Modulation','QPSK','SampleRate',Fs, 'FrequencyResolution', 50);

disp(['Capturing ', num2str(CaptureTime), ' seconds of data...']);
figure(1); clf;

% A. CAPTURE DATA (Single Call)
% This blocks the code until all samples are received, eliminating overruns.
[rxBuffer, ~, overrun] = rxObj(); 

if overrun, warning('Overrun detected during capture! Try reducing SampleRate or CaptureTime.'); end
if size(rxBuffer, 1) < TotalSamples, warning('Incomplete capture.'); end

disp('Capture complete. Starting offline processing.');

% 4. OFFLINE PACKET DETECTION
[c_norm, lags] = xcorr(rxBuffer(:,1), reference_sig); 

% Find all peaks above threshold
[pks, locs] = findpeaks(abs(c_norm), 'MinPeakHeight', normalizedThreshold, 'SortStr', 'descend');

% Filter to only keep peaks with full packet within buffer
validPeaks = [];
validLocs = [];
for i = 1:length(pks)
    lag = lags(locs(i));
    startIdx = lag + 1;
    endIdx = startIdx + N - 1;
    
    % Check if packet fits completely in buffer
    if startIdx > 0 && endIdx <= size(rxBuffer, 1)
        validPeaks(end+1) = pks(i);
        validLocs(end+1) = locs(i);
    end
end

% Take the best valid packet
if ~isempty(validPeaks)
    [maxVal, bestIdx] = max(validPeaks);
    idx = validLocs(bestIdx);
    lag = lags(idx);
    startIdx = lag + 1;
end
% B. THRESHOLD CHECK (Normalized correlation peak)
% A value near 1.0 is a perfect match. Set a confident threshold, e.g., 0.6.
normalizedThreshold = 15; 

if maxVal > normalizedThreshold
    lag = lags(idx);
    startIdx = lag + 1;
    
    fprintf('✅ Packet detected! Normalized Peak: %.4f at Time: %.3f s\n', ...
        maxVal, (startIdx / Fs));
    
    % C. EXTRACT, PROCESS, AND SAVE
    
    % Ensure extraction is within bounds
    if startIdx > 0 && (startIdx + N - 1 <= size(rxBuffer, 1))
        
        % Extraction (Both Antennas)
        packet_ant1 = rxBuffer(startIdx : startIdx + N - 1, 1);
        packet_ant2 = rxBuffer(startIdx : startIdx + N - 1, 2);
        
        % CFO Correction
        [~, estCFO] = cfoObj(packet_ant1);
        t = (0:N-1).' / Fs;
        cfo_vector = exp(-1i * 2 * pi * estCFO * t);
        
        clean_ant1 = packet_ant1 .* cfo_vector;
        clean_ant2 = packet_ant2 .* cfo_vector;

  

        % D. VISUALIZATION
        subplot(2,2,1); plot(lags, abs(c_norm)); 
        hold on; plot(lags([1 end]), [normalizedThreshold normalizedThreshold], 'r--'); hold off;
        title('Normalized Correlation Peak'); xlabel('Lag (Samples)'); ylabel('Magnitude');
        
        subplot(2,2,2); plot(abs(clean_ant1)); title('Packet Envelope (Ant 1)');
        
        %subplot(2,2,3); scatterplot(clean_ant1(1:10:end), 1, 0, 'b.'); 
        %title(['Constellation (CFO:', num2str(estCFO, '%.1f'), ' Hz)']);

        subplot(2,2,4); scatterplot(clean_ant2(1:10:end), 1, 0, 'r.'); 
        title('Constellation (Ant 2)');

        %% ======================================================================
        %  5. MATCHED FILTER + DEMODULATION + BER
        % ======================================================================
        
        % Create receive RRC filter (must match TX)
        rrcRx = comm.RaisedCosineReceiveFilter( ...
            'RolloffFactor', 0.25, ...
            'FilterSpanInSymbols', 6, ...
            'InputSamplesPerSymbol', sps, ...
            'DecimationFactor', sps);
        
        % Apply matched filter to both antennas
        mf_ant1 = rrcRx(clean_ant1);
        mf_ant2 = rrcRx(clean_ant2);
        
        % Remove filter transient (span = 6 symbols)
        filterDelay = 6;   % symbols
        mf_ant1 = mf_ant1(filterDelay+1:end);
        mf_ant2 = mf_ant2(filterDelay+1:end);
        
        % QPSK demodulator
        qpskDemod = comm.QPSKDemodulator('BitOutput',true,'PhaseOffset',pi/4);
        
        % Demodulate
        rx_bits_ant1 = qpskDemod(mf_ant1);
        rx_bits_ant2 = qpskDemod(mf_ant2);
        
        % Compute actual length after processing
        L = min(length(ref_bits), length(rx_bits_ant1));
        
        % Trim both reference and received to same length
        ref_bits_trimmed = ref_bits(1:L);
        rx_bits_ant1 = rx_bits_ant1(1:L);
        rx_bits_ant2 = rx_bits_ant2(1:L);
        
        % Compute bit error rates
        [numErr1, ber1] = biterr(ref_bits_trimmed, rx_bits_ant1);
        [numErr2, ber2] = biterr(ref_bits_trimmed, rx_bits_ant2);
        
        fprintf("\n================ BER RESULTS ================\n");
        fprintf("Antenna 1: Bit Errors = %d / %d   BER = %.6f\n", numErr1, L, ber1);
        fprintf("Antenna 2: Bit Errors = %d / %d   BER = %.6f\n", numErr2, L, ber2);
        fprintf("=============================================\n");
        

        % E. SAVE FINAL DATA
        final_data.clean_ant1 = clean_ant1;
        final_data.clean_ant2 = clean_ant2;
        final_data.cfo = estCFO;
        
        save('nerf_single_capture_offline.mat', 'final_data');
        
        disp("Data successfully processed and saved to nerf_single_capture_offline.mat");

    else
        fprintf('⚠️ Error: Detected packet index (%d) is too close to the buffer edge.\n', startIdx);
    end

else
    disp(['❌ Packet not found. Max correlation peak was ', num2str(maxVal, '%.4f'), '.']);
    
end

release(rxObj);
