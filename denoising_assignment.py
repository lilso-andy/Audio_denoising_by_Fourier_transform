import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# --- PART 1: ALGORITHM DEFINITIONS (CORE LOGIC) ---

def manual_dft(x):
    """Calculate Discrete Fourier Transform directly from the mathematical formula"""
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    # Formula: X[k] = sum(x[n] * exp(-2j * pi * k * n / N))
    exponent = -2j * np.pi * k * n / N
    X = np.dot(np.exp(exponent), x)
    return X

def manual_idft(X):
    """Reconstruct the signal from the frequency domain back to the time domain"""
    N = len(X)
    k = np.arange(N)
    n = k.reshape((N, 1))
    # Inverse formula: x[n] = (1/N) * sum(X[k] * exp(2j * pi * k * n / N))
    exponent = 2j * np.pi * k * n / N
    x_reconstructed = np.dot(np.exp(exponent), X) / N
    return np.real(x_reconstructed)

def denoise(X, threshold_ratio=0.1):
    """Denoise based on amplitude threshold (Magnitude Thresholding)"""
    magnitudes = np.abs(X)
    max_mag = np.max(magnitudes)
    X_clean = X.copy()
    # Eliminate frequencies with magnitudes lower than the threshold
    X_clean[magnitudes < (threshold_ratio * max_mag)] = 0
    return X_clean

# --- PART 2: VISUALIZATION UTILITIES ---

def plot_results(x_old, x_new, X_old, X_new):
    """Draw comparison charts as required by the Assignment"""
    N = len(x_old)
    plt.figure(figsize=(14, 10))
    
    # 1. Original Waveform
    plt.subplot(2, 2, 1)
    plt.plot(x_old, color='salmon')
    plt.title("1. Original Waveform (Time Domain)")
    plt.xlabel("Sample Index (n)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    # 2. Filtered Waveform
    plt.subplot(2, 2, 2)
    plt.plot(x_new, color='skyblue')
    plt.title("2. Filtered Waveform (Time Domain)")
    plt.xlabel("Sample Index (n)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    # 3. Original Frequency Spectrum (Magnitude Spectrum)
    plt.subplot(2, 2, 3)
    plt.stem(np.abs(X_old[:N//2]), markerfmt=" ", linefmt='red')
    plt.title("3. Original Frequency Spectrum |X_k|")
    plt.xlabel("Frequency Bin (k)")
    plt.ylabel("Magnitude")
    
    # 4. Filtered Frequency Spectrum
    plt.subplot(2, 2, 4)
    plt.stem(np.abs(X_new[:N//2]), markerfmt=" ", linefmt='green')
    plt.title("4. Filtered Frequency Spectrum |X_filtered_k|")
    plt.xlabel("Frequency Bin (k)")
    plt.ylabel("Magnitude")
    
    plt.tight_layout()
    plt.show()

# --- PART 3: MAIN EXECUTION FLOW (OVERLAP-ADD + PROGRESS + PLOT) ---

if __name__ == "__main__":
    input_file = 'sample2.wav'
    output_file = 'sample2_cleaned_smooth.wav'

    try:
        # 1. Read file and prepare data
        fs, data = wavfile.read(input_file)
        if len(data.shape) > 1: data = data[:, 0]
        data = data.astype(float)

        # Technical parameters for Overlap-Add
        chunk_size = 1024
        hop_size = chunk_size // 2  
        window = np.hanning(chunk_size)
        
        total_samples = len(data)
        cleaned_data_full = np.zeros(total_samples + chunk_size)
        window_sum = np.zeros(total_samples + chunk_size)

        # Calculate total steps for the progress indicator
        steps = range(0, total_samples - chunk_size, hop_size)
        total_steps = len(steps)

        print(f"--- Start processing: {input_file} ({total_samples/fs:.2f} seconds) ---")
        print(f"Technique: Overlap-Add (Hanning Window) | Total chunks: {total_steps}")

        # 2. Overlap-Add loop with progress display
        for i, start in enumerate(steps):
            end = start + chunk_size
            
            x_chunk = data[start:end]
            x_win = x_chunk * window # Apply window to input
            
            max_val = np.max(np.abs(x_win))
            if max_val > 0:
                # Mathematical processing using the defined functions
                X_chunk = manual_dft(x_win / max_val)
                X_filt = denoise(X_chunk, threshold_ratio=0.25)
                x_res = manual_idft(X_filt)
                
                # Apply window a second time and accumulate
                cleaned_data_full[start:end] += x_res * max_val * window
                window_sum[start:end] += window**2

            # Display progress every 20 chunks
            if (i + 1) % 20 == 0 or (i + 1) == total_steps:
                percent = (i + 1) / total_steps * 100
                print(f"Progress: [{i+1}/{total_steps}] - {percent:.1f}% complete")

        # 3. Normalize amplitude after overlap-add
        valid_idx = window_sum > 1e-10
        cleaned_data_full[valid_idx] /= window_sum[valid_idx]
        final_audio = cleaned_data_full[:total_samples]

        # 4. Save results
        wavfile.write(output_file, fs, final_audio.astype(np.int16))
        print(f"\n==> SUCCESS! Saved to: {output_file}")

        # 5. DRAW ILLUSTRATIVE CHARTS (Take a representative chunk in the middle of the file)
        print("Preparing illustrative charts...")
        mid_idx = total_samples // 2
        # Extract a chunk of 1024 samples for plotting
        sample_noisy = data[mid_idx : mid_idx + chunk_size]
        sample_noisy_norm = sample_noisy / np.max(np.abs(sample_noisy))
        
        # Recalculate parameters for this chunk to pass into plot_results
        X_sample_orig = manual_dft(sample_noisy_norm)
        X_sample_filt = denoise(X_sample_orig, threshold_ratio=0.25)
        x_sample_clean = manual_idft(X_sample_filt)
        
        # Call the plotting function
        plot_results(sample_noisy_norm, x_sample_clean, X_sample_orig, X_sample_filt)

    except Exception as e:
        print(f"An error occurred: {e}")