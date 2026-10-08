import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys

# Tăng giới hạn đệ quy của Python (đề phòng file quá dài, dù chunk 1024 thì đệ quy sâu 10 tầng là đủ)
sys.setrecursionlimit(2000)

# --- PART 1: ALGORITHM DEFINITIONS (FFT CORE LOGIC) ---

def custom_fft(x):
    """
    Calculate Fast Fourier Transform using Radix-2 Cooley-Tukey algorithm.
    Note: The length of x must be a power of 2 (e.g., 1024, 2048).
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    
    # Base case for recursion
    if N <= 1:
        return x
    
    # Divide: Split the array into even and odd indices
    even = custom_fft(x[0::2])
    odd = custom_fft(x[1::2])
    
    # Conquer: Combine using the twiddle factors (Butterfly operation)
    # Twiddle factor: exp(-2j * pi * k / N)
    factor = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    T = factor * odd
    
    # Concatenate the lower half and upper half of the spectrum
    return np.concatenate([even + T, even - T])

def custom_ifft(X):
    """
    Reconstruct the signal using the Inverse FFT via the Conjugate Trick.
    Formula: x = (1/N) * (FFT(X*))*
    """
    X = np.asarray(X, dtype=complex)
    N = len(X)
    
    # 1. Take the complex conjugate of the frequency spectrum
    X_star = np.conjugate(X)
    
    # 2. Pass it through the standard forward FFT function
    fft_star = custom_fft(X_star)
    
    # 3. Take the complex conjugate again and divide by N
    x = np.conjugate(fft_star) / N
    
    return np.real(x)

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
    output_file = 'sample2_fft_cleaned.wav'

    try:
        # 1. Read file and prepare data
        fs, data = wavfile.read(input_file)
        if len(data.shape) > 1: data = data[:, 0]
        data = data.astype(float)

        # Technical parameters for Overlap-Add
        # CRITICAL: chunk_size MUST be a power of 2 for Radix-2 FFT to work (e.g., 512, 1024, 2048)
        chunk_size = 1024
        hop_size = chunk_size // 2  
        window = np.hanning(chunk_size)
        
        total_samples = len(data)
        cleaned_data_full = np.zeros(total_samples + chunk_size)
        window_sum = np.zeros(total_samples + chunk_size)

        steps = range(0, total_samples - chunk_size, hop_size)
        total_steps = len(steps)

        print(f"--- Start processing: {input_file} ({total_samples/fs:.2f} seconds) ---")
        print(f"Technique: Fast Fourier Transform (FFT) + Overlap-Add | Total chunks: {total_steps}")

        # 2. Overlap-Add loop with progress display
        for i, start in enumerate(steps):
            end = start + chunk_size
            
            x_chunk = data[start:end]
            x_win = x_chunk * window 
            
            max_val = np.max(np.abs(x_win))
            if max_val > 0:
                # Using the custom FFT and IFFT functions
                X_chunk = custom_fft(x_win / max_val)
                X_filt = denoise(X_chunk, threshold_ratio=0.45)
                x_res = custom_ifft(X_filt)
                
                cleaned_data_full[start:end] += x_res * max_val * window
                window_sum[start:end] += window**2

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

        # 5. DRAW ILLUSTRATIVE CHARTS
        print("Preparing illustrative charts...")
        mid_idx = total_samples // 2
        sample_noisy = data[mid_idx : mid_idx + chunk_size]
        sample_noisy_norm = sample_noisy / np.max(np.abs(sample_noisy))
        
        # Note: I fixed the threshold_ratio back to 0.25 here so the graph looks normal, 
        # instead of the 0.99 from your experimental code!
        X_sample_orig = custom_fft(sample_noisy_norm)
        X_sample_filt = denoise(X_sample_orig, threshold_ratio=0.45)
        x_sample_clean = custom_ifft(X_sample_filt)
        
        plot_results(sample_noisy_norm, x_sample_clean, X_sample_orig, X_sample_filt)

    except Exception as e:
        print(f"An error occurred: {e}")