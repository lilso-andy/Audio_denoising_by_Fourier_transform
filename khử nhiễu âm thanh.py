import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# --- PHẦN 1: ĐỊNH NGHĨA CÁC THUẬT TOÁN (CORE LOGIC) ---

def manual_dft(x):
    """Tính toán Biến đổi Fourier rời rạc trực tiếp từ công thức toán học"""
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    # Công thức: X[k] = sum(x[n] * exp(-2j * pi * k * n / N))
    exponent = -2j * np.pi * k * n / N
    X = np.dot(np.exp(exponent), x)
    return X

def manual_idft(X):
    """Tái tạo tín hiệu từ miền tần số quay lại miền thời gian"""
    N = len(X)
    k = np.arange(N)
    n = k.reshape((N, 1))
    # Công thức ngược: x[n] = (1/N) * sum(X[k] * exp(2j * pi * k * n / N))
    exponent = 2j * np.pi * k * n / N
    x_reconstructed = np.dot(np.exp(exponent), X) / N
    return np.real(x_reconstructed)

def denoise(X, threshold_ratio=0.1):
    """Lọc nhiễu dựa trên ngưỡng biên độ (Magnitude Thresholding)"""
    magnitudes = np.abs(X)
    max_mag = np.max(magnitudes)
    X_clean = X.copy()
    # Triệt tiêu các tần số có biên độ thấp hơn ngưỡng
    X_clean[magnitudes < (threshold_ratio * max_mag)] = 0
    return X_clean

# --- PHẦN 2: HÀM HỖ TRỢ MINH HỌA (VISUALIZATION) ---

def plot_results(x_old, x_new, X_old, X_new):
    """Vẽ biểu đồ so sánh theo yêu cầu của Assignment"""
    N = len(x_old)
    plt.figure(figsize=(14, 10))
    
    # 1. Sóng âm gốc
    plt.subplot(2, 2, 1)
    plt.plot(x_old, color='salmon')
    plt.title("1. Sóng âm gốc (Miền thời gian)")
    plt.grid(True, alpha=0.3)
    
    # 2. Sóng âm sau lọc
    plt.subplot(2, 2, 2)
    plt.plot(x_new, color='skyblue')
    plt.title("2. Sóng âm sau khi lọc (Miền thời gian)")
    plt.grid(True, alpha=0.3)
    
    # 3. Phổ tần số gốc (Magnitude Spectrum)
    plt.subplot(2, 2, 3)
    plt.stem(np.abs(X_old[:N//2]), markerfmt=" ", linefmt='red')
    plt.title("3. Phổ tần số gốc |X_k|")
    
    # 4. Phổ tần số sau lọc
    plt.subplot(2, 2, 4)
    plt.stem(np.abs(X_new[:N//2]), markerfmt=" ", linefmt='green')
    plt.title("4. Phổ tần số sau khi lọc |X_filtered_k|")
    
    plt.tight_layout()
    plt.show()

# --- PHẦN 3: LUỒNG THỰC THI CHÍNH (OVERLAP-ADD + PROGRESS + PLOT) ---

if __name__ == "__main__":
    input_file = 'sample.wav'
    output_file = 'sample_cleaned_smooth.wav'

    try:
        # 1. Đọc file và chuẩn bị
        fs, data = wavfile.read(input_file)
        if len(data.shape) > 1: data = data[:, 0]
        data = data.astype(float)

        # Thông số kỹ thuật cho Overlap-Add
        chunk_size = 1024
        hop_size = chunk_size // 2  
        window = np.hanning(chunk_size)
        
        total_samples = len(data)
        cleaned_data_full = np.zeros(total_samples + chunk_size)
        window_sum = np.zeros(total_samples + chunk_size)

        # Tính toán tổng số bước để làm thanh tiến trình
        steps = range(0, total_samples - chunk_size, hop_size)
        total_steps = len(steps)

        print(f"--- Bắt đầu xử lý: {input_file} ({total_samples/fs:.2f} giây) ---")
        print(f"Kỹ thuật: Overlap-Add (Hanning Window) | Tổng số đoạn: {total_steps}")

        # 2. Vòng lặp Overlap-Add có hiển thị tiến trình
        for i, start in enumerate(steps):
            end = start + chunk_size
            
            x_chunk = data[start:end]
            x_win = x_chunk * window # Nhân cửa sổ đầu vào
            
            max_val = np.max(np.abs(x_win))
            if max_val > 0:
                # Xử lý toán học bằng các hàm LilSoAndy đã viết
                X_chunk = manual_dft(x_win / max_val)
                X_filt = denoise(X_chunk, threshold_ratio=0.15)
                x_res = manual_idft(X_filt)
                
                # Nhân cửa sổ lần 2 và cộng dồn
                cleaned_data_full[start:end] += x_res * max_val * window
                window_sum[start:end] += window**2

            # Hiển thị tiến trình mỗi 20 đoạn cho đỡ sốt ruột
            if (i + 1) % 20 == 0 or (i + 1) == total_steps:
                percent = (i + 1) / total_steps * 100
                print(f"Tiến độ: [{i+1}/{total_steps}] - {percent:.1f}% hoàn thành")

        # 3. Chuẩn hóa biên độ sau khi ghép
        valid_idx = window_sum > 1e-10
        cleaned_data_full[valid_idx] /= window_sum[valid_idx]
        final_audio = cleaned_data_full[:total_samples]

        # 4. Lưu kết quả
        wavfile.write(output_file, fs, final_audio.astype(np.int16))
        print(f"\n==> THÀNH CÔNG! Đã lưu: {output_file}")

        # 5. VẼ ĐỒ THỊ MINH HỌA (Lấy 1 đoạn tiêu biểu ở giữa file)
        print("Đang chuẩn bị đồ thị minh họa...")
        mid_idx = total_samples // 2
        # Cắt 1 đoạn 1024 mẫu để vẽ
        sample_noisy = data[mid_idx : mid_idx + chunk_size]
        sample_noisy_norm = sample_noisy / np.max(np.abs(sample_noisy))
        
        # Tính toán lại thông số đoạn này để truyền vào hàm plot_results
        X_sample_orig = manual_dft(sample_noisy_norm)
        X_sample_filt = denoise(X_sample_orig, threshold_ratio=0.15)
        x_sample_clean = manual_idft(X_sample_filt)
        
        # Gọi hàm vẽ đồ thị của cậu
        plot_results(sample_noisy_norm, x_sample_clean, X_sample_orig, X_sample_filt)

    except Exception as e:
        print(f"Lỗi rồi bạn ơi: {e}")