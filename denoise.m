clear; clc; close all;

%% PHẦN 1: KỊCH BẢN CHÍNH (MAIN SCRIPT)
input_file = 'noisy_audio.wav';
output_file = 'clean_audio_final.wav';

fprintf('Đang đọc file âm thanh...\n');

[data, fs] = audioread(input_file); % Đọc file: data (tín hiệu), fs (tần số lấy mẫu)
if size(data, 2) > 1; data = mean(data, 2); end % Chuyển Stereo sang Mono
num_samples_to_keep = min(length(data), 2 * fs); % Rút gọn: Chỉ lấy 2 giây đầu
data = data(1:num_samples_to_keep);
total_samples = length(data);

% Thông số Overlap-Add
chunk_size = 1024; % Chiều dài 1 khung (chuẩn N=1024)
hop_size = chunk_size / 2; % Bước nhảy (Overlap 50%)
win = hann(chunk_size); % Cửa sổ Hanning: Giảm rò rỉ phổ ở 2 đầu khung

% Khởi tạo bộ đệm chứa kết quả
cleaned_data_full = zeros(total_samples + chunk_size, 1); 
window_sum = zeros(total_samples + chunk_size, 1); % Lưu trọng số cửa sổ để chuẩn hóa
steps = 1 : hop_size : (total_samples - chunk_size + 1); % Vị trí bắt đầu các khung
total_steps = length(steps);

% Thanh tiến trình
h_bar = waitbar(0, 'Đang phân tích Ma trận Fourier...', 'Name', 'DSP Progress');

% Vòng lặp Frame-by-Frame
for i = 1:total_steps
    start_idx = steps(i);
    end_idx = start_idx + chunk_size - 1;
    
    x_chunk = data(start_idx:end_idx); % Trích xuất 1 khung
    x_win = x_chunk .* win; % Nhân pma trận đường chéo (cửa sổ Hanning)
    
    max_val = max(abs(x_win)); % Lấy biên độ lớn nhất để chuẩn hóa cục b
    if max_val > 0
        x_norm = x_win / max_val;
        
        % ĐSTT: Nhân ma trận đổi cơ sở sang miền tần số
        X_chunk = my_matrix_fft(x_norm);
        
        % Lọc ngưỡng Threshold
        threshold = 0.15 * max(abs(X_chunk));
        X_filt = X_chunk;
        X_filt(abs(X_chunk) < threshold) = 0;
        
        % ĐSTT: Đổi cơ sở ngược về thời gian
        x_res = my_matrix_ifft(X_filt);
        
        % Phục hồi biên độ gốc và nhân cửa sổ lần 2 (chuẩn OLA)
        x_out = x_res * max_val .* win;
        
        % Ghép khung: Cộng dồn kết quả vào bộ đệm (Add)
        cleaned_data_full(start_idx:end_idx) = cleaned_data_full(start_idx:end_idx) + x_out;
        window_sum(start_idx:end_idx) = window_sum(start_idx:end_idx) + win.^2;
    end
    
    % Cập nhật phần trăm tiến trình
    if mod(i, 5) == 0 || i == total_steps
        waitbar(i / total_steps, h_bar, sprintf('Đang tính toán: %.1f%%', (i/total_steps)*100));
    end
end
close(h_bar); % Tắt thanh tiến trình

% Chuẩn hóa cuối cùng: Chia mảng cho tổng trọng số cửa sổ (tránh chia cho 0)
valid_idx = window_sum > 1e-10;
cleaned_data_full(valid_idx) = cleaned_data_full(valid_idx) ./ window_sum(valid_idx);
final_audio = cleaned_data_full(1:total_samples); % Cắt bỏ đuôi dư thừa

audiowrite(output_file, final_audio, fs); % Lưu file
fprintf('==> Đã lưu file sạch: %s\n', output_file);

%% PHẦN 2: VẼ ĐỒ THỊ 2x3 (VISUALIZATION)
% Trích xuất 1 khung ở giữa file để vẽ minh họa
mid_idx = steps(round(total_steps / 2));
sample_noisy = data(mid_idx : mid_idx + chunk_size - 1);
sample_clean = final_audio(mid_idx : mid_idx + chunk_size - 1);

% Tính lại FFT khung đó phục vụ đồ thị
X_sample_orig = my_matrix_fft(sample_noisy);
X_sample_filt = X_sample_orig;
X_sample_filt(abs(X_sample_orig) < 0.15 * max(abs(X_sample_orig))) = 0;

% Cài đặt các trục X (thời gian và tần số)
t_chunk = (0:chunk_size-1)' / fs;
f_chunk = (0:chunk_size-1)' * (fs / chunk_size);
half_N = chunk_size / 2; % Chỉ vẽ phân nửa tần số (tín hiệu thực có phổ đối xứng)

figure('Name', 'Phân tích ĐSTT trong Denoising', 'Position', [50, 50, 1400, 700]);

% --- HÀNG 1: MIỀN THỜI GIAN ---
% Vẽ khung tín hiệu gốc có nhiễu (màu đỏ)
subplot(2, 3, 1); plot(t_chunk, sample_noisy, 'r'); title('1. Sóng âm gốc'); grid on; ylim([-1 1]);

% Vẽ khung tín hiệu sạch sau khi lọc (màu xanh dương)
subplot(2, 3, 2); plot(t_chunk, sample_clean, 'b'); title('2. Sóng âm sau lọc'); grid on; ylim([-1 1]);

% Vẽ đè hai sóng lên nhau: Sóng gốc làm mờ (Color alpha) nằm dưới, sóng sạch nằm trên để so sánh
subplot(2, 3, 3); hold on; plot(t_chunk, sample_noisy, 'Color', [1 0 0 0.3], 'LineWidth', 1); plot(t_chunk, sample_clean, 'b', 'LineWidth', 1.2); hold off; title('3. Overlay Thời gian'); grid on; ylim([-1 1]);

% --- HÀNG 2: MIỀN TẦN SỐ ---
% Vẽ phổ biên độ gốc bằng hàm stem (rời rạc), lấy nửa tần số Nyquist
subplot(2, 3, 4); stem(f_chunk(1:half_N), abs(X_sample_orig(1:half_N)), 'r', 'Marker', '.'); title('4. Phổ gốc |X_k|'); grid on;

% Vẽ phổ biên độ sau khi đã cạo sạch các đỉnh nhiễu nhỏ bằng Threshold
subplot(2, 3, 5); stem(f_chunk(1:half_N), abs(X_sample_filt(1:half_N)), 'b', 'Marker', '.'); title('5. Phổ sau lọc'); grid on;

% Vẽ đè hai phổ lên nhau: Các cọc nhiễu liti màu hồng bị lộ ra vì không có màu xanh đè lên
subplot(2, 3, 6); hold on; stem(f_chunk(1:half_N), abs(X_sample_orig(1:half_N)), 'Color', [1 0.6 0.6], 'Marker', '.'); stem(f_chunk(1:half_N), abs(X_sample_filt(1:half_N)), 'b', 'Marker', '.'); hold off; title('6. Overlay Tần số'); grid on;

%% PHẦN 3: CÁC HÀM LOCAL (LOCAL FUNCTIONS)
function X = my_matrix_fft(x)
    % Tính DFT
    % Hàm tính DFT bằng phép nhân Ma trận (Chuẩn Đại số tuyến tính)
    N = length(x);
    x = x(:); % Đảm bảo x là vector cột (N x 1)
    % Tạo vector hàng n và vector cột k
    n = 0 : N-1;
    k = (0 : N-1)';
    % Xây dựng MA TRẬN FOURIER (Kích thước N x N)
    % Mỗi phần tử F(k,n) = exp(-i * 2*pi * k * n / N)
    F = exp(-1j * 2 * pi * k * n / N);
    % Thực hiện Phép biến đổi tuyến tính: Nhân ma trận với vector
    X = F * x; 
end

function x = my_matrix_ifft(X)
    % Tính IDFT
    N = length(X);
    X = X(:); % Đảm bảo X là vector cột
    n = (0 : N-1)';
    k = 0 : N-1;
    % Xây dựng MA TRẬN FOURIER NGƯỢC (Kích thước N x N)
    % Chú ý: Dấu số mũ dương và chia cho N
    F_inv = exp(1j * 2 * pi * n * k / N) / N;
    % Thực hiện biến đổi ngược: x = F_inv * X
    x_complex = F_inv * X;
    % Lấy phần thực để loại bỏ sai số dấu phẩy động
    x = real(x_complex);
end