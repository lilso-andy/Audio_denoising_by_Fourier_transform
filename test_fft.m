% --- SCRIPT TẠO TÍN HIỆU PHỨC TẠP + NHIỄU TRẮNG (GIỐNG HÌNH PYTHON) ---
fs = 44100; 
t = (0 : fs*2 - 1)' / fs; % Dài 2 giây

% 1. Tạo "Giọng nói" giả lập (Kết hợp nhiều tần số thấp lại với nhau để tạo sóng uốn lượn)
clean_signal = 0.5 * sin(2 * pi * 80 * t) + ...  % Tần số rất trầm
               0.3 * sin(2 * pi * 150 * t) + ... % Tần số trung trầm
               0.2 * sin(2 * pi * 300 * t);      % Tần số trung

% 2. Thêm Nhiễu trắng (White noise) để tạo bề mặt "răng cưa"
% randn sinh ra nhiễu ngẫu nhiên trải đều trên mọi dải tần số
noise = 0.15 * randn(size(t)); 

% 3. Trộn lại
noisy_signal = clean_signal + noise;

% 4. Lưu file
audiowrite('noisy_audio.wav', noisy_signal, fs);
disp('Đã tạo thành công file test phức tạp: noisy_audio.wav');