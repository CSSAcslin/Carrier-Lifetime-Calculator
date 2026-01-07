import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy.signal as signal
import scipy.fft as fft


# ==========================================
# 第一部分：信号生成函数 (拆解版)
# ==========================================

def gen_base_square_wave(fs, duration, density=1.0):
    """
    生成基础随机矩形波 (开:1, 闭:0.3)
    """
    total_samples = int(fs * duration)
    sig = np.zeros(total_samples)

    # 转换为采样点数
    min_on = int(10e-3 * fs)
    max_on = int(50e-3 * fs)

    # density 控制间隔密度
    min_off = int((30e-3 * fs) / density)
    max_off = int((80e-3 * fs) / density)

    if min_off < 1: min_off = 1
    if max_off <= min_off: max_off = min_off + 1

    idx = 0
    while idx < total_samples:
        # 生成开信号 (值为1)
        on_len = np.random.randint(min_on, max_on)
        end_on = min(idx + on_len, total_samples)
        sig[idx:end_on] = 1.0

        idx = end_on
        if idx >= total_samples: break

        # 生成闭信号 (值为0.3)
        off_len = np.random.randint(min_off, max_off)
        end_off = min(idx + off_len, total_samples)
        sig[idx:end_off] = 0.3

        idx = end_off

    return sig


def gen_sine_modulation(fs, duration, freq, amp=0.1):
    """
    生成叠加用的正弦波
    """
    total_samples = int(fs * duration)
    t = np.arange(total_samples) / fs
    return amp * np.sin(2 * np.pi * freq * t)


def gen_random_noise(fs, duration,amp , level=0.05):
    """
    生成高斯白噪声
    """
    total_samples = int(fs * duration)
    return np.random.normal(amp, level, total_samples)


def create_composite_signal(fs=1500, duration=5, mod_freq=150, density=1.0, mod_amp=1.0, noise_amp = 2, noise_level=0.05):
    """
    组合上述三个步骤
    """
    # 1. 基础信号
    base_sig = gen_base_square_wave(fs, duration, density)
    # 2. 正弦调制
    sine_sig = gen_sine_modulation(fs, duration, mod_freq, amp=mod_amp)
    # 3. 噪声
    noise_sig = gen_random_noise(fs, duration, amp = noise_amp, level=noise_level)

    # 叠加
    final_sig = base_sig * sine_sig + noise_sig

    return base_sig, final_sig


# ==========================================
# 第二部分：信号分析 (STFT 和 CWT)
# ==========================================
def quality_cwt(sig,fs,totalscales,wavelet):
    cparam = 2 * pywt.central_frequency(wavelet) * totalscales
    scales = cparam / np.arange(totalscales, 1, -1)
    coefficients, frequencies = pywt.cwt(sig, scales, wavelet, sampling_period=1.0 / fs)
    return np.abs(coefficients), frequencies

def analyze_signals(sig, fs, target_freq=150, wavelet = 'cmor1.5-1'):
    """
    对信号进行STFT和CWT变换，并提取目标频率幅值
    """
    N = len(sig)
    xf = fft.rfftfreq(N, 1 / fs)
    yf = fft.rfft(sig)
    psd_fft = np.abs(yf) ** 2 / N  # 简单的功率谱估计


    # --- 1. STFT (短时傅里叶变换) ---
    # nperseg 决定了频率分辨率。fs=1500, nperseg=256 -> 分辨率约为 5.8Hz

    f_stft, t_stft, Zxx = signal.stft(sig, fs, window='hann' ,nperseg=128, noverlap=127, nfft= fs, return_onesided=True)

    # 计算功率谱密度 PSD (取模的平方)
    psd_stft = np.abs(Zxx) ** 2

    # 提取 150Hz 处的幅值
    # 找到频率数组 f_stft 中最接近 target_freq 的索引
    idx_stft = np.argmin(np.abs(f_stft - target_freq))
    # 提取该频率随时间变化的幅值 (取模)
    amp_trace_stft = np.abs(Zxx[idx_stft, :])

    # --- 2. CWT (连续小波变换) ---
    psd_cwt, freqs_cwt = quality_cwt(sig,fs,256,wavelet)
    target_freqs = np.linspace(target_freq - 1 // 2, target_freq + 1 // 2,
                               1)  # totalscales//4
    scales = pywt.frequency2scale(wavelet, target_freqs * 1.0 / fs)
    coefficients, _ = pywt.cwt(sig, scales, wavelet, sampling_period=1.0 / fs)

    magnitude_avg = np.mean(np.abs(coefficients), axis=0) / 3.16 # 这个倍数是有公式能算出来的

    return {
        'fft': (xf, psd_fft),
        'stft': (t_stft, f_stft, psd_stft, amp_trace_stft),
        'cwt': (np.arange(len(sig)) / fs, freqs_cwt, psd_cwt, magnitude_avg)
    }


# ==========================================
# 第三部分：绘图逻辑
# ==========================================

def plot_all_results(base_sig, final_sig, analysis_res, fs, target_freq, wavelet):
    t_stft, f_stft, psd_stft, trace_stft = analysis_res['stft']
    t_cwt, f_cwt, psd_cwt, trace_cwt = analysis_res['cwt']
    f_fft, psd_fft = analysis_res['fft']

    total_time = len(final_sig) / fs
    t_full = np.arange(len(final_sig)) / fs

    # 创建5行1列的布局
    # sharex=False，因为 FFT 的 x轴是频率，不能和时间的 x轴共享
    fig, axes = plt.subplots(5, 1, figsize=(18, 16), constrained_layout=True)

    # === 1. 原始信号 (Time) ===
    ax1 = axes[0]
    ax1.plot(t_full, base_sig, color='red', alpha=1, linestyle='--', label='Base (Rect)')
    ax1.plot(t_full, final_sig, color='blue', alpha=0.5, linewidth=0.8, label='Final Signal')
    ax1.set_title('1. Time Domain Signal')
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right')

    # === 2. FFT 频谱 (Frequency) - 这是一个频域图，X轴不同 ===
    ax2 = axes[1]
    ax2.semilogy(f_fft, psd_fft, color='darkorange', linewidth=1)
    ax2.set_title('2. FFT Power Spectrum (Global)')
    ax2.set_ylabel('Power (Log Scale)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_xlim(0, min(400,fs//2))  # 重点关注低频区
    # 标记目标频率
    ax2.axvline(target_freq, color='red', linestyle='--', alpha=0.6, label=f'Target {target_freq}Hz')
    ax2.legend()

    # === 3. STFT 谱图 (Time-Freq) ===
    ax3 = axes[2]
    # 使用 shading='gouraud' 可以让图像更平滑，且视觉上对齐更准
    mesh_stft = ax3.pcolormesh(t_stft, f_stft, 10 * np.log10(psd_stft + 1e-10),
                               shading='gouraud', cmap='viridis')
    ax3.set_title('3. STFT Spectrogram')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_ylim(0, min(400,fs//2))
    ax3.axhline(target_freq, color='white', linestyle='--', alpha=0.5)
    plt.colorbar(mesh_stft, ax=ax3, label='dB')

    # === 4. CWT 谱图 (Time-Freq) ===
    ax4 = axes[3]
    # CWT 的 x 轴直接使用 t_cwt (即 t_full)
    mesh_cwt = ax4.pcolormesh(t_cwt, f_cwt, psd_cwt, shading='auto', cmap='viridis')
    ax4.set_title('4. CWT Scalogram')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_ylim(1, min(400,fs//2))
    ax4.axhline(target_freq, color='white', linestyle='--', alpha=0.5)
    plt.colorbar(mesh_cwt, ax=ax4, label='Mag')

    # === 5. 目标频率幅值提取 (Time) ===
    ax5 = axes[4]
    ax5.plot(t_stft, trace_stft, label='STFT Trace', color='red', linewidth=1.5)
    ax5.plot(t_cwt, trace_cwt, label='CWT Trace', color='purple', linewidth=1)
    ax5.plot(t_full, base_sig, label='Base (Rect)', color='blue', linewidth=1, alpha=0.2, linestyle='--')
    ax5.set_title(f'5. Extracted Amplitude @ {target_freq}Hz')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Magnitude')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # === 关键步骤：强制对齐所有时间轴 ===
    # 将 ax1, ax3, ax4, ax5 的 X 轴锁定在一起
    # ax2 是 FFT，不参与时间轴锁定
    time_axes = [ax1, ax3, ax4, ax5]

    # 1. 统一设置显示范围
    for ax in time_axes:
        ax.set_xlim(0, total_time)

    # 2. 启用 sharex 机制 (手动链接)
    # 这样你在交互式窗口缩放 ax1 时，ax3/4/5 也会跟着动
    # ax1.get_shared_x_axes().joined(ax1, ax3, ax4, ax5)

    plt.show()


# ==========================================
# 主程序执行
# ==========================================
if __name__ == "__main__":
    # 参数设定
    FS = 1000
    DURATION = 3.0  # 为了绘图清晰，这里生成2秒数据，你可以改为5秒
    MOD_FREQ = 100
    DENSITY = 0.5
    WAVELET = 'cmor3-3'
    noise_amp = 2
    noise_level = 0.4
    mod_amp = 2

    # 1. 生成信号
    base_sig = gen_base_square_wave(FS, DURATION, DENSITY)
    # 2. 正弦调制
    sine_sig = gen_sine_modulation(FS, DURATION, MOD_FREQ, amp=mod_amp)
    # 3. 噪声
    inner_noise_sig = gen_random_noise(FS, DURATION, amp =noise_amp , level=noise_level)
    outer_noise_sig = gen_random_noise(FS, DURATION, amp=5, level=noise_level)
    # 叠加
    full_signal = (base_sig + inner_noise_sig) * sine_sig + outer_noise_sig

    full_signal = full_signal - np.mean(full_signal[0:300])

    # 2. 分析信号
    results = analyze_signals(full_signal, FS, target_freq=MOD_FREQ, wavelet=WAVELET)

    # 3. 绘图
    plot_all_results(base_sig, full_signal, results, FS, target_freq=MOD_FREQ, wavelet=WAVELET)