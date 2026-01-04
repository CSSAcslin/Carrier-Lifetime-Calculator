import copy
import logging
import numpy as np
import pywt
import h5py
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QElapsedTimer, QThread
from typing import List
from scipy import signal
from scipy.optimize import curve_fit
from scipy.ndimage import zoom
import DataManager
from multiprocessing import shared_memory, Manager, Pool
import math
import traceback


# --- 全局 Worker 函数 ---
def _stft_worker_process(
        input_shm_name, input_shape, input_dtype,
        output_shm_name, output_shape, output_dtype,
        pixel_range,
        fps, window, window_size, noverlap, nfft, target_idx, batch_size,
        queue
):
    """
    独立进程运行的函数：读取共享内存 -> 计算 -> 写入共享内存
    """
    # 1. 连接共享内存
    existing_shm_in = shared_memory.SharedMemory(name=input_shm_name)
    existing_shm_out = shared_memory.SharedMemory(name=output_shm_name)

    try:
        # 2. 通过 buffer 重构 NumPy 数组 (零拷贝)
        unfolded_data = np.ndarray(input_shape, dtype=input_dtype, buffer=existing_shm_in.buf)
        stft_out_flat = np.ndarray(output_shape, dtype=output_dtype, buffer=existing_shm_out.buf)

        # 3. 获取该进程负责的像素范围
        start_global, end_global = pixel_range
        total_pixels_in_task = end_global - start_global

        # 4. 内部循环：Cache 友好的小 Batch 处理
        #    为了保护 L3 Cache，我们不在进程内一次性算完 3万个像素
        #    而是每次算 64-128 个像素
        CACHE_BATCH_SIZE = batch_size

        local_processed_count = 0

        for b_start in range(start_global, end_global, CACHE_BATCH_SIZE):
            b_end = min(b_start + CACHE_BATCH_SIZE, end_global)

            # --- 核心计算逻辑 (同之前的优化版) ---
            batch_data = unfolded_data[b_start:b_end, :]  # [Batch, Time]

            # STFT
            _, _, Zxx_batch = signal.stft(
                batch_data, fs=fps, window=window,
                nperseg=window_size, noverlap=noverlap, nfft=nfft,
                axis=-1
            )

            # 提取目标频率
            target_data = Zxx_batch[:, target_idx, :]
            mag_batch = np.abs(target_data)

            if mag_batch.ndim == 3:
                mag_batch = np.mean(mag_batch, axis=1)

            mag_batch *= 560

            # 写入输出共享内存
            # 注意：stft_out_flat 的形状是 (out_length, total_pixels)
            # 我们需要转置 mag_batch (Batch, Time) -> (Time, Batch)
            valid_len = min(mag_batch.shape[1], stft_out_flat.shape[0])
            stft_out_flat[:valid_len, b_start:b_end] = mag_batch.T[:valid_len, :]

            # 5. 上报进度
            #    为了防止 Queue 拥堵，每处理一定量才发一次
            local_processed_count += (b_end - b_start)
            if local_processed_count >= 1000 or b_end == end_global:
                queue.put(local_processed_count)
                local_processed_count = 0

    except Exception as e:
        traceback.print_exc()
        raise f"Worker Error: {e}"
    finally:
        # 清理连接（不删除内存，只断开连接）
        existing_shm_in.close()
        existing_shm_out.close()

class DataProcessor(QObject):
    """本类包含所有非计算流程的操作（常开线程）"""
    plot_singal = pyqtSignal(np.ndarray,dict)
    def __init__(self):
        super().__init__()
        logging.info("例外数据处理线程已启动")

    @staticmethod
    def process_data(data, max_all, min_all, vmean_array):
        process_show = []
        if np.abs(min_all) > np.abs(max_all):
            # n-type 信号中心为黑色，最强值为负
            data_type = 'central negative'
            for every_data in data:
                normalized_data = (every_data - min_all) / (max_all - min_all)
                process_show.append(normalized_data)
            max_mean = np.min(vmean_array)
            phy_max = -min_all
            phy_min = -max_all
        else:
            # p-type 信号中心为白色，最强值为正
            data_type = 'central positive'
            for every_data in data:
                normalized_data = (max_all - every_data) / (max_all - min_all)
                process_show.append(normalized_data)
            max_mean = np.max(vmean_array)
            phy_max = max_all
            phy_min = min_all
        return process_show, data_type, max_mean, phy_max, phy_min

    @pyqtSlot(object)
    def amend_data(self, data):
        """函数修改方法
        输入修改的源数据，导出修改的数据包"""
        data_origin = data
        vmax_array = []
        vmin_array = []
        vmean_array = []
        for data in data_origin:
            vmax_array.append(np.max(data))
            vmin_array.append(np.min(data))
            vmean_array.append(np.mean(data))
        vmax = np.max(vmax_array)
        vmin = np.min(vmin_array)

        images_show, data_type, max_mean, phy_max, phy_min = self.process_data(data_origin, vmax, vmin, vmean_array)

        return {
            'data_origin' : data_origin,
            'image_import': np.stack(images_show, axis=0),
        }

    @pyqtSlot(np.ndarray,float)
    def detect_bad_frames_auto(self, data: np.ndarray, threshold: float = 3.0) -> List[int]:
        """
        自动检测坏帧
        基于帧间差异和均值离群值检测
        """
        # 计算每帧的均值
        frame_means = np.mean(data, axis=(1, 2))

        # 计算帧间差异
        frame_diff = np.abs(np.diff(frame_means))
        median_diff = np.median(frame_diff)
        mad_diff = 1.4826 * np.median(np.abs(frame_diff - median_diff))

        # 找出异常帧
        z_scores = np.abs((frame_diff - median_diff) / mad_diff)
        potential_bad = np.where(z_scores > threshold)[0]

        # 合并相邻坏帧
        bad_frames = []
        for i in potential_bad:
            if not bad_frames or i > bad_frames[-1] + 1:
                bad_frames.extend([i, i + 1])  # 标记差异大的前后两帧
            elif i == bad_frames[-1] + 1:
                bad_frames.append(i + 1)

        return sorted(list(set(bad_frames)))

    @pyqtSlot(object, list, int)
    def fix_bad_frames(self, data: DataManager.ProcessedData|DataManager.Data, bad_frames: List[int], n_frames: int = 2) -> np.ndarray:
        """
        修复坏帧 - 使用前后n帧的平均值替换
        """
        aim_data = data.data_origin
        fixed_data = aim_data.copy()
        total_frames = len(aim_data)

        for frame_idx in bad_frames:
            # 计算前后n帧的范围
            start = max(0, frame_idx - n_frames)
            end = min(total_frames, frame_idx + n_frames + 1)

            # 排除坏帧本身
            valid_frames = [i for i in range(start, end)
                            if i != frame_idx and i not in bad_frames]

            if valid_frames:
                # 计算平均值
                fixed_data[frame_idx] = np.mean(aim_data[valid_frames], axis=0)
            else:
                print(f"警告: 无法修复帧 {frame_idx} - 无有效参考帧")

        data.update_data(**self.amend_data(fixed_data))

    @pyqtSlot(np.ndarray,str,object)
    def plot_data_prepare(self, data: np.ndarray,name:str, father_obj: DataManager.Data| DataManager.ProcessedData ):
        """完成plot数据的准备操作"""
        try:
            time_point = father_obj.time_point
            father_dict = father_obj.parameters if isinstance(father_obj, DataManager.Data) else father_obj.out_processed
            if data.ndim == 2:
                self.plot_singal.emit(data,{'name':name})
            else:
                if time_point is None or time_point.shape != data.shape:
                    if hasattr(father_dict, 'fps'):
                        self.plot_singal.emit(self.add_time_from_fps(data,father_dict['fps']),
                                              {'name':name, 'time_unit':'s'})
                    else:
                        self.plot_singal.emit(self.add_time_simple(data,father_dict['time_step']),
                                              {'name':name, 'time_unit':father_dict.get('time_unit',None)})
                else:
                    self.plot_singal.emit(np.column_stack((time_point,data)),
                                          {'name':name, 'time_unit':father_dict.get('time_unit',None)})
        except Exception as e:
            logging.error(f'数据无法被绘制由于：{e}')

    @staticmethod
    def add_time_from_fps(data: np.ndarray, sampling_rate: float,start_time: float = 0.0) -> np.ndarray:
        """基于fps给一维数组添加时间码"""
        if data.ndim != 1:
            raise ValueError("输入必须是一维数组")
        if sampling_rate <= 0:
            raise ValueError("采样频率必须大于0")

        n = len(data)
        # 计算采样间隔
        sampling_interval = 1.0 / sampling_rate
        # 生成时间戳
        timestamps = np.arange(n) * sampling_interval + start_time
        # 合并为二维数组
        result = np.column_stack((timestamps, data))

        return result

    @staticmethod
    def add_time_simple(data: np.ndarray, time_step:float, start_time: float = 0.0) -> np.ndarray:
        """基于间隔给一维数组添加时间"""
        if data.ndim != 1:
            raise ValueError("输入必须是一维数组")

        n = len(data)
        # 生成序号
        serial_numbers = np.arange(n) * time_step + start_time
        # 合并为二维数组
        result = np.column_stack((serial_numbers, data))

        return result


class MassDataProcessor(QObject):
    """大型数据（EM-iSCAT）处理的线程解决"""
    processing_progress_signal = pyqtSignal(int, int) # 进度槽
    processed_result = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        logging.info("大数据处理线程已载入")
        self.abortion = False

    @pyqtSlot(DataManager.Data,int,bool)
    def pre_process(self,data,bg_num = 360,unfold=True):
        """数据预处理，包含背景去除，数组展开"""
        try:
            logging.info("开始预处理...")
            self.processing_progress_signal.emit(0, 100)

            # 1. 提取前n帧计算背景帧
            if data.datatype != np.float32:
                data_origin = data.data_origin.astype(np.float32) # 注意这里开始原本Uint8 转为了F32
            else:
                data_origin = data.data_origin
            total_frames = data.timelength

            # 计算背景帧 (前n帧的平均) 并添加遇0处理
            self.processing_progress_signal.emit(10, 100)
            bg_frame = np.median(data_origin[:bg_num], axis=0)
            bg_frame_safe = bg_frame.copy()
            epsilon = 1e-10
            zero_mask = np.abs(bg_frame_safe) < epsilon
            if np.any(zero_mask):
                # 用非零最小值替换零值
                non_zero_min = np.min(np.abs(bg_frame_safe[~zero_mask]))
                if non_zero_min > 0:
                    bg_frame_safe[zero_mask] = non_zero_min
                else:
                    bg_frame_safe[zero_mask] = 1.0
            self.processing_progress_signal.emit(20, 100)

            # 2. 所有帧减去背景

            # processed_data = (data_origin - bg_frame[np.newaxis, :, :]) / bg_frame[np.newaxis, :, :]
            # self.processing_progress_signal.emit(50, 100)
            processed_data = np.empty_like(data_origin)
            for i in range(total_frames):
                if self.abortion:
                    return None

                # 减去背景帧
                processed_data[i] = (data_origin[i] - bg_frame_safe ) / bg_frame_safe

                # 每隔10%的进度更新一次
                if i % max(1, total_frames // 10) == 0:
                    progress_value = 20 + int(60 * i / total_frames)
                    self.processing_progress_signal.emit(progress_value, 100)
            self.processing_progress_signal.emit(80, 100)

            # 3. 展开为二维数组
            if unfold:
                T, H, W = processed_data.shape
                unfolded_data = processed_data.reshape((T, H * W)).T
            else:
                unfolded_data = None
            self.processing_progress_signal.emit(95, 100)
            # 保存结果
            processed = DataManager.ProcessedData(data.timestamp,
                                                  f'{data.name}@EM_pre',
                                                  'EM_pre_processed',
                                                  time_point=data.time_point,
                                                  data_processed=processed_data,
                                                  out_processed={
                                                      'fps' : data.parameters['fps'],
                                                      'bg_frame': bg_frame,
                                                      'unfolded_data': unfolded_data,**data.parameters
                                                  })

            self.processed_result.emit(processed)
            self.processing_progress_signal.emit(100, 100)
            return True
        except Exception as e:
            self.processed_result.emit({'type': "EM_pre_processed", 'error': str(e)})
            return False

    @pyqtSlot(DataManager.ProcessedData,float,int,int,int,int,int,str)
    def quality_stft(self,data,target_freq: float,scale_range:int,fps:int, window_size: int, noverlap: int,
                    custom_nfft: int, window_type: str):
        """STFT质量分析"""
        if not data:
                raise ValueError("请先进行预处理")
        try:
            unfolded_data = data.out_processed['unfolded_data']  # [像素数 x 帧数]

            # 窗函数的选择和生成
            window = self.get_window(window_type, window_size)

            mean_signal = np.mean(unfolded_data, axis=0)
            f, t, Zxx = signal.stft(
                mean_signal,
                fs=fps,
                window=window,
                nperseg=window_size,
                noverlap=noverlap,
                nfft=custom_nfft,
                return_onesided=True,
                scaling='psd'
            )
            # 提取范围内所有对应的索引
            if scale_range > 0:
                low_bound = max(0.0, target_freq - scale_range / 2.0)
                high_bound = min(f[-1], target_freq + scale_range / 2.0)
                target_idx = np.where((f >= low_bound) & (f <= high_bound))[0]
                if len(target_idx) == 0:
                    target_idx = [np.argmin(np.abs(f - target_freq))]
            else:
                target_idx = [np.argmin(np.abs(f - target_freq))]
            self.processed_result.emit(DataManager.ProcessedData(data.timestamp,
                                                                 f'{data.name}@stft_q',
                                                                 'stft_quality',
                                                                 data_processed=Zxx,
                                                                 out_processed={
                                                                     'window_type': window,
                                                                     'window_size': window_size,
                                                                     'out_length': Zxx.shape[1],
                                                                     'frequencies':f,
                                                                     'time_series':t,
                                                                     'target_freq':target_freq,
                                                                     'scale_range':scale_range,
                                                                     'target_idx':target_idx,
                                                                 })
                                       )
            return True
        except Exception as e:
            self.processed_result.emit({'type': "ROI_stft", 'error': str(e)})
            return False

    # @pyqtSlot(DataManager.ProcessedData,object,int,int,int,int,int,int,str)
    # def python_stft(self,data, target_freq,scale_range:int,fps:int, window_size: int, noverlap: int,
    #                 custom_nfft: int, batch_size: int, window_type: str):
    #     """
    #     执行逐像素STFT分析
    #     参数:
    #         avi_data: 预处理后的数据字典
    #         target_freq: 目标分析频率(Hz)
    #         window_size: Hanning窗口大小(样本数)
    #         noverlap: 重叠样本数
    #         custom_nfft: 自定义FFT点数(可选)
    #     """
        # try:
        #     if not data:
        #         raise ValueError("请先进行预处理")
        #     if 'out_length' not in data.out_processed:
        #         raise ValueError("请先进行质量评估")
        #
        #     if isinstance(target_freq, float):
        #         target_idx = data.out_processed['target_idx']
        #     else:
        #         target_idx = target_freq
        #     out_length = data.out_processed['out_length']
        #     time_series = data.out_processed['time_series']
        #     freq = data.out_processed['frequencies']
        #     data = next(data for data in reversed(data.history) if data.type_processed == "EM_pre_processed")# [像素数 x 帧数]
        #     frame_size = data.framesize  # (宽度, 高度)
        #     unfolded_data = data.out_processed['unfolded_data']
        #
        #     # 2. 计算采样率和FFT参数
        #     total_frames = unfolded_data.shape[1]
        #     total_pixels = unfolded_data.shape[0]
        #     nfft = custom_nfft
        #     if nfft < window_size: # 确保nfft大于等于窗长度
        #         nfft = window_size
        #
        #     self.processing_progress_signal.emit(1, total_pixels)
        #
        #     # 4. 初始化结果数组
        #     height, width = frame_size
        #     stft_py_out = np.zeros((out_length, height, width), dtype=np.float32)
        #
        #     # 窗函数的选择和生成
        #     window = self.get_window(window_type, window_size)
        #
        #     # 5. 逐像素STFT处理
        #     self.processing_progress_signal.emit(1, total_pixels)
        #
        #     # 对每个像素执行STFT
        #     for i in range(total_pixels):
        #         if self.abortion:
        #             return
        #
        #         pixel_signal = unfolded_data[i, :]
        #
        #         # 计算当前像素的STFT
        #         # signal.ShortTimeFFT
        #         _, _, Zxx = signal.stft(
        #             pixel_signal,
        #             fs=fps,
        #             window=window,
        #             nperseg=window_size,
        #             noverlap=noverlap,
        #             nfft=nfft,
        #             return_onesided=False,
        #             scaling='psd'
        #         )
        #         # 提取目标频率处的幅度
        #         magnitude = np.mean(np.abs(Zxx[target_idx, :]), axis=0) * 560
        #
        #         # 将结果存入对应像素位置
        #         y = i // width
        #         x = i % width
        #         stft_py_out[:, y, x] = magnitude
        #         # zxx_out[:,:, y, x] = Zxx
        #
        #         self.processing_progress_signal.emit(i, total_pixels)
        #
        #     # with h5py.File('transfer_data.h5', 'w') as f:
        #     #     # 创建数据集并写入数据(for debug)
        #     #     dset = f.create_dataset('big_array', data=stft_py_out, compression='gzip')
        #
        #     # 6. 发送完整结果
        #     self.processed_result.emit(DataManager.ProcessedData(data.timestamp,
        #                                                        f'{data.name}@r_stft',
        #                                                        'ROI_stft',
        #                                                         time_point=time_series,
        #                                                        data_processed=stft_py_out,
        #                                                        out_processed={'whole_mean':np.mean(stft_py_out, axis=(1, 2)),
        #                                                                       'window_type': window,
        #                                                                       'window_size': window_size,
        #                                                                       'window_step': window_size - noverlap,
        #                                                                       'FFT_length': nfft,
        #                                                                       **{k:data.out_processed.get(k)
        #                                                                          for k in data.out_processed if k not in {"unfolded_data"}}}))
        #     self.processing_progress_signal.emit(total_pixels, total_pixels)
        #     return True
        # except Exception as e:
        #     self.processed_result.emit({'type': "ROI_stft", 'error': str(e)})
        #     return False


    # @pyqtSlot(DataManager.ProcessedData, object, int, int, int, int, int, int, str)
    # def python_stft(self, data, target_freq, scale_range: int, fps: int, window_size: int, noverlap: int,
    #                 custom_nfft: int, batch_size: int, window_type: str):
    #     """分块处理的stft"""
    #     # try:
    #     window = self.get_window(window_type, window_size)
    #     frame_size = data.framesize
    #     unfolded_data = data.out_processed['unfolded_data']  # Shape: [Pixels, Frames]
    #
    #     mean_signal = np.mean(unfolded_data, axis=0)
    #     f0, t0, Zxx0 = signal.stft(
    #         mean_signal,
    #         fs=fps,
    #         window=window,
    #         nperseg=window_size,
    #         noverlap=noverlap,
    #         nfft=custom_nfft,
    #         return_onesided=True,
    #         scaling='psd'
    #     )
    #     out_length = Zxx0.shape[1]
    #     time_series = t0
    #
    #     if isinstance(target_freq, float):
    #         if scale_range > 0:
    #             low_bound = max(0.0, target_freq - scale_range / 2.0)
    #             high_bound = min(f0[-1], target_freq + scale_range / 2.0)
    #             target_idx = np.where((f0 >= low_bound) & (f0 <= high_bound))[0]
    #             if len(target_idx) == 0:
    #                 target_idx = [np.argmin(np.abs(f0 - target_freq))]
    #         else:
    #             target_idx = [np.argmin(np.abs(f0 - target_freq))]
    #     else:
    #         target_idx = target_freq
    #
    #     total_pixels = unfolded_data.shape[0]
    #     nfft = max(custom_nfft, window_size)
    #     height, width = frame_size
    #
    #     # --- 2. 准备工作 ---
    #     self.processing_progress_signal.emit(1, total_pixels)
    #
    #     # 创建结果数组 (out_length, height, width)
    #     stft_py_out = np.zeros((out_length, height, width), dtype=np.float32)
    #
    #     # 创建一个临时视图，将空间维度展平，方便按像素索引赋值
    #     # Shape: (out_length, total_pixels)
    #     stft_out_flat = stft_py_out.reshape(out_length, -1)
    #
    #     for start_idx in range(0, total_pixels, batch_size):
    #         if self.abortion:
    #             return
    #
    #         end_idx = min(start_idx + batch_size, total_pixels)
    #
    #         # 取出一批像素数据 (Batch_size, Frames)
    #         batch_data = unfolded_data[start_idx:end_idx, :]
    #
    #         # [关键点]：一次性计算整个 Batch 的 STFT
    #         # axis=-1 表示对最后的时间维度做变换
    #         # 返回 Zxx shape: (Batch_size, Freqs, Time_segments)
    #         _, _, Zxx_batch = signal.stft(
    #             batch_data,
    #             fs=fps,
    #             window=window,
    #             nperseg=window_size,
    #             noverlap=noverlap,
    #             nfft=nfft,
    #             return_onesided=False,
    #             scaling='psd',
    #             axis=-1
    #         )
    #
    #         # 现在的 Zxx_batch 是 3D 的。我们需要切片并保持 batch 维度。
    #
    #         # 提取目标频率层。
    #         # 注意：target_idx 可能是一个切片(slice)或列表，也可能是一个整数。
    #         target_data = Zxx_batch[:, target_idx, :]  # Shape: (Batch_size, [Freqs], Time)
    #
    #         # 计算模值
    #         mag_batch = np.abs(target_data)
    #
    #         # 如果 target_idx 包含了多个频率(是一个频带)，需要像原来一样 dim=0 (现在是 dim=1) 做平均
    #         # 如果 target_idx 只是单个频率，ndim 已经是 2 (Batch, Time) 了
    #         if mag_batch.ndim == 3:
    #             mag_batch = np.mean(mag_batch, axis=1)
    #
    #         mag_batch *= 560
    #
    #         # 此时 mag_batch shape 为 (Batch_size, Time_segments)
    #         # 我们的目标 stft_out_flat 需要 (Time_segments, Batch_size)
    #         # 所以需要转置 (.T)
    #         # 赋值给结果数组
    #         # 注意：这里可能会因为 signal.stft 返回的 time segment 数量与 out_length 有极其微小的差异（取决于padding），
    #         # 建议加个切片 [:out_length] 保护
    #         valid_time_len = min(mag_batch.shape[1], out_length)
    #         stft_out_flat[:valid_time_len, start_idx:end_idx] = mag_batch.T[:valid_time_len, :]
    #
    #         # 每处理完一个 Batch 发送一次信号，而不是每个像素发一次
    #         self.processing_progress_signal.emit(end_idx, total_pixels)
    #
    #     # --- 5. 收尾工作 ---
    #     # 此时 stft_py_out 已经被通过 stft_out_flat 视图填充完毕
    #
    #     result_payload = DataManager.ProcessedData(
    #         data.timestamp,
    #         f'{data.name}@r_stft',
    #         'ROI_stft',
    #         time_point=time_series,
    #         data_processed=stft_py_out,
    #         out_processed={
    #             'whole_mean': np.mean(stft_py_out, axis=(1, 2)),
    #             'window_type': window,
    #             'window_size': window_size,
    #             'window_step': window_size - noverlap,
    #             'target_freq': target_freq,
    #             'FFT_length': nfft,
    #             'batch_size': batch_size,
    #             **{k: data.out_processed.get(k) for k in data.out_processed if k not in {"unfolded_data"}}
    #         }
    #     )
    #
    #     self.processed_result.emit(result_payload)
    #     self.processing_progress_signal.emit(total_pixels, total_pixels)
    #     return True
        # # except Exception as e:
        # #     self.processed_result.emit({'type': "ROI_stft", 'error': str(e)})
        # #     return False

    @pyqtSlot(DataManager.ProcessedData, object, int, int, int, int, int, str, bool, int, int)
    def python_stft(self, data, target_freq, scale_range: int, fps: int, window_size: int, noverlap: int,
                    custom_nfft: int, window_type: str, is_multipro: bool, batch_size: int, cpu_num: int):
        """可选并行计算加速
        执行逐像素STFT分析
        参数:
            avi_data: 预处理后的数据字典
            target_freq: 目标分析频率(Hz)
            window_size: Hanning窗口大小(样本数)
            noverlap: 重叠样本数
            custom_nfft: 自定义FFT点数(可选)
        """
        if is_multipro: # 启用并行加速
            logging.info("将开启并行加速，进度条不代表实时进度，请稍等。。。")
            shm_in = None
            shm_out = None
            pool = None

            try:
                # --- 1. 参数校验与准备 ---
                window = self.get_window(window_type, window_size)
                frame_size = data.framesize
                unfolded_data = data.out_processed['unfolded_data']  # Shape: [Pixels, Frames]

                mean_signal = np.mean(unfolded_data, axis=0)
                f0, t0, Zxx0 = signal.stft(
                    mean_signal,
                    fs=fps,
                    window=window,
                    nperseg=window_size,
                    noverlap=noverlap,
                    nfft=custom_nfft,
                    return_onesided=True,
                    scaling='psd'
                )
                out_length = Zxx0.shape[1]
                time_series = t0

                if isinstance(target_freq, float):
                    if scale_range > 0:
                        low_bound = max(0.0, target_freq - scale_range / 2.0)
                        high_bound = min(f0[-1], target_freq + scale_range / 2.0)
                        target_idx = np.where((f0 >= low_bound) & (f0 <= high_bound))[0]
                        if len(target_idx) == 0:
                            target_idx = [np.argmin(np.abs(f0 - target_freq))]
                    else:
                        target_idx = [np.argmin(np.abs(f0 - target_freq))]
                else:
                    target_idx = target_freq

                height, width = frame_size
                total_pixels = unfolded_data.shape[0]

                nfft = max(custom_nfft, window_size)

                # --- 2. 创建共享内存 (Shared Memory) ---
                # 这一步非常关键：我们在主进程申请内存，子进程直接用，无需复制 16GB 数据

                # A. 输入数据共享内存
                # create=True 表示创建新内存块
                shm_in = shared_memory.SharedMemory(create=True, size=unfolded_data.nbytes)
                # 将 numpy 数据复制进共享内存
                shm_in_arr = np.ndarray(unfolded_data.shape, dtype=unfolded_data.dtype, buffer=shm_in.buf)
                shm_in_arr[:] = unfolded_data[:]  # copy data

                # B. 输出结果共享内存 (预分配)
                # 我们需要一个 (out_length, total_pixels) 的矩阵方便写入
                # 最终结果是 (out_length, height, width)，但计算时平铺比较好处理
                output_shape_flat = (out_length, total_pixels)
                output_dtype = np.float32
                output_bytes = int(np.prod(output_shape_flat) * np.dtype(output_dtype).itemsize)

                shm_out = shared_memory.SharedMemory(create=True, size=output_bytes)
                # 初始化为 0
                shm_out_arr = np.ndarray(output_shape_flat, dtype=output_dtype, buffer=shm_out.buf)
                shm_out_arr[:] = 0

                # 获取核心数
                num_cores = cpu_num

                # 计算每个进程负责的像素范围
                chunk_size = math.ceil(total_pixels / num_cores)
                tasks = []

                # 使用 Manager Queue 进行进程间通信 (进度条)
                m = Manager()
                queue = m.Queue()

                for i in range(num_cores):
                    start = i * chunk_size
                    end = min((i + 1) * chunk_size, total_pixels)
                    if start >= end: break

                    # 打包参数
                    task_args = (
                        shm_in.name, unfolded_data.shape, unfolded_data.dtype,
                        shm_out.name, output_shape_flat, output_dtype,
                        (start, end),  # Pixel Range
                        fps, window, window_size, noverlap, nfft, target_idx,batch_size,
                        queue
                    )
                    tasks.append(task_args)

                self.processing_progress_signal.emit(0, total_pixels)
                print(f"开始多进程计算: 核心数={len(tasks)}, SHM_IN={shm_in.name}, SHM_OUT={shm_out.name}")

                # --- 4. 启动进程池 ---
                # 注意：Windows 下必须使用 starmap_async 或者手动解析 args
                pool = Pool(processes=len(tasks))

                # 异步提交任务
                result_async = pool.starmap_async(_stft_worker_process, tasks)

                # --- 5. 监控进度 ---
                # 主线程在这里循环，直到任务完成
                # 这样既不会阻塞 UI (如果在 thread 里)，又能实时更新进度
                processed_total = 0
                while not result_async.ready():
                    # 检查中止信号
                    if self.abortion:
                        pool.terminate()
                        return False

                    # 尝试从队列获取进度更新
                    # 每次取一点，避免死锁
                    while not queue.empty():
                        processed_total += queue.get()

                    self.processing_progress_signal.emit(processed_total, total_pixels)

                    # 稍微 sleep 一下避免空转烧 CPU
                    QThread.msleep(50)

                    # 确保最后取完队列
                while not queue.empty():
                    processed_total += queue.get()
                self.processing_progress_signal.emit(total_pixels, total_pixels)

                # 检查是否有异常抛出
                result_async.get()  # 如果 worker 报错，这里会 raise Exception

                # --- 6. 结果回收与重构 ---
                # 此时 shm_out_arr 里已经是计算好的数据了
                # 我们需要把它 reshape 成 (out_length, height, width)
                # 注意：这里的 .copy() 很重要，因为 shm_out 马上要 close/unlink 了
                final_result_flat = shm_out_arr.copy()

                # 重塑维度
                stft_py_out = final_result_flat.reshape(out_length, height, width)

                # 发送结果
                self.processed_result.emit(DataManager.ProcessedData(
                    data.timestamp,
                    f'{data.name}@r_stft',
                    'ROI_stft',
                    time_point=time_series,
                    data_processed=stft_py_out,
                    out_processed={
                        'whole_mean': np.mean(stft_py_out, axis=(1, 2)),
                        'window_type': window,
                        'window_size': window_size,
                        'window_step': window_size - noverlap,
                        'target_freq': target_freq,
                        'FFT_length': nfft,
                        **{k: data.out_processed.get(k) for k in data.out_processed if k not in {"unfolded_data"}}
                    }
                ))

                return True

            except Exception as e:
                traceback.print_exc()
                self.processed_result.emit({'type': "ROI_stft", 'error': str(e)})
                return False

            finally:
                # --- 7. 清理资源 (至关重要) ---
                if pool:
                    pool.close()
                    pool.join()

                # 必须手动释放共享内存，否则会造成内存泄漏直到重启电脑
                if shm_in:
                    shm_in.close()
                    shm_in.unlink()  # unlink 表示从系统中删除
                if shm_out:
                    shm_out.close()
                    shm_out.unlink()

                logging.info("共享内存已释放，进程池已关闭。")

        else: # 不启用加速

            try:
                window = self.get_window(window_type, window_size)
                frame_size = data.framesize
                unfolded_data = data.out_processed['unfolded_data']  # Shape: [Pixels, Frames]

                mean_signal = np.mean(unfolded_data, axis=0)
                f0, t0, Zxx0 = signal.stft(
                    mean_signal,
                    fs=fps,
                    window=window,
                    nperseg=window_size,
                    noverlap=noverlap,
                    nfft=custom_nfft,
                    return_onesided=True,
                    scaling='psd'
                )
                out_length = Zxx0.shape[1]
                time_series = t0

                if isinstance(target_freq, float):
                    if scale_range > 0:
                        low_bound = max(0.0, target_freq - scale_range / 2.0)
                        high_bound = min(f0[-1], target_freq + scale_range / 2.0)
                        target_idx = np.where((f0 >= low_bound) & (f0 <= high_bound))[0]
                        if len(target_idx) == 0:
                            target_idx = [np.argmin(np.abs(f0 - target_freq))]
                    else:
                        target_idx = [np.argmin(np.abs(f0 - target_freq))]
                else:
                    target_idx = target_freq

                total_pixels = unfolded_data.shape[0]
                nfft = max(custom_nfft, window_size)
                self.processing_progress_signal.emit(1, total_pixels)

                # 4. 初始化结果数组
                height, width = frame_size
                stft_py_out = np.zeros((out_length, height, width), dtype=np.float32)

                # 窗函数的选择和生成
                window = self.get_window(window_type, window_size)

                # 5. 逐像素STFT处理
                self.processing_progress_signal.emit(1, total_pixels)

                # 对每个像素执行STFT
                for i in range(total_pixels):
                    if self.abortion:
                        return

                    pixel_signal = unfolded_data[i, :]

                    # 计算当前像素的STFT
                    # signal.ShortTimeFFT
                    _, _, Zxx = signal.stft(
                        pixel_signal,
                        fs=fps,
                        window=window,
                        nperseg=window_size,
                        noverlap=noverlap,
                        nfft=nfft,
                        return_onesided=False,
                        scaling='psd'
                    )
                    # 提取目标频率处的幅度
                    magnitude = np.mean(np.abs(Zxx[target_idx, :]), axis=0) * 560

                    # 将结果存入对应像素位置
                    y = i // width
                    x = i % width
                    stft_py_out[:, y, x] = magnitude
                    # zxx_out[:,:, y, x] = Zxx

                    self.processing_progress_signal.emit(i, total_pixels)

                # with h5py.File('transfer_data.h5', 'w') as f:
                #     # 创建数据集并写入数据(for debug)
                #     dset = f.create_dataset('big_array', data=stft_py_out, compression='gzip')

                # 6. 发送完整结果
                self.processed_result.emit(DataManager.ProcessedData(data.timestamp,
                                                                   f'{data.name}@r_stft',
                                                                   'ROI_stft',
                                                                    time_point=time_series,
                                                                   data_processed=stft_py_out,
                                                                   out_processed={'whole_mean':np.mean(stft_py_out, axis=(1, 2)),
                                                                                  'window_type': window,
                                                                                  'window_size': window_size,
                                                                                  'window_step': window_size - noverlap,
                                                                                  'FFT_length': nfft,
                                                                                  **{k:data.out_processed.get(k)
                                                                                     for k in data.out_processed if k not in {"unfolded_data"}}}))
                return True
            except Exception as e:
                self.processed_result.emit({'type': "ROI_stft", 'error': str(e)})
                return False

    def get_window(self,window_type, window_size):
        try:
            if window_type == 'gaussian':
                window = signal.get_window((window_type, window_size / 6), window_size, fftbins=False)
            elif window_type == 'general_gaussian':
                window = signal.get_window((window_type, 1.5, window_size / 6), window_size, fftbins=False)
            else:
                window = signal.get_window(window_type, window_size, fftbins=False)
            # # 计算窗口能量
            # win_energy = np.sum(window ** 2)
            #
            # # 对窗口进行能量归一化
            # normalized_window = window / np.sqrt(win_energy)

            return window
        except Exception as e:
            logging.error(f'Window Fault:{e}')

    @pyqtSlot(DataManager.ProcessedData,float,int,int,int,str)
    def quality_cwt(self,data, target_freq: float,scale_range:int, fps: int, totalscales: int, wavelet: str = 'morl'):
        """
        CWT(连续小波变换)分析信号评估
        参数:
            target_freq: 目标分析频率(Hz)
            EM_fps: 采样频率
            scales: 尺度数组，控制小波变换的频率分辨率
            wavelet: 使用的小波类型(默认为'morl'墨西哥帽小波)
        """
        try:
            if not data:
                raise ValueError("请先进行预处理")

            unfolded_data = data.out_processed['unfolded_data']  # [像素数 x 帧数]
            frame_size = data.framesize  # (宽度, 高度)
            cparam = 2 * pywt.central_frequency(wavelet) * totalscales
            scales = cparam/np.arange(totalscales,1,-1)
            # target_freqs = np.linspace(int(target_freq-5), int(target_freq+5), totalscales//4)
            # scales = pywt.frequency2scale(wavelet, target_freqs * 1.0 / EM_fps)
            self.processing_progress_signal.emit(20, 100)
            # 计算参数
            total_frames = unfolded_data.shape[1]
            total_pixels = unfolded_data.shape[0]
            height, width = frame_size
            self.processing_progress_signal.emit(40, 100)
            # 计算平均信号的CWT (用于质量评估)
            mean_signal = np.mean(unfolded_data, axis=0)
            coefficients, frequencies = pywt.cwt(mean_signal, scales, wavelet, sampling_period=1.0 / fps)
            self.processing_progress_signal.emit(70, 100)
            # 发送平均信号CWT结果
            self.processed_result.emit(DataManager.ProcessedData(data.timestamp,
                                                                 f'{data.name}@cwt_q',
                                                                 'cwt_quality',
                                                                 data_processed=np.abs(coefficients),
                                                                 out_processed={
                                                                     'frequencies' : frequencies,
                                                                     'time_series' : np.arange(total_frames) / fps,
                                                                     'target_freq' : target_freq,
                                                                     'scale_range' : scale_range,
                                                                     'total_scales' : totalscales,
                                                                     'wavelet_name' : wavelet,
                                                                 }))
            self.processing_progress_signal.emit(100, 100)
            return True
        except Exception as e:
            self.processed_result.emit({'type': "cwt_quality", 'error': str(e)})
            return False

    @pyqtSlot(DataManager.ProcessedData,float, int, int,str, float)
    def python_cwt(self,data, target_freq: float, fps: int, totalscales: int, wavelet: str, cwt_scale_range: float):
        """
        执行逐像素CWT分析
        参数:
        参数:
            target_freq: 目标分析频率(Hz)
            EM_fps: 采样频率
            scales: 尺度数组，控制小波变换的频率分辨率
            wavelet: 使用的小波类型(默认为cmor3-3)
        """
        try:
            if not data:
                raise ValueError("请先进行预处理")

            data = next(
                data for data in reversed(data.history) if data.type_processed == "EM_pre_processed")  # [像素数 x 帧数]
            unfolded_data = data.out_processed['unfolded_data']  # [像素数 x 帧数]
            frame_size = data.framesize  # (宽度, 高度)
            # cparam = 2 * pywt.central_frequency(wavelet) * totalscales
            # scales = cparam / np.arange(totalscales, 1, -1)
            target_freqs = np.linspace(target_freq-cwt_scale_range//2, target_freq+cwt_scale_range//2, totalscales)#totalscales//4
            scales = pywt.frequency2scale(wavelet, target_freqs * 1.0 / fps)
            total_frames = unfolded_data.shape[1]
            total_pixels = unfolded_data.shape[0]
            self.processing_progress_signal.emit(1, total_pixels)
            # 初始化结果数组
            height, width = frame_size
            cwt_py_out = np.zeros((data.timelength, height, width), dtype=np.float32)

            # 5. 逐像素STFT处理
            self.processing_progress_signal.emit(2, total_pixels)

            mid_idx = totalscales // 8

            # 对每个像素执行cwt
            for i in range(total_pixels):
                if self.abortion:
                    return

                pixel_signal = unfolded_data[i, :]

                # 计算当前像素的STFT
                coefficients, _ = pywt.cwt(pixel_signal, scales, wavelet, sampling_period=1.0 / fps)

                # 提取目标频率处（中间32个值）的幅度
                # aim_coefficients = coefficients[mid_idx-16:mid_idx+16,:]
                # magnitude_avg = np.mean(np.abs(aim_coefficients),axis = 0)
                magnitude_avg = np.mean(np.abs(coefficients),axis = 0)

                # 将结果存入对应像素位置
                y = i // width
                x = i % width
                cwt_py_out[:, y, x] = magnitude_avg * 30

                # 每100个像素更新一次进度
                self.processing_progress_signal.emit(i, total_pixels)

            # 发送完整结果
            times = np.arange(cwt_py_out.shape[0]) / fps
            self.processed_result.emit(DataManager.ProcessedData(data.timestamp,
                                                                 f'{data.name}@cwt',
                                                                 'ROI_cwt',
                                                                 time_point=times,
                                                                 data_processed=cwt_py_out,
                                                                 out_processed={'whole_mean':np.mean(cwt_py_out, axis=(1, 2)),
                                                                                'total_scales': totalscales,
                                                                                'wavelet_name':wavelet,
                                                                                'scale_range': cwt_scale_range,
                                                                                **{k: data.out_processed.get(k)
                                                                                   for k in data.out_processed if
                                                                                   k not in {"unfolded_data"}}}))
            self.processing_progress_signal.emit(total_pixels, total_pixels)
            return True
        except Exception as e:
            self.processed_result.emit({'type':"ROI_cwt",'error':str(e)})
            return False

    @pyqtSlot(object)
    def accumulate_amplitude(self,data):
        """累计时间振幅图"""
        self.processed_result.emit(DataManager.ProcessedData(data.timestamp,
                                         f'{data.name}@atam',
                                         "Accumulated_time_amplitude_map",
                                         time_point=data.time_point,
                                         data_processed=np.mean(data.data_processed if isinstance(data,DataManager.ProcessedData) else data.data_origin, axis=0),
                                         out_processed={**data.out_processed}
                                                             ))
        logging.info("累计时间振幅计算已完成")

    @staticmethod
    def D2GaussFunction(xy, A, x0, sigmax, y0, sigmay, b):
        """二维高斯函数
        参数:
        coords: 网格坐标 (x, y)
        A: 振幅
        x0, y0: 中心位置
        sigma_x, sigma_y: X/Y方向标准差
        offset: 背景偏移量

        返回:
        二维高斯函数值
        """
        x, y = xy[:, 0], xy[:, 1]
        return A * np.exp(-((x - x0) ** 2 / (2 * sigmax ** 2) + (y - y0) ** 2 / (2 * sigmay ** 2))) + b

    @pyqtSlot(DataManager.ProcessedData,int,float,bool)
    def twoD_gaussian_fit(self,data:DataManager.ProcessedData,zm = 2,thr = 2.5,thr_known = False):
        """
        对三维时序数据逐帧进行二维高斯拟合

        参数:
        data: numpy.ndarray, 三维数组 (T, H, W)
        zm 插值系数
        返回:
        results: list of dict, 每帧的拟合参数
        """
        try:
            timer = QElapsedTimer()
            timer.start()

            T, H, W = data.datashape
            self.processing_progress_signal.emit(0, T)

            amplitudes = np.zeros(T)
            centers_x = np.zeros(T)
            centers_y = np.zeros(T)
            mean_signal = np.zeros(T)
            max_signal = np.zeros(T)

            for m in range(T):
                frame = data.data_processed[m]
                self.processing_progress_signal.emit(m, T)
                # 图像插值
                if zm >1:
                    Z = zoom(frame, zm, order=3)
                else:
                    Z = frame

                h_z, w_z = Z.shape
                X, Y = np.meshgrid(np.arange(w_z), np.arange(h_z))
                xy = np.column_stack((X.ravel(), Y.ravel()))
                mean_signal[m] = np.mean(Z)
                max_signal[m] = np.max(Z)
                if thr_known: # 如果知道阈值
                    # 检查是否有超过阈值的点
                    if np.any(Z > thr):
                        max_value = np.max(Z)
                        y0_g, x0_g = np.unravel_index(np.argmax(Z), Z.shape)

                        # 初始参数 [A, x0, sigmax, y0, sigmay, b]
                        x0 = [max_value, x0_g, 1.0, y0_g, 1.0, np.mean(Z)]

                        # 参数边界
                        lb = [0, 0, 0.1, 0, 0.1, 0]
                        ub = [100, w_z, (w_z/2)**2, h_z, (h_z/2)**2, max_value]

                        try:
                            # 二维高斯拟合
                            popt, _ = curve_fit(self.D2GaussFunction,
                                                xy,
                                                Z.ravel(),
                                                p0=x0,
                                                bounds=(lb, ub),
                                                maxfev=5000)

                            amplitudes[m] = popt[0]
                            centers_x[m] = popt[1]
                            centers_y[m] = popt[3]
                        except RuntimeError:
                            amplitudes[m] = np.mean(Z)
                            centers_x[m] = np.nan
                            centers_y[m] = np.nan
                    else:
                        amplitudes[m] = np.mean(Z)
                        centers_x[m] = np.nan
                        centers_y[m] = np.nan
                else:
                    max_value = np.max(Z)
                    y0_g, x0_g = np.unravel_index(np.argmax(Z), Z.shape)

                    # 初始参数 [A, x0, sigmax, y0, sigmay, b]
                    x0 = [max_value, x0_g, 1.0, y0_g, 1.0, np.mean(Z)]

                    # 参数边界
                    lb = [0, 0, 0.1, 0, 0.1, 0]
                    ub = [100, w_z, (w_z / 2) ** 2, h_z, (h_z / 2) ** 2, max_value]
                    mean_signal[m] = np.mean(Z)
                    try:
                        # 二维高斯拟合
                        popt, _ = curve_fit(self.D2GaussFunction,
                                            xy,
                                            Z.ravel(),
                                            p0=x0,
                                            bounds=(lb, ub),
                                            maxfev=5000)

                        amplitudes[m] = popt[0]
                        centers_x[m] = popt[1]
                        centers_y[m] = popt[3]
                    except RuntimeError:
                        amplitudes[m] = np.mean(Z)
                        centers_x[m] = np.nan
                        centers_y[m] = np.nan
            self.processing_progress_signal.emit(T, T)
            self.processed_result.emit(DataManager.ProcessedData(data.timestamp,
                                             f'{data.name}@scs',
                                             "Single_channel_signal",
                                             time_point=data.time_point,
                                             data_processed=copy.deepcopy(amplitudes),
                                             out_processed={
                                                 'thr_known': thr_known,
                                                 'thr': thr,
                                                 'mean_signal':mean_signal,
                                                 'amplitudes':amplitudes,
                                                 'max_signal':max_signal,**data.out_processed
                                             }))
            return True
        except Exception as e:
            self.processed_result.emit({'type':"Single_channel_signal",'error':str(e)})
            return False

    @pyqtSlot(DataManager.ProcessedData, int, float, bool)
    def simple_single_channel(self, data: DataManager.ProcessedData, zm=2, thr=2.5, thr_known=False):
        """简单的单通道信号处理办法"""
        try:
            timer = QElapsedTimer()
            timer.start()

            T, H, W = data.datashape
            self.processing_progress_signal.emit(0, T)

            amplitudes = np.zeros(T)
            mean_signal = np.zeros(T)
            max_signal = np.zeros(T)
            min_signal = np.zeros(T)

            for m in range(T):
                frame = data.data_processed[m]
                self.processing_progress_signal.emit(m, T)
                # 图像插值
                if zm > 1:
                    Z = zoom(frame, zm, order=3)
                else:
                    Z = frame

                h_z, w_z = Z.shape
                mean_signal[m] = np.mean(Z)
                max_signal[m] = np.max(Z)
                min_signal[m] = np.min(Z)
                if thr_known:  # 如果知道阈值
                    # 检查是否有超过阈值的点
                    if np.any(Z > thr):
                        amplitudes[m] = np.max(Z)
                    else:
                        amplitudes[m] = np.mean(Z)
                else:
                    amplitudes[m] = np.max(Z)
            self.processing_progress_signal.emit(T, T)
            self.processed_result.emit(DataManager.ProcessedData(data.timestamp,
                                                                 f'{data.name}@scs',
                                                                 "Single_channel_signal",
                                                                 time_point=data.time_point,
                                                                 data_processed=copy.deepcopy(amplitudes),
                                                                 out_processed={
                                                                     'thr_known': thr_known,
                                                                     'thr': thr,
                                                                     'mean_signal': mean_signal,
                                                                     'max_signal': max_signal,
                                                                     'min_signal': min_signal,
                                                                     **{k: data.out_processed.get(k)
                                                                        for k in {'roi_shape','whole_mean'} if k in data.out_processed}
                                                                 }))
            return True
        except Exception as e:
            self.processed_result.emit({'type': "简单 Single_channel_signal", 'error': str(e)})
            return False

    @pyqtSlot(object)
    def twoD_fourier_transform(self,data,):
        """
            对3D时序视频数据进行2D傅里叶变换

            参数:
            video_data: 形状为 (帧数, 高度, 宽度) 的numpy数组

            返回:
            magnitude_spectra: 傅里叶幅度谱数组
            magnitude_log: 幅度谱的log（默认主参数）
            phase_spectra: 傅里叶相位谱数组
            """
        if isinstance(data, DataManager.ProcessedData):
            data = data.data_processed
        else:
            data = data.data_origin
            pass
        try:
            if data.timelength == 1:
                frames = 1
                height, width = data.datashape
            else:
                frames, height, width = data.datashape
            magnitude_spectra = np.zeros((frames, height, width))
            magnitude_log = np.zeros((frames, height, width))
            phase_spectra = np.zeros((frames, height, width))
            self.processing_progress_signal.emit(1,frames)
            for i in range(frames):
                # 2D傅里叶变换
                f = np.fft.fft2(data[i])
                fshift = np.fft.fftshift(f)  # 将低频移到中心

                # 幅度谱
                magnitude_spectra[i] = np.abs(fshift)

                # 相位谱
                phase_spectra[i] = np.angle(fshift)
                self.processing_progress_signal.emit(i, frames)

                magnitude_log[i] = np.log(magnitude_spectra[i]+1)

            self.processed_result.emit(DataManager.ProcessedData(data.timestamp,
                                       f'{data.name}@2DFT',
                                       "2D_Fourier_transform",
                                       time_point = data.time_point,
                                       data_processed=np.squeeze(magnitude_log),
                                       out_processed={'twoD_FFT':np.squeeze(fshift),
                                                      'magnitude_spectra': np.squeeze(magnitude_spectra),
                                                      'phase_spectra': np.squeeze(phase_spectra),**data.out_processed}))
            self.processing_progress_signal.emit(frames +1, frames)
            return True
        except Exception as e:
            self.processed_result.emit({'type': "2D_Fourier_transform", 'error': str(e)})
            return False

    def stop(self):
        """请求中止处理"""
        self.abort = True

    @staticmethod
    def calculate_amp_dur(data, thr, mode='open'):
        """
        计算单峰事件的振幅和持续时间

        参数:
        data: 一维时序数据
        thr: 阈值
        mode: 'open' 或 'close' (默认'open')

        返回:
        amplitudes: 事件振幅列表
        durations: 事件持续时间列表
        """
        amplitudes = []
        durations = []
        i = 0
        n = len(data)

        while i < n:
            if (mode == 'open' and data[i] > thr) or (mode == 'close' and data[i] < thr):
                start = i
                # 寻找事件结束点
                while i < n and ((mode == 'open' and data[i] > thr) or
                                 (mode == 'close' and data[i] < thr)):
                    i += 1
                end = i - 1

                # 计算事件振幅和持续时间
                event_data = data[start:end + 1]
                amplitude = np.mean(event_data)
                duration = (end - start + 1) * 0.65  # 假设采样间隔为0.65

                amplitudes.append(amplitude)
                durations.append(duration)
            else:
                i += 1

        return np.array(amplitudes), np.array(durations)

    # 上升/下降时间常数拟合(未启用)
    def fit_exponential_time(data, thr, mode='up', n=5):
        """
        拟合指数时间常数

        参数:
        data: 一维时序数据
        thr: 阈值
        mode: 'up' (上升) 或 'down' (下降)
        n: 用于拟合的点数 (默认5)

        返回:
        time_constants: 时间常数列表
        mse_values: 均方误差列表
        """
        time_constants = []
        mse_values = []
        x = np.arange(0, n * 0.65, 0.65)  # 时间轴

        for i in range(n, len(data) - n):
            if mode == 'up':
                # 上升沿检测: 当前点超过阈值，前一点低于阈值
                if data[i] > thr and data[i - 1] < thr:
                    # 取前n个点并反转
                    y = data[i - 1:i + n - 1][::-1]
                    # 调整基线
                    y = 2 * data[i - 1] - y
            else:  # mode == 'down'
                # 下降沿检测: 当前点低于阈值，前一点高于阈值
                if data[i] < thr and data[i - 1] > thr:
                    # 取后n个点并反转
                    y = data[i:i + n][::-1]

            # 指数拟合
            try:
                if mode == 'up' or mode == 'down':
                    # 初始参数估计
                    A0 = y[0] - y[-1]
                    tau0 = 1.0
                    b0 = y[-1]
                    p0 = [A0, tau0, b0]

                    # 指数函数模型
                    def exp_model(x, A, tau, b):
                        return A * np.exp(-x / tau) + b

                    # 拟合
                    popt, pcov = curve_fit(exp_model, x, y, p0=p0)

                    # 计算拟合质量
                    y_fit = exp_model(x, *popt)
                    mse = np.mean((y - y_fit) ** 2)

                    time_constants.append(popt[1])
                    mse_values.append(mse)
            except (RuntimeError, ValueError):
                # 拟合失败时跳过
                continue

        return np.array(time_constants), np.array(mse_values)

    # 4. 主分析流程
    # def analyze_single_peak_data(data, T, x_m, y_m):
    #     """
    #     完整分析流程
    #
    #     参数:
    #     data: 三维时序数据 (T, H, W)
    #     T: 时间轴
    #     x_m, y_m: ROI起始坐标
    #
    #     返回:
    #     所有分析结果
    #     """
    #     # 1. ROI高斯拟合
    #     amplitudes, centers_x, centers_y = fit_gaussian_roi(data, x_m, y_m)
    #
    #     # 绘制振幅变化
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(T, amplitudes)
    #     plt.xlabel('Time')
    #     plt.ylabel('Amplitude')
    #     plt.title('Amplitude over Time')
    #     plt.xlim([0, 15])
    #     plt.ylim([0, 10])
    #     plt.show()
    #
    #     # 2. 单峰振幅-持续时间计算
    #     open_amps, open_durs = calculate_amp_dur(amplitudes, thr=10, mode='open')
    #     close_amps, close_durs = calculate_amp_dur(amplitudes, thr=10, mode='close')
    #
    #     # 3. 时间常数拟合
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         tau_up, mse_up = fit_exponential_time(amplitudes, thr=10, mode='up', n=5)
    #         tau_down, mse_down = fit_exponential_time(amplitudes, thr=10, mode='down', n=5)
    #
    #     # 过滤低质量拟合
    #     tau_up = tau_up[mse_up < 0.1]
    #     tau_down = tau_down[mse_down < 0.1]
    #
    #     # 返回所有结果
    #     results = {
    #         'amplitudes': amplitudes,
    #         'centers_x': centers_x,
    #         'centers_y': centers_y,
    #         'open_amps': open_amps,
    #         'open_durs': open_durs,
    #         'close_amps': close_amps,
    #         'close_durs': close_durs,
    #         'tau_up': tau_up,
    #         'tau_down': tau_down
    #     }
    #
    #     return results
