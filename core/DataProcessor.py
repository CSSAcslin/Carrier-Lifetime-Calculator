import glob
import logging
import os
import re
import numpy as np
import tifffile as tiff
import sif_parser
import cv2
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QElapsedTimer
from skimage.exposure import equalize_adapthist
from typing import List, Union, Optional
from scipy import signal


class DataProcessor:
    """本类仅包含导入数据时的数据处理"""
    def __init__(self,path,normalize_type='linear',**kwargs):
        self.path = path
        self.normalize_type = normalize_type

    """tiff"""
    def load_and_sort_tiff(self, current_group):
        # 因为tiff存在两种格式，n,p
        files = []
        find = self.path + '/*.tiff'
        for f in glob.glob(find):
            match = re.search(r'(\d+)([a-zA-Z]+)\.tiff', f)
            if match and match.group(2) == current_group:
                files.append((int(match.group(1)), f))
        return sorted(files, key=lambda x: x[0])

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

    def process_tiff(self, files):
        '''初步数据处理'''
        images_original = []
        vmax_array = []
        vmin_array = []
        vmean_array = []
        for _, fpath in files:
            img_data = tiff.imread(fpath)
            vmax_array.append(np.max(img_data))
            vmin_array.append(np.min(img_data))
            vmean_array.append(np.mean(img_data))
            images_original.append(img_data)
        #   以最值为边界
        vmax = np.max(vmax_array)
        vmin = np.min(vmin_array)

        images_show, data_type, max_mean, phy_max, phy_min = self.process_data(images_original, vmax, vmin, vmean_array)

        return {
            'data_origin': np.stack(images_original, axis=0),
            'data_type': data_type,
            'images': np.stack(images_show, axis=0),
            'time_points': np.arange(len(images_show)),
            'data_mean': max_mean,
            'boundary': {'max':phy_max,'min':phy_min},
        }

    def amend_data(self, data, mask = None):
        """函数修改方法
        输入修改的源数据，导出修改的数据包"""
        if isinstance(data, dict): # 加roi来的
            data_origin = data['data_origin']
        elif isinstance(data, np.ndarray): # 坏点修复来的
            data_origin = data
        if mask is not None and mask.shape == data_origin[0].shape:
            data_mask = [ ]
            for every_data in data_origin:
                # data_mask.append(np.multiply(every_data, mask)) 目前这里有问题 还没想好怎么改
                every_data[~mask] = data['boundary']['min']
            data_origin = data_mask
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

            'images': np.stack(images_show, axis=0),

            'data_mean': max_mean,
            'boundary': {'max': phy_max, 'min': phy_min},
        }

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

    def fix_bad_frames(self, data: np.ndarray, bad_frames: List[int], n_frames: int = 2) -> np.ndarray:
        """
        修复坏帧 - 使用前后n帧的平均值替换
        """
        fixed_data = data.copy()
        total_frames = len(data)

        for frame_idx in bad_frames:
            # 计算前后n帧的范围
            start = max(0, frame_idx - n_frames)
            end = min(total_frames, frame_idx + n_frames + 1)

            # 排除坏帧本身
            valid_frames = [i for i in range(start, end)
                            if i != frame_idx and i not in bad_frames]

            if valid_frames:
                # 计算平均值
                fixed_data[frame_idx] = np.mean(data[valid_frames], axis=0)
            else:
                print(f"警告: 无法修复帧 {frame_idx} - 无有效参考帧")

        return fixed_data

    """sif"""
    def load_and_sort_sif(self):
        time_data = {}  # 存储时间点数据
        background = None  # 存储背景数据

        for filename in os.listdir(self.path):
            if filename.endswith('.sif'):
                filepath = os.path.join(self.path, filename)
                name = os.path.splitext(filename)[0]  # 去除扩展名

                # 检查是否是背景文件（文件名包含 "no"）
                if name.lower() == 'no':
                    background = sif_parser.np_open(filepath)[0][0]
                    continue

                # 否则尝试提取时间点（文件名中的数字）
                match = re.search(r'(\d+)', name)
                if match:
                    time = int(match.group(1))
                    data = sif_parser.np_open(filepath)[0][0]
                    time_data[time] = data
            else: return False

        # 检查是否找到背景
        if background is None:
            raise logging.error("未找到背景文件（文件名应包含 'no'）")

        # 按时间排序
        self.sif_sorted_times = sorted(time_data.keys())

        # 创建三维数组（时间, 高度, 宽度）并减去背景
        sample_data = next(iter(time_data.values()))
        self.sif_data_original = np.zeros((len(self.sif_sorted_times), *sample_data.shape), dtype=np.float32)

        for i, time in enumerate(self.sif_sorted_times):
            self.sif_data_original[i] = (time_data[time] - background)/background

        return True

    def process_sif(self):
        if not hasattr(self,'sif_data_original'):
            return logging.error('无有效数据')
        if not hasattr(self,'sif_sorted_times'):
            return logging.error('时间无效')
        min_val = np.min(self.sif_data_original)
        max_val = np.max(self.sif_data_original)

        normalized = self.normalize_data(self.sif_data_original,self.normalize_type)
        return {
            # 'signal':np.stack(), 不写了先
            'data_origin': np.stack(self.sif_data_original , axis=0),
            'data_type': 'sif',
            'images': np.stack(normalized, axis=0),
            'time_points': np.stack(self.sif_sorted_times,axis=0),
            'boundary': {'max': max_val, 'min': min_val},
        }


    def normalize_data(self,
            data: np.ndarray,
            method: str = 'linear',
            low: float = 10,
            high: float = 100,
            k: Optional[float] = None,
            clip_limit: float = 0.03,
            eps: float = 1e-6
    ) -> np.ndarray:
        """
        多种归一化方法可选
        Parameters:
            method:
                'linear'    - 线性归一化 (min-max)
                'sigmoid'  - Sigmoid归一化
                'percentile'- 百分位裁剪归一化 (默认)
                'log'      - 对数归一化
                'clahe'    - 自适应直方图均衡化
            low/high: 百分位裁剪的上下界（method='percentile'时生效）
            k: Sigmoid的斜率系数（method='sigmoid'时生效，None则自动计算）
            clip_limit: CLAHE的裁剪限制（method='clahe'时生效）
            eps: 对数归一化的微小增量（method='log'时生效）
        """
        if method == 'linear':
            # 线性归一化
            return (data - np.min(data)) / (np.max(data) - np.min(data))

        elif method == 'sigmoid':
            # Sigmoid归一化
            mu = np.median(data)
            std = np.std(data)
            k = 10 / std if k is None else k
            centered = data - mu
            return 1 / (1 + np.exp(-k * centered))

        elif method == 'percentile':
            # 百分位裁剪归一化
            plow = np.percentile(data, low)
            phigh = np.percentile(data, high)
            clipped = np.clip(data, plow, phigh)
            return (clipped - plow) / (phigh - plow)

        elif method == 'log':
            # 对数归一化
            logged = np.log(data + eps)
            return (logged - np.min(logged)) / (np.max(logged) - np.min(logged))

        elif method == 'clahe':
            # CLAHE自适应直方图均衡化
            return equalize_adapthist(data, clip_limit=clip_limit)



class MassDataProcessor(QObject):
    """大型数据（EM-iSCAT）处理的线程解决"""
    mass_finished = pyqtSignal(dict) # 数据读取
    processing_progress_signal = pyqtSignal(int, int) # 进度槽
    stft_completed = pyqtSignal(np.ndarray)  # 三维STFT结果数组
    avg_stft_result = pyqtSignal(np.ndarray, np.ndarray, np.ndarray,float)  # 平均信号STFT结果

    def __init__(self):
        super(MassDataProcessor,self).__init__()
        logging.info("大数据处理线程已载入")
        self.abortion = False

    """avi"""
    @pyqtSlot(str)
    def load_avi(self,path):
        """
        处理AVI视频文件，返回包含视频数据和元信息的字典
        返回字典:
            - data_origin: 原始视频帧数据 (n_frames, height, width)
            - images: 归一化后的视频帧数据
            - time_points: 时间点数组
            - data_type: 数据类型标识 ('video')
            - boundary: 最大最小值边界
            - fps: 视频帧率
            - frame_size: 视频帧尺寸 (width, height)
            - duration: 视频时长(秒)
        """

        # 读取视频文件
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件: {path}")

        # 获取视频元信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        # codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        duration = frame_count / fps if fps > 0 else 0

        # 读取所有帧
        frames = []
        loading_bar_value = 0  # 进度条
        total_l = frame_count+1
        while not self.abortion:
            ret, frame = cap.read()
            if not ret:
                break
            # 转换为灰度图(如果原始是彩色)
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
            loading_bar_value += 1
            self.processing_progress_signal.emit(loading_bar_value, total_l)

        cap.release()

        if not frames:
            raise ValueError("视频中没有读取到有效帧")

        # 转换为numpy数组
        frames_array = np.stack(frames, axis=0)

        # 计算统计信息
        vmax = np.max(frames_array)
        vmin = np.min(frames_array)

        # 归一化处理
        normalized_frames = self.normalize_data(frames_array)

        self.processing_progress_signal.emit(loading_bar_value+1, total_l)
        avi_data = {
            'data_origin': frames_array,
            'images': normalized_frames,
            'time_points': np.arange(len(frames)) / fps if fps > 0 else np.arange(len(frames)),
            'data_type': 'video',
            'boundary': {'max': vmax, 'min': vmin},
            'fps': fps,
            'frame_size': (width, height),
            'duration': duration,
            # 'codec': codec_str,
        }
        self.mass_finished.emit(avi_data)

    @pyqtSlot(str)
    def load_tiff(self,path):
        pass
    @pyqtSlot(dict,int,bool)
    def pre_process(self,data_dict:dict,bg_num = 360,unfold=True):
        """数据预处理，包含背景去除，数组展开"""
        try:
            logging.info("开始预处理...")
            self.processing_progress_signal.emit(0, 100)

            # 1. 提取前n帧计算背景帧
            data_origin = data_dict['data_origin']
            total_frames = data_origin.shape[0]

            # 计算背景帧 (前n帧的平均)
            self.processing_progress_signal.emit(10, 100)
            bg_frame = np.mean(data_origin[:bg_num], axis=0)
            self.processing_progress_signal.emit(20, 100)

            # 2. 所有帧减去背景
            processed_data = np.empty_like(data_origin)
            for i in range(total_frames):
                if self.abortion:
                    return None

                # 减去背景帧
                processed_data[i] = data_origin[i] - bg_frame

                # 每隔10%的进度更新一次
                if i % max(1, total_frames // 10) == 0:
                    progress_value = 20 + int(60 * i / total_frames)
                    self.processing_progress_signal.emit(progress_value, 100)

            # 3. 可选：展开为二维数组
            self.processing_progress_signal.emit(85, 100)
            if unfold:
                T, H, W = processed_data.shape
                unfolded_data = processed_data.reshape((T, H * W)).T
                data_dict['unfolded_data'] = unfolded_data

            self.processing_progress_signal.emit(95, 100)

            # 4. 更新结果字典
            data_dict['bg_frame'] = bg_frame
            data_dict['data_process'] = processed_data

            # 计算统计信息
            vmax = np.max(processed_data)
            vmin = np.min(processed_data)
            data_dict['boundary'] = {'max': vmax, 'min': vmin}

            self.processing_progress_signal.emit(100, 100)
            self.data = data_dict
            logging.info("预处理结束")

        except Exception as e:
            logging.error(f"预处理错误: {str(e)}")

    def python_stft(self, target_freq: float, window_size: int, noverlap: int,
                    custom_nfft: int):
        """
        执行逐像素STFT分析
        参数:
            avi_data: 预处理后的数据字典
            target_freq: 目标分析频率(Hz)
            window_size: Hanning窗口大小(样本数)
            noverlap: 重叠样本数
            custom_nfft: 自定义FFT点数(可选)
        """
        try:
            if not self.data:
                raise ValueError("清闲静心预处理")
            # 1. 检查必要数据存在
            if 'unfolded_data' not in self.data:
                raise ValueError("需要先进行unfold预处理")

            timer = QElapsedTimer()
            timer.start()

            unfolded_data = self.data['unfolded_data']  # [像素数 x 帧数]
            frame_size = self.data['frame_size']  # (宽度, 高度)
            fps = self.data['fps']  # 原始帧率

            # 2. 计算采样率和FFT参数
            total_frames = unfolded_data.shape[1]
            total_pixels = unfolded_data.shape[0]
            nfft = custom_nfft

            # 3. 计算平均信号的STFT (用于质量评估)
            mean_signal = np.mean(unfolded_data, axis=0)
            f, t, Zxx = signal.stft(
                mean_signal,
                fs=fps,
                window=signal.windows.hann(window_size),
                nperseg=window_size,
                noverlap=noverlap,
                nfft=nfft,
                return_onesided=True
            )
            # 发送平均信号STFT结果
            self.avg_stft_result.emit(f, t, np.abs(Zxx),target_freq)
            self.processing_progress_signal.emit(1, total_pixels)

            # 4. 初始化结果数组
            width, height = frame_size
            stft_py_out = np.zeros((Zxx.shape[1], height, width), dtype=np.float32)

            # 5. 逐像素STFT处理
            self.processing_progress_signal.emit(0, total_pixels)

            # 找到目标频率最近的索引
            target_idx = np.argmin(np.abs(f - target_freq))

            # 对每个像素执行STFT
            for i in range(total_pixels):
                if self.abortion:
                    return

                pixel_signal = unfolded_data[i, :]

                # 计算当前像素的STFT
                _, _, Zxx = signal.stft(
                    pixel_signal,
                    fs=fps,
                    window=signal.windows.hann(window_size),
                    nperseg=window_size,
                    noverlap=noverlap,
                    nfft=nfft,
                    return_onesided=False
                )

                # 提取目标频率处的幅度
                magnitude = np.abs(Zxx[target_idx, :])

                # 将结果存入对应像素位置
                y = i // width
                x = i % width
                stft_py_out[:, y, x] = magnitude

                # 每100个像素更新一次进度
                if i % 100 == 0:
                    self.processing_progress_signal.emit(i, total_pixels)

            # 6. 发送完整结果
            self.stft_completed.emit(stft_py_out)
            self.processing_progress_signal.emit(total_pixels, total_pixels)
            logging.info(f"计算完成，总耗时{timer.elapsed()}ms")

        except Exception as e:
            logging.error(f"STFT计算错误: {str(e)}")

    def normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def stop_thread(self):
        """请求中止处理"""
        self.abort = True