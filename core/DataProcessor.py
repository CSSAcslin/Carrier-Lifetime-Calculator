import glob
import logging
import os
import re
import numpy as np
import tifffile as tiff
import sif_parser
import cv2
import pywt
from PIL import Image
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QElapsedTimer
from skimage.exposure import equalize_adapthist
from typing import List, Union, Optional
from scipy import signal
from scipy.optimize import curve_fit
from scipy.ndimage import zoom
import DataManager


class DataProcessor:
    """本类仅包含导入数据时的数据处理"""
    def __init__(self,path,normalize_type='linear',**kwargs):
        self.path = path
        self.normalize_type = normalize_type
        self.tiff_type = 'np'

    """tiff"""
    def load_and_sort_tiff(self, current_group):
        # 因为tiff存在两种格式，n,p
        files = []
        find = self.path + '/*.tiff'
        if current_group != '不区分':
            self.tiff_type = current_group
            for f in glob.glob(find):
                match = re.search(r'(\d+)([a-zA-Z]+)\.tiff', f)
                if match and match.group(2) == current_group:
                    files.append((int(match.group(1)), f))
            if not files: # 当找不到任何tiff时，使其能够寻找tif结尾的文件
                find_tif = self.path + '/*.tif'
                for f in glob.glob(find_tif):
                    match = re.search(r'(\d+)([a-zA-Z]+)\.tif', f)
                    if match and match.group(2) == current_group:
                        files.append((int(match.group(1)), f))
        if current_group == '不区分':
            self.tiff_type = 'np'
            find_tif =  self.path + '/*.tiff'
            for f in glob.glob(find_tif):
                # 提取数字并排序
                num_groups = re.findall(r'\d+', f)
                last_num = int(num_groups[-1]) if num_groups else 0
                files.append((last_num, f))
            if not files:  # 当找不到任何tiff时，使其能够寻找tif结尾的文件
                find_tif = self.path + '/*.tif'
                for f in glob.glob(find_tif):
                    # 提取数字并排序
                    num_groups = re.findall(r'\d+', f)
                    last_num = int(num_groups[-1]) if num_groups else 0
                    files.append((last_num, f))

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

        return DataManager.Data(np.stack(images_original, axis=0),
                                np.arange(len(images_show)),
                                'tiff',
                                np.stack(images_show, axis=0),
                                parameters={
                                    'vmax_array':vmax_array,
                                    'vmin_array':vmin_array,
                                    'data_type':data_type,
                                })


    def amend_data(self, data, mask = None):
        """函数修改方法
        输入修改的源数据，导出修改的数据包"""
        data_origin = data.data_origin
        # if isinstance(data, dict): # 加roi来的
        #     data_origin = data.data_origin
        # elif isinstance(data, np.ndarray): # 坏点修复来的
        #     data_origin = data
        if mask is not None and mask.shape == data.framesize:
            data_mask = [ ]
            for every_data in data_origin:
                # data_mask.append(np.multiply(every_data, mask)) 目前这里有问题 还没想好怎么改
                every_data[~mask] = data.datamin
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
            'image_import': np.stack(images_show, axis=0),
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

        normalized = self.normalize_data(self.sif_data_original,self.normalize_type)
        return DataManager.Data(np.stack(self.sif_data_original , axis=0),
                                np.stack(self.sif_sorted_times,axis=0),
                                'sif',
                                np.stack(normalized, axis=0))

    @staticmethod
    def normalize_data(
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
    mass_finished = pyqtSignal(DataManager.Data) # 数据读取
    processing_progress_signal = pyqtSignal(int, int) # 进度槽
    processed_result = pyqtSignal(DataManager.ProcessedData)

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

        # 归一化处理
        normalized_frames = self.normalize_data(frames_array)

        self.processing_progress_signal.emit(loading_bar_value+1, total_l)
        avi_data = DataManager.Data(frames_array,
                                    np.arange(len(frames)) / fps if fps > 0 else np.arange(len(frames)),
                                    'avi',
                                    normalized_frames,
                                    parameters={'fps': fps,
                                                'frame_size': (width, height),
                                                'duration': duration,})

        self.mass_finished.emit(avi_data)

    """TIF"""
    @pyqtSlot(str)
    def load_tiff(self,path):
        try:
            tiff_files = []
            for f in os.listdir(path):
                if f.lower().endswith(('.tif', '.tiff')):
                    # 提取数字并排序
                    num_groups = re.findall(r'\d+', f)
                    last_num = int(num_groups[-1]) if num_groups else 0
                    tiff_files.append((last_num, f))

            # 按最后一组数字排序
            tiff_files.sort(key=lambda x: x[0])
            frame_numbers, file_names = zip(*tiff_files) if tiff_files else ([], [])

            logging.info(f"找到{len(file_names)}个TIFF文件")

            # 检查数字连续性（使用已排序的frame_numbers）
            unique_nums = sorted(set(frame_numbers))
            is_continuous = (
                    len(unique_nums) == len(frame_numbers) and
                    (unique_nums[-1] - unique_nums[0] + 1) == len(unique_nums)
            )

            # 读取图像数据
            frames = []
            total_files = len(file_names)
            self.processing_progress_signal.emit(0, total_files)

            for i, filename in enumerate(file_names):
                if self.abortion:
                    break

                img_path = os.path.join(path, filename)
                img = tiff.imread(img_path)

                if img is None:
                    logging.warning(f"无法读取文件: {filename}")
                    continue

                frames.append(img)
                self.processing_progress_signal.emit(i + 1, total_files)

            if not frames:
                raise ValueError("没有有效图像数据被读取")

            # 转换为numpy数组
            frames_array = np.stack(frames, axis=0)
            height, width = frames[0].shape

            # 计算统计信息
            normalized_frames = self.normalize_data(frames_array)

            # 生成时间点
            if is_continuous:
                time_points = (np.array(frame_numbers) - frame_numbers[0])
                logging.info("使用文件名数字作为时间序列")
            else:
                time_points = np.arange(len(frames))
                logging.info("使用默认顺序作为时间序列")

            tiff_data = DataManager.Data(frames_array,
                                         time_points,
                                         'tiff',
                                         normalized_frames,
                                         parameters={
                                                    'frame_size': (width, height),
                                                    'original_files': tiff_files})
            self.mass_finished.emit(tiff_data)

        except Exception as e:
            logging.error(f"处理TIFF序列时出错: {str(e)}")
            self.processing_progress_signal.emit(1, 1)

    @pyqtSlot(DataManager.Data,int,bool)
    def pre_process(self,data,bg_num = 360,unfold=True):
        """数据预处理，包含背景去除，数组展开"""
        try:
            logging.info("开始预处理...")
            self.processing_progress_signal.emit(0, 100)
            timer = QElapsedTimer()
            timer.start()
            # 1. 提取前n帧计算背景帧
            if data.datatype != np.float32:
                data_origin = data.data_origin.astype(np.float32) # 注意这里开始原本Uint8 转为了F32
            total_frames = data.timelength

            # 计算背景帧 (前n帧的平均)
            self.processing_progress_signal.emit(10, 100)
            bg_frame = np.median(data_origin[:bg_num], axis=0)
            self.processing_progress_signal.emit(20, 100)

            # 2. 所有帧减去背景

            # processed_data = (data_origin - bg_frame[np.newaxis, :, :]) / bg_frame[np.newaxis, :, :]
            # self.processing_progress_signal.emit(50, 100)
            processed_data = np.empty_like(data_origin)
            for i in range(total_frames):
                if self.abortion:
                    return None

                # 减去背景帧
                processed_data[i] = (data_origin[i] - bg_frame ) / bg_frame

                # 每隔10%的进度更新一次
                if i % max(1, total_frames // 10) == 0:
                    progress_value = 20 + int(60 * i / total_frames)
                    self.processing_progress_signal.emit(progress_value, 100)
            self.processing_progress_signal.emit(80, 100)
            # 保存结果
            processed = DataManager.ProcessedData(data.timestamp,
                                                  f'{data.name}@EM_pre',
                                                  'EM_pre_processed',
                                                  data_processed=processed_data,
                                                  out_processed={})
            processed.out_processed['bg_frame'] = bg_frame
            # 3. 展开为二维数组
            self.processing_progress_signal.emit(95, 100)
            if unfold:
                T, H, W = processed_data.shape
                unfolded_data = processed_data.reshape((T, H * W)).T
            else:
                unfolded_data = None

            processed = DataManager.ProcessedData(data.timestamp,
                                                  f'{data.name}@EM_pre',
                                                  'EM_pre_processed',
                                                  data_processed=processed_data,
                                                  out_processed={
                                                      'bg_frame': bg_frame,
                                                      'unfolded_data': unfolded_data,
                                                  })

            self.processed_result.emit(processed)
            self.processing_progress_signal.emit(100, 100)
            logging.info(f"预处理完成，总耗时{timer.elapsed()}ms")


        except Exception as e:
            logging.error(f"预处理错误: {str(e)}")

    @pyqtSlot(DataManager.ProcessedData,float,int,int,int,int,str)
    def quality_stft(self,data,target_freq: float,fps:int, window_size: int, noverlap: int,
                    custom_nfft: int, window_type: str):
        """STFT质量分析"""
        if not data:
            raise ValueError("请先进行预处理")
        timer = QElapsedTimer()
        timer.start()

        unfolded_data = data.out_processed['unfolded_data']  # [像素数 x 帧数]

        # 窗函数的选择和生成
        window = self.get_window(window_type, window_size)

        mean_signal = np.mean(unfolded_data, axis=0)
        # if len(mean_signal) % 2 == 1:
        #     mean_signal = mean_signal[:-1]
        # mean_signal = hilbert(mean_signal)
        # wvd = WignerVilleDistribution(mean_signal)
        # wvd.run()
        # wvd.plot(kind='contour', extent=[0, 1000, 0, 60])
        # wvd.plot(kind='contour', sqmod= True,extent=[0, len(mean_signal), 0, 60])

        f, t, Zxx = signal.stft(
            mean_signal,
            fs=fps,
            window=window,
            nperseg=window_size,
            noverlap=noverlap,
            nfft=custom_nfft,
            return_onesided=True
        )
        # 发送平均信号STFT结果
        # self.avg_stft_result.emit(f, t, np.abs(Zxx), target_freq)
        # self.out_length = Zxx.shape[1]
        # self.target_idx = np.argmin(np.abs(f - target_freq))
        # self.time_series = t
        self.processed_result.emit(DataManager.ProcessedData(data.timestamp,
                                                             f'{data.name}@stft_q',
                                                             'stft_quality',
                                                             data_processed=np.abs(Zxx),
                                                             out_processed={
                                                                 'out_length': Zxx.shape[1],
                                                                 'frequencies':f,
                                                                 'time_series':t,
                                                                 'target_freq':target_freq,
                                                                 'target_idx':np.argmin(np.abs(f - target_freq)),
                                                             })
                                   )



    @pyqtSlot(DataManager.ProcessedData,float,int,int,int,int,str)
    def python_stft(self,data, target_freq: float,fps:int, window_size: int, noverlap: int,
                    custom_nfft: int, window_type: str):
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
            if not data:
                raise ValueError("请先进行预处理")
            if 'out_length' not in data.out_processed:
                raise ValueError("请先进行质量评估")

            timer = QElapsedTimer()
            timer.start()

            target_idx = data.out_processed['target_idx']
            out_length = data.out_processed['out_length']
            time_series = data.out_processed['time_series']
            data = next(data for data in reversed(data.history) if data.type_processed == "EM_pre_processed")# [像素数 x 帧数]
            frame_size = data.framesize  # (宽度, 高度)
            unfolded_data = data.out_processed['unfolded_data']

            # 2. 计算采样率和FFT参数
            total_frames = unfolded_data.shape[1]
            total_pixels = unfolded_data.shape[0]
            nfft = custom_nfft
            if nfft < window_size: # 确保nfft大于等于窗长度
                nfft = window_size

            self.processing_progress_signal.emit(1, total_pixels)

            # 4. 初始化结果数组
            width, height = frame_size
            stft_py_out = np.zeros((out_length, height, width), dtype=np.float32)

            # 窗函数的选择和生成
            window = self.get_window(window_type, window_size)

            # 5. 逐像素STFT处理
            self.processing_progress_signal.emit(0, total_pixels)

            # 找到目标频率最近的索引
            # target_idx = np.argmin(np.abs(f - target_freq))
            # 对每个像素执行STFT
            for i in range(total_pixels):
                if self.abortion:
                    return

                pixel_signal = unfolded_data[i, :]

                # 计算当前像素的STFT
                _, _, Zxx = signal.stft(
                    pixel_signal,
                    fs=fps,
                    window=window,
                    nperseg=window_size,
                    noverlap=noverlap,
                    nfft=nfft,
                    return_onesided=False
                )

                # 提取目标频率处的幅度
                magnitude = np.abs(Zxx[target_idx, :]) * 5

                # 将结果存入对应像素位置
                y = i // width
                x = i % width
                stft_py_out[:, y, x] = magnitude

                # 每100个像素更新一次进度
                if i % 100 == 0:
                    self.processing_progress_signal.emit(i, total_pixels)

            # 6. 发送完整结果
            self.processed_result.emit(DataManager.ProcessedData(data.timestamp,
                                                               f'{data.name}@r_stft',
                                                               'ROI_stft',
                                                               data_processed=stft_py_out,
                                                               out_processed={
                                                                   'time_series' : time_series,
                                                               }))
            self.processing_progress_signal.emit(total_pixels, total_pixels)
            logging.info(f"计算完成，总耗时{timer.elapsed()}ms")

        except Exception as e:
            logging.error(f"STFT计算错误: {str(e)}")

    @pyqtSlot(DataManager.ProcessedData,float,int,int,str)
    def quality_cwt(self,data, target_freq: float, fps: int, totalscales: int, wavelet: str = 'morl'):
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

            timer = QElapsedTimer()
            timer.start()

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
            width, height = frame_size
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
                                                                 }))
            self.processing_progress_signal.emit(100, 100)
        except Exception as e:
            logging.error(e)

    @pyqtSlot(DataManager.ProcessedData,float, int, int,str, float)
    def python_cwt(self,data, target_freq: float, fps: int, totalscales: int, wavelet: str, cwt_scale_range: float):
        """
        执行逐像素CWT分析
        参数:
        参数:
            target_freq: 目标分析频率(Hz)
            EM_fps: 采样频率
            scales: 尺度数组，控制小波变换的频率分辨率
            wavelet: 使用的小波类型(默认为'morl'墨西哥帽小波)
        """
        try:
            if not data:
                raise ValueError("请先进行预处理")

            timer = QElapsedTimer()
            timer.start()

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
            self.processing_progress_signal.emit(0, total_pixels)
            # 初始化结果数组
            width, height = frame_size
            cwt_py_out = np.zeros((self.coefficients.shape[1], height, width), dtype=np.float32)

            # 5. 逐像素STFT处理
            self.processing_progress_signal.emit(1, total_pixels)

            # 找到目标频率最近的索引
            # target_idx = np.argmin(np.abs(self.frequencies - target_freq))
            mid_idx = totalscales // 8

            # 对每个像素执行STFT
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
                cwt_py_out[:, y, x] = magnitude_avg

                # 每100个像素更新一次进度
                if i % 100 == 0:
                    self.processing_progress_signal.emit(i, total_pixels)

            # 发送完整结果
            times = np.arange(cwt_py_out.shape[0]) / fps
            self.processed_result.emit(DataManager.ProcessedData(data.timestamp,
                                                                 f'{data.name}@cwt',
                                                                 'ROI_cwt',
                                                                 data_processed=cwt_py_out,
                                                                 out_processed={
                                                                     'time_series': times,
                                                                 }))
            self.processing_progress_signal.emit(total_pixels, total_pixels)
            logging.info(f"计算完成，总耗时{timer.elapsed()}ms")

        except Exception as e:
            logging.error(f"CWT计算错误: {str(e)}")

    def normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    @pyqtSlot(np.ndarray,str,str,str)
    def export_EM_data(self, result, output_dir, prefix, format_type='tif'):
        """
        时频变换后目标频率下的结果导出
        支持多种格式: tif, avi, png, gif

        参数:
            result: 输入数据数组
            output_dir: 输出目录路径
            prefix: 文件前缀
            format_type: 导出格式 ('tif', 'avi', 'png', 'gif')
        """
        format_type = format_type.lower()

        # 根据格式类型调用不同的导出函数
        if format_type == 'tif':
            return self.export_as_tif(result, output_dir, prefix)
        elif format_type == 'avi':
            return self.export_as_avi(result, output_dir, prefix)
        elif format_type == 'png':
            return self.export_as_png(result, output_dir, prefix)
        elif format_type == 'gif':
            return self.export_as_gif(result, output_dir, prefix)
        else:
            logging.error(f"不支持的格式类型: {format_type}")
            raise ValueError(f"不支持格式: {format_type}。请使用 'tif', 'avi', 'png' 或 'gif'")

    def export_as_tif(self, result, output_dir, prefix):
        """导出为TIFF格式序列"""
        num_frames = result.shape[0]
        num_digits = len(str(num_frames))
        self.processing_progress_signal.emit(0, num_frames)
        created_files = []

        for frame_idx in range(num_frames): # 生成带序号的完整文件路径
            frame_name = f"{prefix}-{frame_idx:0{num_digits+1}d}.tif"
            output_path = os.path.join(output_dir, frame_name)

            tiff.imwrite(output_path, result[frame_idx], photometric='minisblack')
            created_files.append(output_path)
            self.processing_progress_signal.emit(frame_idx + 1, num_frames)

        logging.info(f'完成TIFF导出，目标文件夹{output_dir}, 总数{num_frames}张')
        return

    def export_as_avi(self, result, output_dir, prefix):
        """导出为AVI视频格式"""
        num_frames = result.shape[0]
        self.processing_progress_signal.emit(0, num_frames)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 规范化数据到0-255范围 (假设输入是浮点数)
        # normalized = np.zeros_like(result, dtype=np.uint8)
        if result.dtype != np.uint8:
            normalized = ((result - result.min()) / (result.max() - result.min()) * 255).astype(np.uint8)
        else:
            normalized = result

        # 创建视频编写器
        output_path = os.path.join(output_dir, f"{prefix}.avi")
        height, width = result.shape[1], result.shape[2]

        # 根据帧数调整FPS
        fps = max(10, min(30, num_frames // 10))

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

        for frame_idx in range(num_frames):
            out.write(normalized[frame_idx])
            self.processing_progress_signal.emit(frame_idx + 1, num_frames)

        out.release()
        logging.info(f'完成AVI导出: {output_path}, 总数{num_frames}帧')
        return [output_path]

    def export_as_png(self, result, output_dir, prefix):
        """导出为PNG格式序列"""
        num_frames = result.shape[0]
        num_digits = len(str(num_frames))
        self.processing_progress_signal.emit(0, num_frames)
        created_files = []

        # 规范化数据
        normalized = np.zeros_like(result, dtype=np.uint8)
        if result.dtype != np.uint8:
            normalized = ((result - result.min()) / (result.max() - result.min()) * 255).astype(np.uint8)
        else:
            normalized = result

        for frame_idx in range(num_frames):
            frame_name = f"{prefix}-{frame_idx:0{num_digits+1}d}.png"
            output_path = os.path.join(output_dir, frame_name)

            # 使用PIL保存PNG
            img = Image.fromarray(normalized[frame_idx])
            img.save(output_path)
            created_files.append(output_path)
            self.processing_progress_signal.emit(frame_idx + 1, num_frames)

        logging.info(f'完成PNG导出，目标文件夹{output_dir}, 总数{num_frames}张')
        return created_files

    def export_as_gif(self, result, output_dir, prefix):
        """导出为GIF动画"""
        num_frames = result.shape[0]
        self.processing_progress_signal.emit(0, num_frames)

        # 规范化数据
        # normalized = np.zeros_like(result, dtype=np.uint8)
        if result.dtype != np.uint8:
            normalized = ((result - result.min()) / (result.max() - result.min()) * 255).astype(np.uint8)
        else:
            normalized = result

        # 创建PIL图像列表
        images = []
        for frame_idx in range(num_frames):
            images.append(Image.fromarray(normalized[frame_idx]))
            self.processing_progress_signal.emit(frame_idx + 1, num_frames)

        # 保存GIF
        output_path = os.path.join(output_dir, f"{prefix}.gif")
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=100,  # 每帧持续时间(毫秒)
            loop=0  # 无限循环
        )

        logging.info(f'完成GIF导出: {output_path}, 总数{num_frames}帧')
        return [output_path]

    def get_window(self,window_type, window_size):
        try:
            if window_type == 'gaussian':
                window = signal.get_window((window_type, window_size / 6), window_size, fftbins=False)
            elif window_type == 'general_gaussian':
                window = signal.get_window((window_type, 1.5, window_size / 6), window_size, fftbins=False)
            else:
                window = signal.get_window(window_type, window_size, fftbins=False)
            return window
        except Exception as e:
            logging.error(f'Window Fault:{e}')

    @pyqtSlot(DataManager.ProcessedData)
    def accumulate_amplitude(self,data:DataManager.ProcessedData):
        """累计时间振幅图"""
        self.processed_result.emit(DataManager.ProcessedData(data.timestamp,
                                         f'{data.name}@atam',
                                         "Accumulated_time_amplitude_map",
                                         data_processed=np.sum(data.data_processed, axis=0)))
        logging.info("累计时间振幅计算已完成")

    @staticmethod
    def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y, offset):
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
        x, y = coords
        return A * np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2))) + offset

    @pyqtSlot(DataManager.ProcessedData)
    def twoD_gaussian_fit(self,data_3d:DataManager.ProcessedData,zm = 2,thr = 2.5):
        """
        对三维时序数据逐帧进行二维高斯拟合

        参数:
        data_3d: numpy.ndarray, 三维数组 (T, H, W)
        zm 插值系数
        返回:
        results: list of dict, 每帧的拟合参数
        """
        T, H, W = data_3d.datashape
        x_zmed = W * zm - (zm - 1)
        y_zmed = H * zm - (zm - 1)
        X, Y = np.meshgrid(np.arange(x_zmed), np.arange(y_zmed))

        amplitudes = np.zeros(T)
        centers_x = np.zeros(T)
        centers_y = np.zeros(T)

        for m in range(T):
            frame = data_3d.data_processed[m]

            # 图像插值
            if zm >1:
                Z = zoom(frame, zm, order=3)
            else:
                Z = frame

            # 检查是否有超过阈值的点
            if np.any(Z > thr):
                max_value = np.max(Z)
                y0_g, x0_g = np.unravel_index(np.argmax(Z), Z.shape)

                # 初始参数 [A, x0, sigmax, y0, sigmay, b]
                x0 = [max_value, x0_g, 1.0, y0_g, 1.0, np.mean(Z)]

                # 参数边界
                lb = [0, 0, 0.1, 0, 0.1, 0]
                ub = [100, x_zmed, (x_zmed / 2) ** 2, y_zmed, (y_zmed / 2) ** 2, max_value]

                try:
                    # 二维高斯拟合
                    popt, _ = curve_fit(self.gaussian_2d, (X.ravel(),Y.ravel()), Z.ravel(),
                                        p0=x0, bounds=(lb, ub), maxfev=5000)

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

        return amplitudes, centers_x, centers_y

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

    # 3. 上升/下降时间常数拟合
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
    # def analyze_single_peak_data(data_3d, T, x_m, y_m):
    #     """
    #     完整分析流程
    #
    #     参数:
    #     data_3d: 三维时序数据 (T, H, W)
    #     T: 时间轴
    #     x_m, y_m: ROI起始坐标
    #
    #     返回:
    #     所有分析结果
    #     """
    #     # 1. ROI高斯拟合
    #     amplitudes, centers_x, centers_y = fit_gaussian_roi(data_3d, x_m, y_m)
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
