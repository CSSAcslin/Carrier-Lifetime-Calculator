import glob
import logging
import os
import re
import numpy as np
import tifffile as tiff
import sif_parser 
from skimage.exposure import equalize_adapthist
from typing import List, Union, Optional


class DataProcessor:
    """本类仅包含导入数据时的数据处理"""
    def __init__(self,path,normalize_type='linear'):
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
        data_origin = data['data_origin']
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
            'signal':np.stack(),
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