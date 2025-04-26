import glob
import os
import re
import numpy as np
import tifffile as tiff
import sif_parser as sr
from typing import List, Union

class DataProcessor:
    """本类仅包含导入数据时的数据处理"""
    def __init__(self,path):
        self.path = path

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
                # data_mask.append(np.multiply(every_data, mask))
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
                    background = sr.np_open(filepath)[0][0]
                    continue

                # 否则尝试提取时间点（文件名中的数字）
                match = re.search(r'(\d+)', name)
                if match:
                    time = int(match.group(1))
                    data = sr.np_open(filepath)[0][0]
                    time_data[time] = data

        # 检查是否找到背景
        if background is None:
            raise ValueError("未找到背景文件（文件名应包含 'no'）")

        # 按时间排序
        sorted_times = sorted(time_data.keys())

        # 创建三维数组（时间, 高度, 宽度）并减去背景
        sample_data = next(iter(time_data.values()))
        data_3d = np.zeros((len(sorted_times), *sample_data.shape), dtype=np.float32)

        for i, time in enumerate(sorted_times):
            data_3d[i] = time_data[time] - background

        return data_3d, sorted_times