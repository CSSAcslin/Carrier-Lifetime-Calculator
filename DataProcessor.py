import glob
import re
import numpy as np
import tifffile as tiff


class DataProcessor:
    """数据处理"""
    def __init__(self,path):
        self.path = path

    def load_and_sort_files(self,current_group):
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
            data_type = 'central black'
            for every_data in data:
                normalized_data = (every_data - min_all) / (max_all - min_all)
                process_show.append(normalized_data)
            max_mean = np.min(vmean_array)
        else:
            # p-type 信号中心为白色，最强值为负
            data_type = 'central white'
            for every_data in data:
                normalized_data = (max_all - every_data) / (max_all - min_all)
                process_show.append(normalized_data)
            max_mean = np.max(vmean_array)
        return process_show, data_type, max_mean

    def process_files(self, files, time_start_input, time_unit):
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
        images_show, data_type, max_mean = self.process_data(images_original, vmax, vmin, vmean_array)
        return {
            'data_origin': np.stack(images_original, axis=0),
            'data_type': data_type,
            'images': np.stack(images_show, axis=0),
            'time_points': np.arange(float(time_start_input.value()) + len(images_show)) * time_unit,
            'data_mean': max_mean
        }
