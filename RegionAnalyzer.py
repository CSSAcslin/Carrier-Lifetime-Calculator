import glob
import os
import re

import numpy as np
from scipy.optimize import curve_fit
from PIL import Image
import tifffile as tiff
import matplotlib as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QScrollArea,
                             QFileDialog, QSlider, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
                             )
from PyQt5.QtCore import Qt
import pandas as pd

from transient_data_processing.LifetimeCalculator import LifetimeCalculator


class RegionAnalyzer:
    """特定区域寿命分析器"""

    @staticmethod
    def analyze_region(data, time_points, center, shape='square', size=5, model_type='single'):
        """
        分析特定区域的载流子寿命

        参数:
            data: 3D numpy数组 (time, height, width)
            time_points: 时间点序列
            center: (y, x) 中心坐标
            shape: 'square' 或 'circle'
            size: 区域大小 (正方形边长或圆形半径)
            model_type: 衰减模型类型

        返回:
            avg_curve: 平均时间曲线
            lifetime: 计算得到的寿命
            fit_curve: 拟合曲线
        """
        y, x = center
        h, w = data.shape[1], data.shape[2]

        # 创建区域掩模
        if shape == 'square':
            y_min = max(0, y - size // 2)
            y_max = min(h, y + size // 2 + 1)
            x_min = max(0, x - size // 2)
            x_max = min(w, x + size // 2 + 1)
            mask = np.zeros((h, w), dtype=bool)
            mask[y_min:y_max, x_min:x_max] = True
        else:  # circle
            yy, xx = np.ogrid[:h, :w]
            mask = (yy - y) ** 2 + (xx - x) ** 2 <= size ** 2

        # 计算区域平均时间曲线
        region_data = data[:, mask]
        avg_curve = np.mean(region_data, axis=1)

        # 计算寿命
        if model_type == 'single':
            popt, lifetime = LifetimeCalculator.calculate_lifetime(avg_curve, time_points, 'single')
            fit_curve = LifetimeCalculator.single_exponential(
                time_points[np.argmax(avg_curve):] - time_points[np.argmax(avg_curve)],
                popt[0], popt[1], popt[2])
        elif model_type == 'double':
            popt, lifetime = LifetimeCalculator.calculate_lifetime(avg_curve, time_points, 'double')
            fit_curve = LifetimeCalculator.double_exponential(
                time_points[np.argmax(avg_curve):] - time_points[np.argmax(avg_curve)],
                popt[0], popt[1], popt[2], popt[3], popt[4])

        return avg_curve, lifetime, fit_curve, mask