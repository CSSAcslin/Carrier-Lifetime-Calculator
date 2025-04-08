import glob
import logging
import os
import re

import numpy as np
from scipy.ndimage import convolve
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
from PyQt5.QtCore import Qt, QObject
import pandas as pd
from scipy.stats import pearsonr


class LifetimeCalculator:
    """
    载流子寿命计算类（此类为静态方法类，不要放别的进来）
    """
    @staticmethod
    def single_exponential(t, A, tau, C):
        """单指数衰减模型"""
        return A * np.exp(-t / tau) + C

    @staticmethod
    def double_exponential(t, A1, tau1, A2, tau2, C):
        """双指数衰减模型"""
        return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + C

    @staticmethod
    def calculate_lifetime(data_type, time_series, time_points, model_type='single'):
        """
        计算载流子寿命

        参数:
            time_series: 时间序列信号
            time_points: 对应的时间点
            model_type: 'single' 或 'double' 表示单/双指数衰减

        返回:
            拟合参数和寿命值
        """
        # 获得具有实际意义的信号序列
        phy_signal = None
        if data_type == 'central negative':
            phy_signal = -time_series
        elif data_type == 'central positive':
            phy_signal = time_series

        # 找到最大值位置
        max_idx = np.argmax(phy_signal)
        decay_signal = abs(phy_signal[max_idx:]) #全部正置
        decay_time = time_points[max_idx:] - time_points[max_idx]

        # 初始猜测
        A_guess = np.max(decay_signal) - np.min(decay_signal)
        tau_guess = (decay_time[-1] - decay_time[0]) / 5
        C_guess = np.min(decay_signal)

        try:
            if model_type == 'single':
                # 单指数拟合
                popt, pcov = curve_fit(
                    LifetimeCalculator.single_exponential,
                    decay_time,
                    decay_signal,
                    p0=[A_guess, tau_guess, C_guess],
                    bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]))
                if popt[1] <=0.001 or popt[1] >=100:
                    lifetime = 0
                    r_squared = np.nan
                else:
                    # 计算R方
                    y_pred = LifetimeCalculator.single_exponential(decay_time, *popt)
                    ss_res = np.sum((decay_signal - y_pred) ** 2)
                    ss_tot = np.sum((decay_signal - np.mean(decay_signal)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    if r_squared <= 0.8:
                        lifetime = 0
                    else:
                        lifetime = popt[1]  # tau
                return popt, lifetime, r_squared, phy_signal

            elif model_type == 'double':
                # 双指数拟合(没做好，不要用)
                A2_guess = A_guess / 2
                tau2_guess = tau_guess * 2

                popt, pcov = curve_fit(
                    LifetimeCalculator.double_exponential,
                    decay_time,
                    decay_signal,
                    p0=[A_guess, tau_guess, A2_guess, tau2_guess, C_guess],
                    bounds=([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf]))

                # 计算平均寿命
                A1, tau1, A2, tau2, C = popt
                # 计算R方
                y_pred = LifetimeCalculator.single_exponential(decay_time, *popt)
                ss_res = np.sum((decay_signal - y_pred) ** 2)
                ss_tot = np.sum((decay_signal - np.mean(decay_signal)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                if r_squared <= 0.8:
                    avg_lifetime = 0
                else:
                    avg_lifetime = (A1 * tau1 + A2 * tau2) / (A1 + A2)
                return popt, avg_lifetime

        except:
            # 拟合失败时返回NaN
            if model_type == 'single':
                return [np.nan, np.nan, np.nan], np.nan
            else:
                return [np.nan, np.nan, np.nan, np.nan, np.nan], np.nan

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
        global lifetime, fit_curve, phy_signal, r_squared
        y, x = center
        h, w = data['data_origin'].shape[1], data['data_origin'].shape[2]
        data_type = data['data_type']

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
        region_data = data['data_origin'][:, mask]
        avg_curve = np.mean(region_data, axis=1)

        # 计算寿命
        if model_type == 'single':
            popt, lifetime, r_squared, phy_signal = LifetimeCalculator.calculate_lifetime(data_type, avg_curve, time_points, 'single')
            fit_curve = LifetimeCalculator.single_exponential(
                time_points[np.argmax(phy_signal):] - time_points[np.argmax(phy_signal)],
                popt[0], popt[1], popt[2])
        elif model_type == 'double':
            popt, lifetime, r_squared, phy_signal = LifetimeCalculator.calculate_lifetime(data_type, avg_curve, time_points, 'double')
            fit_curve = LifetimeCalculator.double_exponential(
                time_points[np.argmax(avg_curve):] - time_points[np.argmax(avg_curve)],
                popt[0], popt[1], popt[2], popt[3], popt[4])

        return lifetime, fit_curve, mask, phy_signal, r_squared

    @staticmethod
    def apply_custom_kernel(data, kernel_type='smooth'):
        """
        应用自定义卷积核
        参数:
            data: 2D numpy数组
            kernel_type:
                'smooth' - 中心0.2周边0.1的平滑核
                'sharpen' - 锐化核(可选扩展)
        返回:
            处理后的数组
        """
        if kernel_type == 'smooth':
            kernel = np.array([
                [0.1, 0.1, 0.1],
                [0.1, 0.2, 0.1],
                [0.1, 0.1, 0.1]
            ])
        else:  # 默认不处理
            return data

        # 边界处理采用镜像模式
        return convolve(data, kernel, mode='mirror')

class CalculationThread(QObject):
    def __init__(self):
        super(CalculationThread,self).__init__()

    def region_analyze(self):
        """分析选定区域"""
        if self.data is None or not hasattr(self, 'time_points'):
            return

        # 获取参数
        self.time_unit = float(self.time_step_input.value())
        self.time_points = self.data['time_points']
        center = (self.region_y_input.value(), self.region_x_input.value())
        shape = 'square' if self.region_shape_combo.currentText() == "正方形" else 'circle'
        size = self.region_size_input.value()
        model_type = 'single' if self.model_combo.currentText() == "单指数衰减" else 'double'

        # 执行区域分析
        lifetime, fit_curve, mask, phy_signal, r_squared = LifetimeCalculator.analyze_region(
            self.data, self.time_points, center, shape, size, model_type)

        # 显示结果
        self.result_display.display_lifetime_curve(phy_signal, lifetime, r_squared, fit_curve,self.time_points, self.data['boundary'])


    def distribution_analyze(self):
        """分析载流子寿命"""
        if self.data is None:
            return

        # 获取时间点
        self._is_calculating = True
        self.time_unit = float(self.time_step_input.value())
        self.time_points = self.data['time_points'] * self.time_unit
        self.data_type = self.data['data_type']
        self.value_mean_max = np.abs(self.data['data_mean'])

        # 获取模型类型
        model_type = 'single' if self.model_combo.currentText() == "单指数衰减" else 'double'

        # 计算每个像素的寿命
        height, width = self.data['data_origin'].shape[1], self.data['data_origin'].shape[2]
        lifetime_map = np.zeros((height, width))
        logging.info("开始计算载流子寿命...")
        loading_bar =0 #进度条
        total_l = height * width
        for i in range(height):
            if self._is_calculating :
                for j in range(width):
                    time_series = self.data['data_origin'][:, i, j]
                    # 用皮尔逊系数判断噪音(滑动窗口法)
                    window_size = min(10, len(self.time_points) // 2)
                    pr = []
                    for k in range(len(time_series) - window_size):
                        window = time_series[k:k + window_size]
                        time_window = self.time_points[k:k + window_size]
                        r, _ = pearsonr(time_window, window)
                        pr.append(r)
                        if abs(r) >= 0.8:
                            _, lifetime, r_squared, _ = LifetimeCalculator.calculate_lifetime(self.data_type, time_series, self.time_points, model_type)
                            continue
                        else:
                            pass
                    if np.all(np.abs(pr) < 0.8):
                        lifetime = np.nan
                    else:
                        pass
                    lifetime_map[i, j] = lifetime if not np.isnan(lifetime) else 0
                    self.update_progress(loading_bar+1, total_l)
                    self.console_widget.update_progress(loading_bar+1, total_l)
            else:
                logging.info("计算终止")
                return
        logging.info("计算完成!")
        # 显示结果
        smoothed_map = LifetimeCalculator.apply_custom_kernel(lifetime_map)
        self.result_display.display_distribution_map(smoothed_map)
