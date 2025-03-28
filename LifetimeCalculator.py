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


class LifetimeCalculator:
    """
    载流子寿命计算类
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
    def calculate_lifetime(time_series, time_points, model_type='single'):
        """
        计算载流子寿命

        参数:
            time_series: 时间序列信号
            time_points: 对应的时间点
            model_type: 'single' 或 'double' 表示单/双指数衰减

        返回:
            拟合参数和寿命值
        """
        # 找到最大值位置
        max_idx = np.argmax(time_series)
        decay_signal = time_series[max_idx:]
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

                # 计算R方
                y_pred = LifetimeCalculator.single_exponential(decay_time, *popt)
                ss_res = np.sum((decay_signal - y_pred) ** 2)
                ss_tot = np.sum((decay_signal - np.mean(decay_signal)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                if r_squared <= 0.8:
                    lifetime = 0
                else:
                    lifetime = popt[1]  # tau
                return popt, lifetime

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