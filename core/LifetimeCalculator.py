import glob
import logging
import os
import re

import numpy as np
from PyQt5.QtWidgets import QMessageBox
from scipy.ndimage import convolve
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot, QElapsedTimer
from DataManager import *




class LifetimeCalculator:
    """
    载流子寿命计算类（此类为静态方法类，不要放别的进来）
    """
    _cal_params = {
        'from_start_cal':False,
        'r_squared_min': 0.6,
        'peak_range': (0.0, 10.0),
        'tau_range': (1e-3, 1e2)
    }

    @classmethod
    def set_cal_parameters(cls,cal_set_params):
        """更新参数"""
        from_start_cal = cal_set_params['from_start_cal']
        r_squared_min = cal_set_params['r_squared_min']
        peak_range = (cal_set_params['peak_min'], cal_set_params['peak_max'])
        tau_range = (cal_set_params['tau_min'], cal_set_params['tau_max'])
        cls._cal_params['from_start_cal'] = from_start_cal
        cls._cal_params['r_squared_min'] = r_squared_min
        cls._cal_params['peak_range'] = peak_range
        cls._cal_params['tau_range'] = tau_range
        pass

    @staticmethod
    def single_exponential(t, A, tau, C):
        """单指数衰减模型"""
        return A * np.exp(-t / tau) + C

    @staticmethod
    def double_exponential(t, A1, tau1, A2, tau2, C):
        """双指数衰减模型"""
        return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + C

    @staticmethod
    def calculate_lifetime(data_type, time_series, time_points, arg_bundle = None, model_type='single'):
        """
        计算载流子寿命
        """
        params = LifetimeCalculator._cal_params
        from_start_cal = params['from_start_cal']
        r_squared_min = params['r_squared_min']
        peak_range = params['peak_range']
        tau_range = params['tau_range']
        

        # 获得具有实际意义的信号序列
        phy_signal = None
        if data_type == 'central negative':
            phy_signal = -time_series
            max_idx = np.argmax(phy_signal)
            if not from_start_cal:
                decay_signal = abs(phy_signal[max_idx:]) # 全部正置 且从最大值之后开始拟合
                decay_time = time_points[max_idx:] - time_points[max_idx]
            else:
                decay_signal= abs(phy_signal)
                decay_time = time_points
        elif data_type == 'central positive':
            phy_signal = time_series
            max_idx = np.argmax(phy_signal)
            if not from_start_cal:
                decay_signal = abs(phy_signal[max_idx:]) # 全部正置 且从最大值之后开始拟合
                decay_time = time_points[max_idx:] - time_points[max_idx]
            else:
                decay_signal = abs(phy_signal)
                decay_time = time_points
        elif data_type == 'sif':
            phy_signal = time_series
            max_idx = np.argmax(phy_signal)
            if not from_start_cal:
                decay_signal = phy_signal[max_idx:] # 全部正置 且从最大值之后开始拟合
                decay_time = time_points[max_idx:] - time_points[max_idx]
            else:
                decay_signal = phy_signal
                decay_time = time_points
        else: # 不可能走到这里，我只是觉得代码高亮不舒服 所以加的 (在alpha 1.10.0时遇到了这个问题，于是乎打脸了）
            phy_signal = time_series
            max_idx = np.argmax(phy_signal)
            if not from_start_cal:
                decay_signal = phy_signal[max_idx:]  # 全部正置 且从最大值之后开始拟合
                decay_time = time_points[max_idx:] - time_points[max_idx]
            else:
                decay_signal = phy_signal
                decay_time = time_points

        # 初始猜测
        A_guess = np.max(decay_signal) - np.min(decay_signal)
        tau_guess = (decay_time[-1] - decay_time[0]) / 5
        C_guess = np.min(decay_signal)

        try:
            if model_type == 'single':
                # 单指数拟合
                if peak_range[0] <= max_idx <= peak_range[1]: # 峰值位置筛选
                    popt, pcov = curve_fit(
                        LifetimeCalculator.single_exponential,
                        decay_time,
                        decay_signal,
                        p0=[A_guess, tau_guess, C_guess],
                        bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]))
                    if popt[1] <= tau_range[0] or popt[1] >= tau_range[1]: # 寿命大小筛选
                        lifetime = 0
                        r_squared = np.nan
                    else:
                        # 计算R方
                        y_pred = LifetimeCalculator.single_exponential(decay_time, *popt)
                        ss_res = np.sum((decay_signal - y_pred) ** 2)
                        ss_tot = np.sum((decay_signal - np.mean(decay_signal)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        if r_squared <= r_squared_min: # R方筛选
                            lifetime = 0
                        else:
                            lifetime = popt[1]  # tau
                else:
                    lifetime = 0
                    r_squared = np.nan
                    popt = [0,0,0]
                return popt, lifetime, r_squared, phy_signal

            elif model_type == 'double':
                # 双指数拟合
                if peak_range[0] <= max_idx <= peak_range[1]:  # 峰值位置筛选
                    A2_guess = A_guess / 2
                    tau2_guess = tau_guess * 2

                    popt, pcov = curve_fit(
                        LifetimeCalculator.double_exponential,
                        decay_time,
                        decay_signal,
                        p0=[A_guess, tau_guess, A2_guess, tau2_guess, C_guess],
                        bounds=([0, 0, 10, 10, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]))

                    # 计算平均寿命
                    A1, tau1, A2, tau2, C = popt
                    # 计算R方
                    if (tau1 <= tau_range[0] or tau1 >= tau_range[1]) and (tau2 <= tau_range[0] or tau2 >= tau_range[1]):  # 寿命大小筛选
                        tau1,tau2 = (0,0)
                        r_squared = np.nan
                    else:
                        y_pred = LifetimeCalculator.double_exponential(decay_time, *popt)
                        ss_res = np.sum((decay_signal - y_pred) ** 2)
                        ss_tot = np.sum((decay_signal - np.mean(decay_signal)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        if r_squared <= r_squared_min: # R方筛选
                            tau1,tau2 = (0,0)
                        else:
                            tau1,tau2 = (popt[1], popt[3]) # tau1 and 2
                else:
                    tau1,tau2 = (0,0)
                    r_squared = np.nan
                    popt = [0, 0, 0,0,0]
                return popt, (tau1, tau2), r_squared, phy_signal

        except Exception as e:
            # 拟合失败时返回NaN
            if model_type == 'single':
                return [np.nan, np.nan, np.nan], np.nan, np.nan ,phy_signal
            else:
                return [np.nan, np.nan, np.nan, np.nan, np.nan], (np.nan, np.nan), np.nan ,phy_signal

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
        h, w = data.framesize
        data_type = data.parameters['data_type'] if 'data_type' in data.parameters else None

        # 创建区域掩模
        if shape == 'square':
            y_min = max(0, y - (size - 1) // 2)
            y_max = min(h, y_min + size)
            x_min = max(0, x - (size - 1) // 2)
            x_max = min(w, x_min + size)
            mask = np.zeros((h, w), dtype=bool)
            mask[y_min:y_max, x_min:x_max] = True
        elif shape == 'circle':  # circle
            yy, xx = np.ogrid[:h, :w]
            mask = (yy - y) ** 2 + (xx - x) ** 2 <= (size-1) ** 2
        elif shape == 'custom':  # 留给绘制roi
            pass

        # 计算区域平均时间曲线
        region_data = data.data_origin[:, mask]
        avg_curve = np.mean(region_data, axis=1)

        # 计算寿命
        if model_type == 'single':
            popt, lifetime, r_squared, phy_signal = LifetimeCalculator.calculate_lifetime(data_type, avg_curve, time_points, model_type='single')
            if LifetimeCalculator._cal_params['from_start_cal']: # 从头算
                fit_curve = LifetimeCalculator.single_exponential(
                time_points, popt[0], popt[1], popt[2])
            else: # 从最大值算
                fit_curve = LifetimeCalculator.single_exponential(
                time_points[np.argmax(phy_signal):] - time_points[np.argmax(phy_signal)],
                popt[0], popt[1], popt[2])
        elif model_type == 'double':
            popt, lifetime, r_squared, phy_signal = LifetimeCalculator.calculate_lifetime(data_type, avg_curve, time_points, model_type='double')
            if LifetimeCalculator._cal_params['from_start_cal']: # 从头算
                fit_curve = LifetimeCalculator.double_exponential(
                time_points, popt[0], popt[1], popt[2], popt[3], popt[4])
            else: # 从最大值算
                fit_curve = LifetimeCalculator.double_exponential(
                time_points[np.argmax(phy_signal):] - time_points[np.argmax(phy_signal)],
                popt[0], popt[1], popt[2], popt[3], popt[4])
        return lifetime, fit_curve, mask, phy_signal, r_squared

    @staticmethod
    def apply_custom_kernel(data, kernel_type='smooth'):
        """
        应用自定义卷积核
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

    @staticmethod
    def gaussian_func(x, a, mu, sigma):
        """高斯函数"""
        return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    @staticmethod
    def gaussian_fit(x, y):
        """高斯拟合"""
        # 初始猜测参数 [幅值, 均值, 标准差]
        a_guess = max(abs(y))
        mu_guess = np.median(x)
        sigma_guess = (max(x) - min(x)) / 4

        popt, pcov = curve_fit(
            LifetimeCalculator.gaussian_func,
            x, y,
            p0=[a_guess, mu_guess, sigma_guess]
        )
        return popt, pcov

class CalculationThread(QObject):
    """仅在线程中使用，目前未加锁（仍无必要）"""
    cal_running_status = pyqtSignal(bool)
    processed_result = pyqtSignal(ProcessedData)
    processed_result = pyqtSignal(ProcessedData)
    processed_result = pyqtSignal(ProcessedData)
    calculating_progress_signal = pyqtSignal(int, int)
    stop_thread_signal = pyqtSignal()
    cal_time = pyqtSignal(float)
    update_status = pyqtSignal(str,str)


    def __init__(self):
        super(CalculationThread,self).__init__()
        logging.info('计算线程已载入')
        self._is_calculating = False

    @pyqtSlot(dict, float, tuple, str, int, str)
    def region_analyze(self,data,time_unit,center,shape,size,model_type):
        """分析选定区域"""
        logging.info("开始计算选区载流子寿命...")
        self.calculating_progress_signal.emit(1, 3)
        self.cal_running_status.emit(True)
        try:
            # 计时器
            timer = QElapsedTimer()
            timer.start()
            # 获取参数
            time_points = data.time_point * time_unit
            self.calculating_progress_signal.emit(2, 3)
            # 执行区域分析
            lifetime, fit_curve, mask, phy_signal, r_squared = LifetimeCalculator.analyze_region(
                data, time_points, center, shape, size, model_type)

            self.processed_result.emit(ProcessedData(data.timestamp,
                                                       f'{data.name}@r-lft',
                                                       'ROI_lifetime',
                                                       time_point=time_points,
                                                       out_processed={'phy_signal': phy_signal,
                                                                      'lifetime': lifetime,
                                                                      'fit_curve': fit_curve,
                                                                      'r_squared': r_squared,
                                                                      'model_type': model_type,
                                                                      'boundary': {'min':data.datamin, 'max':data.datamax}}))
            self.cal_time.emit(timer.elapsed())
            logging.info("计算完成!")
            self.calculating_progress_signal.emit(3, 3)
            self.cal_running_status.emit(False)
            self.stop_thread_signal.emit()
        except Exception as e:
            self.update_status.emit(f'区域数据拟合出错:{e}','error')
            self._is_calculating = False
            self.cal_running_status.emit(False)
            self.stop_thread_signal.emit()

    @pyqtSlot(dict, float, str)
    def distribution_analyze(self,data,time_unit,model_type):
        """分析全图载流子寿命"""
        self._is_calculating = True
        self.cal_running_status.emit(True)
        try:

            # 计时器
            timer = QElapsedTimer()
            timer.start()
            time_points = data.time_point * time_unit
            data_type = data.parameters['data_type'] if 'data_type' in data.parameters else None

            # 计算每个像素的寿命
            height, width = data.framesize
            lifetime_map = np.zeros((height, width))
            logging.info("开始计算载流子寿命...")

            loading_bar_value =0 #进度条
            total_l = height * width
            for i in range(height):
                if self._is_calculating :# 线程关闭控制（目前仅针对长时计算）
                    for j in range(width):
                        time_series = data.data_origin[:, i, j]
                        # 用皮尔逊系数判断噪音(滑动窗口法)
                        window_size = min(10, len(time_points) // 2)
                        pr = []
                        for k in range(len(time_series) - window_size):
                            window = time_series[k:k + window_size]
                            time_window = time_points[k:k + window_size]
                            r, _ = pearsonr(time_window, window)
                            pr.append(r)
                            if abs(r) >= 0.8:
                                _, lifetime, r_squared, _ = LifetimeCalculator.calculate_lifetime(data_type, time_series, time_points, model_type)
                                continue
                            else:
                                pass
                        if np.all(np.abs(pr) < 0.8):
                            lifetime = np.nan
                        else:
                            pass
                        lifetime_map[i, j] = lifetime if not np.isnan(lifetime) else 0
                        loading_bar_value += 1
                        self.calculating_progress_signal.emit(loading_bar_value, total_l)
                else:
                    logging.info("计算终止")
                    self.calculating_progress_signal.emit(total_l, total_l) # 进度条更新
                    self.cal_running_status.emit(False)
                    self.stop_thread_signal.emit() # 目前来说，计算终止也会关闭线程，后续可考虑分开命令
                    return
            logging.info("计算完成!")
            # 显示结果
            smoothed_map = LifetimeCalculator.apply_custom_kernel(lifetime_map)
            self.processed_result.emit(ProcessedData(data.timestamp,
                                                      f'{data.name}@d-lft',
                                                      'lifetime_distribution',
                                                      data_processed=smoothed_map))
            self.cal_time.emit(timer.elapsed())
            self._is_calculating = False
            self.cal_running_status.emit(False)
            self.stop_thread_signal.emit()
        except Exception as e:
            self.update_status.emit(f'区域数据拟合出错:{e}','error')
            self._is_calculating = False
            self.cal_running_status.emit(False)
            self.stop_thread_signal.emit()


    def diffusion_calculation(self,frame_data,time_unit,space_unit,timestamp,name):
        # 存储拟合方差结果 [时间, 方差]
        self._is_calculating = True
        self.cal_running_status.emit(True)
        timer = QElapsedTimer()
        timer.start()

        sigma_results = np.zeros((2, len(frame_data)))
        time_series = []
        loading_bar_value =0 #进度条
        total_l = len(frame_data)
        fitting_result = []
        signal_series = []
        if self._is_calculating:  # 线程关闭控制（目前仅针对循环计算）
            for i, (frame_idx, data) in enumerate(frame_data.items()):
                positions = data[:, 0] * space_unit # 此处合并单位长度
                intensities = data[:, 1]

                # 高斯拟合
                try:
                    popt, pcov = LifetimeCalculator.gaussian_fit(positions, intensities)
                    fit_curve = LifetimeCalculator.gaussian_func(positions, *popt)

                    # 保存拟合结果
                    sigma_results[0, i] = frame_idx * time_unit # 此处合并单位时间
                    sigma_results[1, i] = popt[2] ** 2  # 保存方差
                    time_series.append(frame_idx * time_unit)  # 时间点单独算
                    fitting_result.append(np.stack((positions,fit_curve),axis=0))
                    signal_series.append(np.stack((positions,intensities),axis=0))
                except Exception as e:
                    logging.error(f"拟合失败,报错{e}")
                    self.update_status.emit(f'扩散拟合失败:{e}', 'error')
                    self.stop_thread_signal.emit()
                loading_bar_value += 1
                self.calculating_progress_signal.emit(loading_bar_value, total_l)
        else:
            logging.info("计算终止")
            self.calculating_progress_signal.emit(total_l, total_l)  # 进度条更新
            self.cal_running_status.emit(False)
            self.stop_thread_signal.emit()  # 目前来说，计算终止也会关闭线程，后续可考虑分开命令
            return

        dif_data_dict = {
            'sigma': np.stack(sigma_results, axis=0),
            'signal': np.stack(signal_series, axis=0),
            'fitting': np.stack(fitting_result, axis=0),
            'time_series': np.stack(time_series, axis=0)
        }
        self.processed_result.emit(ProcessedData(timestamp,
                                                 f'{name}@dif',
                                                 'diffusion',
                                                 out_processed=dif_data_dict))
        self.cal_time.emit(timer.elapsed())
        self._is_calculating = False
        self.cal_running_status.emit(False)
        self.stop_thread_signal.emit()

    def heat_transfer_calculation(self,origin_data):
        """计算传热系数"""
        self._is_calculating = True
        pass

    def stop(self):
        self._is_calculating = False

    def force_stop(self):
        self._is_calculating = False
        self.cal_running_status.emit(False)
        self.stop_thread_signal.emit()