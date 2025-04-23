import logging

import numpy as np
import matplotlib as plt
import matplotlib.font_manager as fm
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QScrollArea,
                             QFileDialog, QSlider, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
                             )

class ResultDisplayWidget(QWidget):
    """结果热图显示部件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # self.font1 = plt.font_manager.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
        self.init_ui()
        self.font_list = fm.findSystemFonts(fontpaths=r"C:\Windows\Fonts" , fontext='ttf')
        self.chinese_fonts = [f for f in self.font_list if
                         any(c in f.lower() for c in ['simhei', 'simsun', 'microsoft yahei', 'fang'])]
        self.plot_settings = {
            'current_mode': 'heatmap',  # 'heatmap' 或 'curve'
            'line_style': '--',
            'line_width': 2,
            'marker_style': 's',
            'marker_size': 6,
            'color': '#1f77b4',
            'show_grid': False,
            'heatmap_cmap': 'jet',
            'contour_levels': 10
        }

        # 设置Matplotlib默认字体
        if self.chinese_fonts:
            plt.rcParams['font.sans-serif'] = ['microsoft yahei']  # Windows常用
            # 或者使用其他字体如: 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi'
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    def init_ui(self):
        self.layout = QVBoxLayout(self)

        # 创建matplotlib图形
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)


        self.layout.addWidget(self.canvas)

    def update_plot_settings(self, new_settings):
        """更新绘图设置"""
        self.plot_settings.update(new_settings)
        self.update_plot()

    def update_plot(self):
        """根据当前设置重新绘图，有问题 先留着"""
        pass
        # if self.plot_settings['plot_type'] == 'heatmap':
        #     self.display_distribution_map()
        # else:
        #     self.display_lifetime_curve()

    def display_distribution_map(self, lifetime_map):
        """显示寿命热图"""
        self.current_mode = "heatmap"
        self.figure.clear()
        cmap = self.plot_settings['heatmap_cmap']
        levels = self.plot_settings['contour_levels']

        ax = self.figure.add_subplot(111)
        # 显示热图
        im = ax.imshow(lifetime_map, cmap=cmap)
        self.figure.colorbar(im, ax=ax, label='lifetime')
        ax.set_title("载流子寿命分布图")
        ax.axis('off')
        self.figure.tight_layout()
        self.canvas.draw()

        # 保存当前数据
        self.current_data = pd.DataFrame(lifetime_map)

    def display_lifetime_curve(self, phy_signal, lifetime, r_squared, fit_curve,time_points,boundary, model_type):
        """显示区域分析结果"""
        # 使用原来的结果显示区域
        self.current_mode = "curve"
        self.figure.clear()
        line_style = self.plot_settings['line_style']
        line_width = self.plot_settings['line_width']
        marker_style = self.plot_settings['marker_style']
        marker_size = self.plot_settings['marker_size']
        color = self.plot_settings['color']
        show_grid = self.plot_settings['show_grid']

        ax = self.figure.add_subplot(111)
        max_bound = boundary['max']
        min_bound = 0
        ax.set_ylim(min_bound, max_bound)

        # 绘制原始曲线
        # time_points = time_points - time_points[0]  # 从0开始
        ax.plot(time_points, phy_signal,
                markeredgecolor=color, # 点边缘色
                markeredgewidth=line_width,
                label='原始数据',
                 marker=marker_style,       # 方形点
                 markersize=marker_size,    # 点大小
                 linestyle='')

        # 绘制拟合曲线
        max_idx = np.argmax(phy_signal)
        fit_time = time_points[max_idx:]
        ax.plot(fit_time, fit_curve, 'r',linestyle = line_style, label='拟合曲线')

        # 标记最大值
        ax.axvline(time_points[max_idx], color='g', linestyle=':', label='峰值位置')
        # 标记r^2和τ
        if model_type == 'single':
            ax.text(0.05, 0.95, f' τ={lifetime:.2f}\n'
                                +r'$R^2$='
                                +f'{r_squared:.3f}',
                     transform=ax.transAxes,
                     ha='left', va='top',
                     bbox=dict(facecolor='white', alpha=0.8))
        elif model_type == 'double':
            ax.text(0.05, 0.95, f' τ1={lifetime[0]:.2f}\n'
                    + f' τ2={lifetime[1]:.2f}\n'
                    + r'$R^2$='
                    + f'{r_squared:.3f}',
                    transform=ax.transAxes,
                    ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.8))

        ax.set_xlabel('时间/ps')
        ax.set_ylabel('信号强度')
        ax.set_title('载流子寿命曲线')
        ax.legend()
        ax.grid(show_grid)

        self.canvas.draw()
        self.current_data = pd.DataFrame({
                                'time': pd.Series(time_points),
                                'signal': pd.Series(phy_signal),
                                'fit_time': pd.Series(fit_time),
                                'fit_curve':pd.Series(fit_curve)
                            })

    def display_roi_series(self, positions, intensities, title=""):
        """绘制向量ROI信号强度曲线"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        # line_style = self.plot_settings['line_style']
        # line_width = self.plot_settings['line_width']
        marker_style = self.plot_settings['marker_style']
        marker_size = self.plot_settings['marker_size']
        color = self.plot_settings['color']

        # 绘制曲线
        # ax.plot(positions, intensities,
        #         markeredgecolor=color,
        #         marker=marker_style,       # 方形点
        #         markersize=marker_size,    # 点大小
        #         linewidth=line_width,
        #         label='信号强度')
        ax.scatter(positions, intensities,
                c=color,
                marker=marker_style,       # 方形点
                label='采样点')

        # 设置图表属性
        ax.set_title(title)
        ax.set_xlabel("位置 (μm)")
        ax.set_ylabel("对比度")
        # ax.grid(True)
        ax.legend()

        self.canvas.draw()

        self.current_data = pd.DataFrame({
                                        'time': pd.Series(positions),
                                        'signal': pd.Series(intensities),
                                    })

    def display_diffusion_coefficient(self, frame_data_dict):
        """绘制多帧信号及高斯拟合,方差随时间变化图并计算扩散系数"""
        if self.fit_results is None or self.fit_results.shape[1] < 2:
            logging.warning('缺数据或数据有问题无法绘图')
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        for i, (frame_idx, data) in enumerate(frame_data_dict['signal'].items()):
            positions = data[:, 0]
            intensities = data[:, 1]

            # 绘制原始数据
            ax.scatter(positions, intensities, label=f'帧 {frame_idx} 数据点')

        for series in frame_data_dict['fitting']:
            positions = series[0]
            fitting_curve = series[1]
            ax.plot(positions, fitting_curve, '--',
                    label=f'帧拟合')

        # 设置图表属性
        ax.set_title("多帧ROI信号强度及高斯拟合")
        ax.set_xlabel("位置 (μm)")
        ax.set_ylabel("信号强度 (a.u.)")
        ax.grid(True)
        ax.legend()

        self.canvas.draw()

    def plot_variance_evolution(self):
        """绘制方差随时间变化图并计算扩散系数"""
        if self.fit_results is None or self.fit_results.shape[1] < 2:
            logging.warning('缺数据或数据有问题无法绘图')
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        times = self.fit_results[0]
        variances = self.fit_results[1]

        # 绘制数据点
        ax.scatter(times, variances, c='r', s=50, label='方差数据')

        # 线性拟合
        slope, intercept = np.polyfit(times, variances, 1)
        fit_line = slope * times + intercept
        ax.plot(times, fit_line, 'b--',
                label=f'线性拟合 (D={slope / 2:.2e})')

        # 设置图表属性
        ax.set_title("高斯方差随时间演化")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("方差 (μm²)")
        ax.grid(True)
        ax.legend()

        self.canvas.draw()
        return slope / 2  # 返回扩散系数 D = slope / 2


def clear(self):
        """清除显示"""
        self.figure.clear()
        # self.canvas.draw()