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
        self.current_mode = "heatmap"# 或 "curve"

        self.font_list = fm.findSystemFonts(fontpaths=r"C:\Windows\Fonts" , fontext='ttf')
        self.chinese_fonts = [f for f in self.font_list if
                         any(c in f.lower() for c in ['simhei', 'simsun', 'microsoft yahei', 'fang'])]

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
        self.ax = self.figure.add_subplot(111)
        self.ax.axis('off')

        self.layout.addWidget(self.canvas)

    def display_distribution_map(self, lifetime_map):
        """显示寿命热图"""
        self.current_mode = "heatmap"
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # 显示热图
        im = ax.imshow(lifetime_map, cmap='jet')
        self.figure.colorbar(im, ax=self.ax, label='lifetime')
        ax.set_title("载流子寿命分布图")
        ax.axis('off')
        self.canvas.draw()

        # 保存当前数据
        self.current_data = pd.DataFrame(lifetime_map)

    def display_lifetime_curve(self, phy_signal, lifetime, r_squared, fit_curve,time_points,boundary):
        """显示区域分析结果"""
        # 使用原来的结果显示区域
        self.current_mode = "curve"
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        max_bound = boundary['max']
        min_bound = 0
        ax.set_ylim(min_bound, max_bound)

        # 绘制原始曲线
        # time_points = time_points - time_points[0]  # 从0开始
        ax.plot(time_points, phy_signal,
                markeredgecolor='blue', # 点边缘色
                markeredgewidth=2,
                label='原始数据',
                 marker='s',       # 方形点
                 markersize=6,    # 点大小
                 linestyle='')

        # 绘制拟合曲线
        max_idx = np.argmax(phy_signal)
        fit_time = time_points[max_idx:]
        ax.plot(fit_time, fit_curve, 'r--', label='拟合曲线')

        # 标记最大值
        ax.axvline(time_points[max_idx], color='g', linestyle=':', label='峰值位置')
        # 标记r^2和τ
        ax.text(0.05, 0.95, f' τ={lifetime:.2f}\n'
                            +r'$R^2$='
                            +f'{r_squared:.3f}',
                 transform=ax.transAxes,
                 ha='left', va='top',
                 bbox=dict(facecolor='white', alpha=0.8))

        ax.set_xlabel('时间/ps')
        ax.set_ylabel('信号强度')
        ax.set_title('载流子寿命曲线')
        ax.legend()
        ax.grid(False)

        self.canvas.draw()
        self.current_data = pd.DataFrame({
                                'time': pd.Series(time_points),
                                'signal': pd.Series(phy_signal),
                                'fit_time': pd.Series(fit_time),
                                'fit_curve':pd.Series(fit_curve)
                            })


    def clear(self):
        """清除显示"""
        self.figure.clear()
        if self.current_mode == "heatmap":
            ax = self.figure.add_subplot(111)
            ax.set_title("载流子寿命热图")
        elif self.current_mode == "curve":
            ax = self.figure.add_subplot(111)
            ax.set_title("区域分析结果")
        ax.axis('off')
        self.canvas.draw()