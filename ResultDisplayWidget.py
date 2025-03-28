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

class ResultDisplayWidget(QWidget):
    """结果热图显示部件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.font1 = plt.font_manager.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
        self.init_ui()
        self.current_mode = "heatmap"# 或 "curve"


    def init_ui(self):
        self.layout = QVBoxLayout(self)

        # 创建matplotlib图形
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("载流子寿命热图",fontproperties=self.font1)
        self.ax.axis('off')

        self.layout.addWidget(self.canvas)

    def display_heatmap(self, lifetime_map):
        """显示寿命热图"""
        self.current_mode = "heatmap"
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # 显示热图
        im = self.ax.imshow(lifetime_map, cmap='jet')
        self.figure.colorbar(im, ax=self.ax, label='lifetime')
        self.ax.set_title("载流子寿命热图",fontproperties=self.font1)
        self.ax.axis('off')
        self.canvas.draw()

        # 保存当前数据
        self.current_data = lifetime_map

    def display_analysis_curve(self, avg_curve, lifetime, fit_curve):
        """显示分析曲线"""
        self.current_mode = "curve"
        # 已在MainWindow中实现具体绘制

    def clear(self):
        """清除显示"""
        self.figure.clear()
        if self.current_mode == "heatmap":
            ax = self.figure.add_subplot(111)
            ax.set_title("载流子寿命热图")
        else:
            ax = self.figure.add_subplot(111)
            ax.set_title("区域分析结果")
        ax.axis('off')
        self.canvas.draw()