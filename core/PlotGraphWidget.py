import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QTabWidget, QWidget, QVBoxLayout, QApplication,
                             QPushButton, QHBoxLayout, QCheckBox, QLabel)
from PyQt5.QtCore import Qt, pyqtSignal


# 全局配置：白色背景，黑色前景色 (符合科研论文习惯)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
class PlotGraphWidget(QWidget):
    '''基于PyQtGraph实现的强大数据显示及分析控件'''

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.data_items = []

        #十字光标相关
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen='g')  # 垂直线
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen='g')  # 水平线
        self.label = pg.TextItem(anchor=(0, 1), color='k')  # 坐标显示文本
        self._crosshair_enabled = False

    def toggle_crosshair(self, enable):
        """开启/关闭鼠标跟随的十字光标"""
        self._crosshair_enabled = enable
        if enable:
            self.plot_widget.addItem(self.v_line, ignoreBounds=True)
            self.plot_widget.addItem(self.h_line, ignoreBounds=True)
            self.plot_widget.addItem(self.label, ignoreBounds=True)
            # 代理信号：鼠标移动时触发
            self.plot_widget.scene().sigMouseMoved.connect(self._update_crosshair)
        else:
            self.plot_widget.removeItem(self.v_line)
            self.plot_widget.removeItem(self.h_line)
            self.plot_widget.removeItem(self.label)
            try:
                self.plot_widget.scene().sigMouseMoved.disconnect(self._update_crosshair)
            except:
                pass

    def _update_crosshair(self, pos):
        """内部方法：更新光标位置"""
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            self.v_line.setPos(x)
            self.h_line.setPos(y)
            self.label.setText(f"X={x:.2f}, Y={y:.2f}")
            self.label.setPos(x, y)

    def plot_data(self, array:np.ndarray, **kwargs):
        """
        添加或更新数据
        :param data_id: 数据的唯一标识 (str)
        :param x, y: 数据数组
        :param kwargs: 绘图参数
               - mode: 'line' 或 'scatter'
               - color: 线条/点颜色
               - width: 线宽
               - symbol: 散点形状 ('o', 's', 't', 'd', '+')
               - name: 图例名称
        """
        # 提取样式参数
        mode = kwargs.get('mode', 'line')
        color = kwargs.get('color', '#00ccff')  # 默认青色
        width = kwargs.get('width', 2)
        name = kwargs.get('name', 'data')
        symbol = kwargs.get('symbol', 'o')
        symbol_size = kwargs.get('symbol_size', 8)

        # 构造 Pen
        pen = pg.mkPen(color=color, width=width)

        # 如果是散点模式，pen 设为 None (只画点)
        plot_pen = pen if mode == 'line' else None
        plot_symbol = symbol if mode == 'scatter' else None

        # === 创建新数据 ===
        item = self.plot_widget.plot(
            array[0],array[1],
            pen=plot_pen,
            symbol=plot_symbol,
            symbolBrush=color,
            symbolSize=symbol_size,
            name=name
        )
        self.data_items.append(item)

        # 自动添加图例（如果还没加）
        if self.plot_widget.plotItem.legend is None:
            self.plot_widget.addLegend()

    def clear_all(self):
        """清空所有绘图内容"""
        self.plot_widget.clear()
        self.data_items.clear()