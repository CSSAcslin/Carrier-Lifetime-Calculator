import numpy as np
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QScrollArea,
                             QFileDialog, QSlider, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
                             )
from PyQt5.QtCore import Qt, pyqtSignal


class ImageDisplayWidget(QWidget):
    """图像显示部件 (使用QPixmap实现)"""
    mouse_position_signal = pyqtSignal(int, int, int, float)
    mouse_clicked_signal = pyqtSignal(int, int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mouse_pos = None

        self.current_time_idx = 0
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)

        # 创建图形视图和场景
        self.graphics_view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.layout.addWidget(self.graphics_view)

        # 添加缩放控制（暂不启用）
        # self.zoom_label = QLabel("缩放倍数:")
        # self.zoom_spinbox = QSpinBox()
        # self.zoom_spinbox.setMinimum(1)
        # self.zoom_spinbox.setMaximum(10)
        # self.zoom_spinbox.setValue(1)
        # self.zoom_spinbox.valueChanged.connect(self.update_zoom)

        # zoom_layout = QHBoxLayout()
        # zoom_layout.addWidget(self.zoom_label)
        # zoom_layout.addWidget(self.zoom_spinbox)
        # zoom_layout.addStretch()

        # self.layout.addLayout(zoom_layout)

        # 启用鼠标跟踪
        self.graphics_view.setMouseTracking(True)
        self.graphics_view.viewport().setMouseTracking(True)
        self.graphics_view.mouseMoveEvent = self.mouse_move_event
        self.graphics_view.mousePressEvent = self.mouse_press_event

    def mouse_press_event(self, event):
        """鼠标点击事件处理"""
        if event.button() == Qt.LeftButton and hasattr(self, 'current_image'):
            # 获取点击位置图像坐标（不考虑缩放）
            pos = self.graphics_view.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())

            # 检查坐标有效性
            h, w = self.current_image.shape
            if 0 <= x < w and 0 <= y < h:
                self.mouse_clicked_signal.emit(x,y)

    def mouse_move_event(self, event):
        """鼠标移动事件处理"""
        if not hasattr(self, 'current_image'):
            return

        # 获取鼠标在图像上的坐标
        pos = self.graphics_view.mapToScene(event.pos())
        x, y = int(pos.x()), int(pos.y())

        # 考虑缩放因子
        zoom = 1 #self.zoom_spinbox.value()
        x_img, y_img = x // zoom, y // zoom

        # 检查坐标是否在图像范围内
        h, w = self.current_image.shape
        if 0 <= x_img < w and 0 <= y_img < h:
            self.mouse_pos = (x_img, y_img)
            value = self.current_image[y_img, x_img]
            # 发射信号(需要主窗口连接此信号)
            self.mouse_position_signal.emit(x_img, y_img, self.current_time_idx, value)

        super().mouseMoveEvent(event)

    def display_image(self, image_data,time_idx=1, zoom=1):
        """显示图像数据 (使用QPixmap)并记录当前时间索引"""
        # 清除现有内容
        self.scene.clear()
        self.current_time_idx = time_idx

        # 原始图像尺寸
        h, w = image_data.shape

        # 应用缩放 (不插值的放大)
        if zoom > 1:
            # 创建放大后的数组
            zoomed_data = np.zeros((h * zoom, w * zoom), dtype=np.uint8)

            norm_data = (image_data * 255).astype(np.uint8)

            # 每个原始像素复制zoom×zoom次
            for i in range(h):
                for j in range(w):
                    zoomed_data[i * zoom:(i + 1) * zoom, j * zoom:(j + 1) * zoom] = norm_data[i, j]

            image_to_show = zoomed_data
        else:
            # 归一化数据到0-255范围
            image_to_show =  (image_data * 255).astype(np.uint8)

        # 创建QImage并转换为QPixmap
        qimage = QImage(image_to_show.data, image_to_show.shape[1], image_to_show.shape[0],
                        image_to_show.shape[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        # 显示图像
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    # def update_zoom(self):
    #     """更新缩放级别"""
    #     if hasattr(self, 'current_image'):
    #         self.display_image(self.current_image, self.zoom_spinbox.value())

    def clear(self):
        """清除显示"""
        self.scene.clear()