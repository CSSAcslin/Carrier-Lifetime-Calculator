import numpy as np
from PyQt5.QtGui import QPixmap, QImage, QPainter, QWheelEvent, QTransform
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
        self.drag_start_pos = None  # 拖动起始位置
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

        # 视图设置
        self.graphics_view.setMouseTracking(True)
        self.graphics_view.viewport().setMouseTracking(True)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        self.graphics_view.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.graphics_view.setResizeAnchor(QGraphicsView.NoAnchor)
        # 鼠标功能响应
        self.graphics_view.wheelEvent = self.wheel_event  # 滚轮缩放
        self.graphics_view.mouseMoveEvent = self.mouse_move_event
        self.graphics_view.mousePressEvent = self.mouse_press_event
        self.graphics_view.mouseReleaseEvent = self.mouse_release_event

    def wheel_event(self, event: QWheelEvent):
        """滚轮缩放实现"""
        if not hasattr(self, 'current_image'):
            return

        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        # 获取鼠标在视图和场景中的当前位置
        mouse_view_pos = event.pos()
        mouse_scene_pos = self.graphics_view.mapToScene(mouse_view_pos)

        # 计算缩放
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

         # 获取当前变换矩阵
        current_transform = self.graphics_view.transform()
        # 应用缩放
        new_scale = current_transform.m11() * zoom_factor
        new_scale = max(0.1, min(new_scale, 10.0))  # 限制缩放范围
        self.graphics_view.setTransform(QTransform())
        self.graphics_view.scale(new_scale, new_scale)

        # 计算缩放后鼠标应处的新场景位置
        new_view_pos = self.graphics_view.transform().map(mouse_scene_pos)

        # 移动视图补偿偏移
        delta = mouse_view_pos - new_view_pos
        self.graphics_view.translate(delta.x(), delta.y())

        # 阻止事件继续传播
        event.accept()

    def mouse_press_event(self, event):
        """鼠标点击事件处理"""
        if not hasattr(self, 'current_image'):
            return
        if event.button() == Qt.MidButton :
            # 中键按下：准备拖动
            self.drag_start_pos = event.pos()
            self.graphics_view.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            # 左键点击：发射坐标信号
            pos = self.graphics_view.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())
            h, w = self.current_image.shape
            if 0 <= x < w and 0 <= y < h:
                self.mouse_clicked_signal.emit(x, y)

        super(QGraphicsView, self.graphics_view).mousePressEvent(event)

    def mouse_move_event(self, event):
        """鼠标移动事件处理"""
        if not hasattr(self, 'current_image'):
            return

        if self.drag_start_pos is not None:
            delta = event.pos() - self.drag_start_pos
            self.graphics_view.horizontalScrollBar().setValue(
                self.graphics_view.horizontalScrollBar().value() - delta.x())
            self.graphics_view.verticalScrollBar().setValue(
                self.graphics_view.verticalScrollBar().value() - delta.y())
            self.drag_start_pos = event.pos()

        # 获取鼠标在图像上的坐标
        pos = self.graphics_view.mapToScene(event.pos())
        x_img, y_img = int(pos.x()), int(pos.y())

        # 检查坐标是否在图像范围内
        h, w = self.current_image.shape
        if 0 <= x_img < w and 0 <= y_img < h:
            self.mouse_pos = (x_img, y_img)
            value = self.current_image[y_img, x_img]
            # 发射信号(需要主窗口连接此信号)
            self.mouse_position_signal.emit(x_img, y_img, self.current_time_idx, value)

        super().mouseMoveEvent(event)

    def mouse_release_event(self, event):
        """鼠标释放事件（结束拖动）"""
        if event.button() == Qt.MidButton:
            self.drag_start_pos = None
            self.graphics_view.setCursor(Qt.ArrowCursor)
        super(QGraphicsView, self.graphics_view).mouseReleaseEvent(event)

    def display_image(self, image_data,time_idx=1):
        """显示图像数据 (使用QPixmap)并记录当前时间索引"""
        # 清除现有内容
        self.scene.clear()
        self.current_time_idx = time_idx

        image_to_show =  (image_data * 255).astype(np.uint8)

        # 创建QImage并转换为QPixmap
        qimage = QImage(image_to_show.data, image_to_show.shape[1], image_to_show.shape[0],
                        image_to_show.shape[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        # 显示图像
        self.scale_factor = 1.0        # 重置视图缩放
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.graphics_view.resetTransform()
        self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio) # 实现适合尺寸的放大

    def update_display_idx(self, image_data, time_idx=1):
        """仅更新图像数据，不改变视图状态"""
        self.current_time_idx = time_idx
        self.current_image = image_data

        # 归一化到0-255
        image_to_show = (image_data * 255).astype(np.uint8)
        qimage = QImage(image_to_show.data, image_to_show.shape[1], image_to_show.shape[0],
                        image_to_show.shape[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        # 直接更新现有pixmap，避免重置场景
        self.pixmap_item.setPixmap(pixmap)

    def reset_view(self):
        """手动重置视图（缩放和平移）"""
        self.scale_factor = 1.0
        self.graphics_view.resetTransform()
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def clear(self):
        """清除显示"""
        self.scene.clear()