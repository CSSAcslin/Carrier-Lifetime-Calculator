import logging

import numpy as np
from PyQt5.QtGui import QPixmap, QImage, QPainter, QWheelEvent, QTransform, QIcon
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QScrollArea,
                             QFileDialog, QSlider, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QToolBar, QAction, QDockWidget
                             )
from PyQt5.QtCore import Qt, pyqtSignal


class ImageDisplayWidget(QMainWindow):
    """图像显示部件 (使用QPixmap实现)"""
    mouse_position_signal = pyqtSignal(int, int, int, float)
    mouse_clicked_signal = pyqtSignal(int, int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mouse_pos = None
        self.current_time_idx = 0
        self.drag_start_pos = None  # 拖动起始位置
        self.last_scale = None
        self.initial_scale = None
        self.min_scale = 0.1
        self.max_scale = 15.0
        self.draw_layer_opacity = 0.8
        self.init_ui()
        self.init_tool_bars()

    def init_ui(self):
        self.main_widget = QDockWidget('',self)
        self.layout = QVBoxLayout(self)

        # 创建图形视图和场景
        self.graphics_view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        self.data_layer = QGraphicsPixmapItem()
        self.draw_layer = QGraphicsPixmapItem()
        self.scene.addItem(self.data_layer)
        self.scene.addItem(self.draw_layer)

        self.layout.addWidget(self.graphics_view)

        # 视图设置
        self.graphics_view.setMouseTracking(True)
        self.graphics_view.viewport().setMouseTracking(True)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        self.graphics_view.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.graphics_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.graphics_view.setResizeAnchor(QGraphicsView.NoAnchor)
        self.graphics_view.setTransform(QTransform())
        # 鼠标功能响应
        self.graphics_view.wheelEvent = self.wheel_event  # 滚轮缩放
        self.graphics_view.mouseMoveEvent = self.mouse_move_event
        self.graphics_view.mousePressEvent = self.mouse_press_event
        self.graphics_view.mouseReleaseEvent = self.mouse_release_event

        self.graph_widget = QWidget(self)
        self.graph_widget.setLayout(self.layout)
        self.main_widget.setWidget(self.graph_widget)
        # self.main_widget.setFeatures(QDockWidget::NoDockWidgetFeatures)
        self.main_widget.setTitleBarWidget(QWidget())
        self.addDockWidget(Qt.LeftDockWidgetArea,self.main_widget)

    def init_tool_bars(self):
        Canvas_bar = QToolBar('Canvas')
        add_canvas = QAction('add', self)
        add_canvas.setStatusTip("Add new canvas")
        add_canvas.triggered.connect(self.add_canvas)
        Canvas_bar.addAction(add_canvas)

        del_canvas = QAction('del', self)
        del_canvas.setStatusTip("Delete the latest canvas")
        del_canvas.triggered.connect(self.del_canvas)
        Canvas_bar.addAction(del_canvas)

        cursor = QAction('cursor', self)
        cursor.setStatusTip("cursor")
        # cursor.triggered.connect(self.cursor)
        Canvas_bar.addAction(cursor)

        self.addToolBar(Canvas_bar)

        Drawing_bar = QToolBar('Drawing')
        draw_pen = QAction('Pen', self)
        draw_pen.setStatusTip("Draw the pen")
        # draw_pen.triggered.connect(self.draw_pen)
        Drawing_bar.addAction(draw_pen)

        draw_line = QAction('Line', self)
        draw_line.setStatusTip("Draw the line")
        # draw_line.triggered.connect(self.draw_line)
        Drawing_bar.addAction(draw_line)

        draw_rect = QAction('Rect', self)
        draw_rect.setStatusTip("Draw the rect")
        draw_rect.setToolTip("Draw the rect")
        # draw_rect.triggered.connect(self.draw_rect)
        Drawing_bar.addAction(draw_rect)
        Drawing_bar.addSeparator()

        draw_ellipse = QAction('Ellipse', self)
        draw_ellipse.setStatusTip("Draw the ellipse")
        draw_ellipse.setToolTip("Draw the ellipse")
        # draw_ellipse.triggered.connect(self.draw_ellipse)
        Drawing_bar.addAction(draw_ellipse)

        draw_eraser = QAction('Eraser', self)
        draw_eraser.setStatusTip("Draw the eraser")
        draw_eraser.setToolTip("Draw the eraser")
        # draw_eraser.triggered.connect(self.draw_eraser)
        Drawing_bar.addAction(draw_eraser)

        draw_fill = QAction('Fill', self)
        draw_fill.setStatusTip("Draw the fill")
        draw_fill.setToolTip("Draw the fill")
        # draw_fill.triggered.connect(self.draw_fill)
        Drawing_bar.addAction(draw_fill)

        self.addToolBar(Drawing_bar)


    def add_canvas(self):
        logging.info("add_canvas test works")

    def del_canvas(self):
        logging.info("del_canvas test works")

    # def add_display_region(self, data=None, is_temporal=False, sync_with_primary=False):
    #     """添加一个新的显示区域"""
    #     # if len(self.display_regions) >= 4:
    #     #     logging.warning("已达到最大显示区域数量 (4)")
    #     #     return False
    #
    #     region_id = len(self.display_regions)
    #     # new_region = ImageDisplayWidget(region_id)
    #
    #     # 设置数据和同步属性
    #     if data is not None:
    #         new_region.set_data(data, is_temporal, sync_with_primary)
    #
    #     self.display_regions.append(new_region)
    #     self._update_layout()
    #     return True
    #
    # def remove_display_region(self, region_id=None):
    #     """移除指定显示区域（默认移除最后一个）"""
    #     if not self.display_regions:
    #         return False
    #
    #     if region_id is None:
    #         region_id = len(self.display_regions) - 1
    #
    #     if 0 <= region_id < len(self.display_regions):
    #         region = self.display_regions.pop(region_id)
    #         region.setParent(None)
    #         region.deleteLater()
    #
    #         # 更新剩余区域的ID
    #         for idx, region in enumerate(self.display_regions):
    #             region.region_id = idx
    #
    #         # 重新布局
    #         self._update_layout()
    #         return True
    #     return False
    #
    # def _update_layout(self):
    #     """根据区域数量更新布局"""
    #     # 清除当前布局
    #     for i in reversed(range(self.layout.count())):
    #         widget = self.layout.itemAt(i).widget()
    #         if widget:
    #             widget.setParent(None)
    #
    #     # 根据区域数量设置布局
    #     num_regions = len(self.display_regions)
    #
    #     if num_regions == 1:
    #         self.layout.addWidget(self.display_regions[0], 0, 0)
    #     elif num_regions == 2:
    #         self.layout.addWidget(self.display_regions[0], 0, 0)
    #         self.layout.addWidget(self.display_regions[1], 0, 1)
    #     elif num_regions >= 3:
    #         self.layout.addWidget(self.display_regions[0], 0, 0)
    #         self.layout.addWidget(self.display_regions[1], 0, 1)
    #         self.layout.addWidget(self.display_regions[2], 1, 0)
    #
    #         if num_regions == 4:
    #             self.layout.addWidget(self.display_regions[3], 1, 1)

    def wheel_event(self, event: QWheelEvent):
        """滚轮缩放实现"""
        if not hasattr(self, 'current_image'):
            return

        zoom_step  = 1.25
        if event.angleDelta().y() > 0:  # 放大
            new_scale = min(self.last_scale * zoom_step, self.max_scale)
        else:  # 缩小
            new_scale = max(self.last_scale / zoom_step, self.min_scale)

        if new_scale == self.last_scale:
            return

        # 获取鼠标在视图和场景中的位置
        mouse_view_pos = event.pos()
        old_scene_pos = self.graphics_view.mapToScene(mouse_view_pos)

        # 应用新缩放因子
        self.graphics_view.setTransform(QTransform.fromScale(new_scale, new_scale))
        self.last_scale = new_scale  # 必须更新缩放因子记录

        # 仅在放大时调整视口位置
        if new_scale > self.initial_scale:
            # 计算缩放后的鼠标位置差
            new_view_pos = self.graphics_view.mapFromScene(old_scene_pos)
            delta = new_view_pos - mouse_view_pos
            # 通过滚动条补偿位置变化
            self.graphics_view.horizontalScrollBar().setValue(
                self.graphics_view.horizontalScrollBar().value() + delta.x()
            )
            self.graphics_view.verticalScrollBar().setValue(
                self.graphics_view.verticalScrollBar().value() + delta.y()
            )

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
        self.x_img, self.y_img = int(pos.x()), int(pos.y())

        # 检查坐标是否在图像范围内
        h, w = self.current_image.shape
        if 0 <= self.x_img < w and 0 <= self.y_img < h:
            self.mouse_pos = (self.x_img, self.y_img)
            value = self.current_image[self.y_img, self.x_img]
            # 发射信号(需要主窗口连接此信号)
            self.mouse_position_signal.emit(self.x_img, self.y_img, self.current_time_idx, value)

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
        self.data_layer = self.scene.addPixmap(pixmap)

        self.graphics_view.resetTransform()
        self.graphics_view.fitInView(self.data_layer, Qt.KeepAspectRatio) # 实现适合尺寸的放大
        # 绘制层
        height, width = image_to_show.shape
        self.top_pixmap = QPixmap(width, height)
        self.top_pixmap.fill(Qt.transparent)
        self.draw_layer= self.scene.addPixmap(self.top_pixmap)
        self.draw_layer.setOpacity(self.draw_layer_opacity)
        # 获取缩放因子
        self.last_scale = self.graphics_view.transform().m11()  # 实时更新
        self.initial_scale = self.graphics_view.transform().m11()  # 只获取一次

    def update_display_idx(self, image_data, time_idx=1):
        """仅更新图像数据，不改变视图状态"""
        self.current_time_idx = time_idx
        self.current_image = image_data

        # 归一化到0-255
        image_to_show = (image_data * 255).astype(np.uint8)
        qimage = QImage(image_to_show.data, image_to_show.shape[1], image_to_show.shape[0],
                        image_to_show.shape[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        # 直接更新现有pixmap，避免重置场景，其它层保持不变
        self.data_layer.setPixmap(pixmap)

    def reset_view(self):
        """手动重置视图（缩放和平移）"""
        self.graphics_view.resetTransform()
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.last_scale = self.graphics_view.transform().m11()
    def clear(self):
        """清除显示"""
        self.scene.clear()