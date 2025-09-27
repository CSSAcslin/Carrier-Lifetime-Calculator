import logging

import numpy as np
from PyQt5.QtGui import QPixmap, QImage, QPainter, QWheelEvent, QTransform, QIcon
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QScrollArea,
                             QFileDialog, QSlider, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QToolBar, QAction, QDockWidget, QStyle
                             )
from PyQt5.QtCore import Qt, pyqtSignal

from transient_data_processing.core.DataManager import ImagingData


class ImageDisplayWindow(QMainWindow):
    """图像显示部件 (使用QPixmap实现)"""
    add_canvas_signal = pyqtSignal()
    # mouse_position_signal = pyqtSignal(int, int, int, float)
    # mouse_clicked_signal = pyqtSignal(int, int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.display_canvas = []
        # self.display_data = []
        # self.init_ui()
        self.init_tool_bars()

    # def init_ui(self):
    #     """此处目前要包含初次数据导入后的图像显示"""
    #     self.main_widget = QDockWidget('',self)
    #     self.layout = QVBoxLayout(self)


    def init_tool_bars(self):
        Canvas_bar = QToolBar('Canvas')
        add_canvas = QAction('add', self)
        add_canvas.setStatusTip("Add new canvas")
        add_canvas.triggered.connect(lambda: self.add_canvas_signal.emit())
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


    def add_canvas(self,data):
        """新增图像显示画布"""
        if len(self.display_canvas) >= 4:
            logging.warning("已达到最大显示区域数量 (4)")
            return False
        canvas_id = len(self.display_canvas)
        # self.display_data.append(data)
        data.canvas_num = canvas_id
        new_canvas = SubImageDisplayWidget(name=f'{canvas_id}-{data.source_name}',canvas_id=canvas_id,data=data)
        self.display_canvas.append(new_canvas)
        self.addDock(self.display_canvas[-1])
        # self.addDockWidget(Qt.LeftDockWidgetArea, self.display_canvas[-1])

    def del_canvas(self,canvas_id = None):
        """删除画布"""
        logging.info("del_canvas test works")
        if not self.display_canvas:
            return False

        if canvas_id is None:
            canvas_id = len(self.display_canvas) - 1

        elif 0 <= canvas_id < len(self.display_canvas):
            del_canvas = self.display_canvas.pop(canvas_id)
            del_canvas.setParent(None)
            del_canvas.deleteLater()
            for dock in self.findChildren(QDockWidget):
                if hasattr(dock, 'id') and dock.id == canvas_id:
                    self.removeDockWidget(dock)  # 从布局中移除
                    dock.deleteLater()  # 安全删除对象
            # 更新剩余区域的ID
            for idx, canvas in enumerate(self.display_canvas):
                canvas.canvas_id = idx

            return True
        elif canvas_id == -1: # 全部清除
            for idx in range(len(self.display_canvas)):
                del_canvas = self.display_canvas.pop(idx)
                del_canvas.setParent(None)
                del_canvas.deleteLater()
                for dock in self.findChildren(QDockWidget):
                    if hasattr(dock, 'id') and dock.id == canvas_id:
                        self.removeDockWidget(dock)  # 从布局中移除
                        dock.deleteLater()  # 安全删除对象
        return False

    def addDock(self, dock):
        """根据区域数量更新布局"""
        # 获取所有DockWidget并按id排序

        dock_count = len(self.display_canvas)

            # 根据目标DockWidget数量重新布局
        if dock_count == 1:
            self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        elif dock_count == 2:
            self.addDockWidget(Qt.RightDockWidgetArea, dock)

        elif dock_count == 3:
            # 创建左侧区域
            self.addDockWidget(Qt.LeftDockWidgetArea, dock)
            # # 垂直分割左侧区域
            # self.splitDockWidget(target_docks[0], target_docks[1], Qt.Vertical)
            # # 添加右侧区域
            # self.addDockWidget(Qt.RightDockWidgetArea, target_docks[2])

        elif dock_count >= 4:
            # 只处理前4个
            # docks = target_docks[:4]
            # # 创建左侧区域
            # self.addDockWidget(Qt.LeftDockWidgetArea, docks[0])
            # # 垂直分割左侧区域
            # self.splitDockWidget(docks[0], docks[1], Qt.Vertical)
            # 添加右侧区域
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
            # # 垂直分割右侧区域
            # self.splitDockWidget(docks[2], docks[3], Qt.Vertical)

    # def update_time_slice(self,idx,first_create = False):
        '''新的时间序列显示更新'''
        # max_frame = self.display_data[0].totalframes
        # if not 0 <= idx < max_frame:
        #     raise ValueError('idx out of range(impossible Fault)')
        # label_text = f"时间点: {idx}/{max_frame - 1}"
        # for i, data in enumerate(self.display_data):
        #     self.display_canvas[i].current_time_idx = idx # 目前是否时序数据没有在序号上做区分
        #     if data.ROI_applied: # 判断是否加了ROI
        #         if data.is_temporary : # 判断是否是时序数据
        #             self.display_canvas[i].current_image = data.image_ROI[idx]
        #             if data.totalframes == max_frame: # 判断时序是否与主窗口尺度一致
        #                 if first_create: # 判断是否为第一次创建
        #                     self.display_canvas[i].display_image(data.image_ROI[idx], idx)
        #                 else:
        #                     self.display_canvas[i].update_display_idx(data.image_ROI[idx], idx)
        #                     self._handle_hover(t=idx)
        #             else:
        #                 synco = data.totalframes // max_frame
        #                 self.display_canvas[i].update_display_idx(data.image_ROI[int(idx*synco)], idx)
        #         else:
        #             self.display_canvas[i].update_display_idx(data.image_ROI)
        #     else:
        #         if data.is_temporary :
        #             self.display_canvas[i].current_image = data.image_data[idx]
        #             if data.totalframes == max_frame:
        #                 if first_create:
        #                     self.display_canvas[i].display_image(data.image_data[idx], idx)
        #                 else:
        #                     self.display_canvas[i].update_display_idx(data.image_data[idx], idx)
        #             else:
        #                 synco = data.totalframes // max_frame
        #                 self.display_canvas[i].update_display_idx(data.image_data[int(idx*synco)], idx)
        #         else:
        #             self.display_canvas[i].update_display_idx(data.image_data)
        # # self.parent().time_label.setText(label_text)
        # # super().time_label.setText(label_text)
        # return label_text

class SubImageDisplayWidget(QDockWidget):
    """图像显示部件 (使用QPixmap实现)"""
    mouse_position_signal = pyqtSignal(int, int, int, float,float)
    mouse_clicked_signal = pyqtSignal(int, int)
    def __init__(self, parent=None,canvas_id = None,name = None, data :ImagingData = None):
        super().__init__(name, parent)
        self.id = canvas_id
        self.data = data
        self.mouse_pos = None
        self.current_time_idx = 0
        self.max_time_idx = self.data.totalframes if self.data.is_temporary else 0
        self.drag_start_pos = None  # 拖动起始位置
        self.last_scale = None
        self.initial_scale = None
        self.min_scale = 0.1
        self.max_scale = 30.0
        self.draw_layer_opacity = 0.8
        # self.setMinimumSize(300,300)
        self.init_ui()
        self.map_view = False


    def init_ui(self):
        widget = QWidget(self)
        layout = QVBoxLayout(self)

        # 创建图形视图和场景
        self.graphics_view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        # self.graphics_view.setMinimumSize(350, 350)
        self.data_layer = QGraphicsPixmapItem()
        self.draw_layer = QGraphicsPixmapItem()
        self.scene.addItem(self.data_layer)
        # self.scene.addItem(self.draw_layer)

        layout.addWidget(self.graphics_view)

        # 视图设置
        self.graphics_view.setMouseTracking(True)
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

        slider_layout = QHBoxLayout()
        self.start_button = QPushButton()
        self.start_button.setIcon(QIcon(QApplication.style().standardIcon(QStyle.SP_MediaPlay)))
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(self.max_time_idx)
        self.time_label = QLabel(f"{self.current_time_idx}/{self.max_time_idx}")
        slider_layout.addWidget(self.start_button)
        slider_layout.addWidget(self.time_slider)
        slider_layout.addWidget(self.time_label)
        if self.data.is_temporary :
            layout.addLayout(slider_layout)
            self.time_slider.valueChanged.connect(self.update_time_slice)
        widget.setLayout(layout)
        self.setWidget(widget)

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
        if not self.map_view:
            self.display_image()
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
            if self.data.is_temporary :
                original_value = self.data.image_backup[self.current_time_idx][self.y_img, self.x_img]
            else:
                original_value = self.data.image_backup[self.y_img, self.x_img]
            # 发射信号(需要主窗口连接此信号)
            self.mouse_position_signal.emit(self.x_img, self.y_img, self.current_time_idx, value,original_value)

        super().mouseMoveEvent(event)

    def mouse_release_event(self, event):
        """鼠标释放事件（结束拖动）"""
        if event.button() == Qt.MidButton:
            self.drag_start_pos = None
            self.graphics_view.setCursor(Qt.ArrowCursor)
        super(QGraphicsView, self.graphics_view).mouseReleaseEvent(event)

    def update_time_slice(self,idx=0):
        if not 0 <= idx < self.max_time_idx:
            raise ValueError('idx out of range(impossible Fault)')
        self.current_time_idx = idx
        if self.data.ROI_applied:
            self.update_display_idx(self.data.image_ROI[idx])
        if not self.data.ROI_applied:
            self.update_display_idx(self.data.image_data[idx])

    def display_image(self):
        """显示图像数据 (使用QPixmap)并记录当前时间索引"""
        # 清除现有内容

        if self.data.is_temporary and self.data.ROI_applied:
            image_data = self.data.image_ROI[0]
        elif self.data.is_temporary and not self.data.ROI_applied:
            image_data = self.data.image_data[0]
        elif not self.data.is_temporary and self.data.ROI_applied:
            image_data = self.data.image_ROI
        elif not self.data.is_temporary and not self.data.ROI_applied:
            image_data = self.data.image_data
        else:
            raise ValueError('nodata(impossible Fault)')

        self.graphics_view.resize(self.width(), self.height())
        self.scene.clear()
        self.current_time_idx = 0
        # 创建QImage并转换为QPixmap
        qimage = QImage(image_data, image_data.shape[1], image_data.shape[0],
                        image_data.shape[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        # 显示图像
        self.current_image = image_data
        self.data_layer = self.scene.addPixmap(pixmap)
        self.graphics_view.resetTransform()
        self.graphics_view.fitInView(self.data_layer, Qt.KeepAspectRatio)
        self.last_scale = self.graphics_view.transform().m11()
        self.initial_scale = self.graphics_view.transform().m11()
        self.map_view = True

        # self.graphics_view.resetTransform()
        # self.graphics_view.fitInView(self.data_layer, Qt.KeepAspectRatio) # 实现适合尺寸的放大
        # # 绘制层
        # # height, width = image_data.shape
        # # self.top_pixmap = QPixmap(width, height)
        # # self.top_pixmap.fill(Qt.transparent)
        # # self.draw_layer= self.scene.addPixmap(self.top_pixmap)
        # # self.draw_layer.setOpacity(self.draw_layer_opacity)
        # # 获取缩放因子
        # self.last_scale = self.graphics_view.transform().m11()  # 实时更新
        # self.initial_scale = self.graphics_view.transform().m11()  # 只获取一次

    def update_display_idx(self, image_data):
        """仅更新图像数据，不改变视图状态"""
        self.current_image = image_data

        # 归一化到0-255
        qimage = QImage(image_data, image_data.shape[1], image_data.shape[0],
                        image_data.shape[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        # 直接更新现有pixmap，避免重置场景，其它层保持不变
        self.data_layer.setPixmap(pixmap)

    def reset_view(self):
        """手动重置视图（缩放和平移）"""
        self.graphics_view.resetTransform()
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.last_scale = self.graphics_view.transform().m11()