import logging

import numpy as np
from PyQt5.QtGui import QPixmap, QImage, QPainter, QWheelEvent, QTransform, QIcon, QPen, QBrush, QColor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QScrollArea,
                             QFileDialog, QSlider, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QToolBar, QAction, QDockWidget, QStyle,
                             QGraphicsRectItem, QActionGroup
                             )
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QSize

from transient_data_processing.core.DataManager import ImagingData


class ImageDisplayWindow(QMainWindow):
    """图像显示管理"""
    add_canvas_signal = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.display_canvas = []
        self.init_tool_bars()
        self.cursor_id = 0

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
        self.drawing_action_group = QActionGroup(self)
        self.drawing_action_group.setExclusive(True)

        def create_drawing_action(name, tip, slot=None):
            action = QAction(name, self)
            action.setStatusTip(tip)
            action.setToolTip(tip)
            action.setCheckable(True)  # 允许选中状态
            action.triggered.connect(lambda : self.set_tools(name))
            self.drawing_action_group.addAction(action)  # 添加到互斥组
            Drawing_bar.addAction(action)
            return action

        # 创建所有绘图工具
        draw_pen = create_drawing_action('Pen', "Draw the pen")
        draw_line = create_drawing_action('Line', "Draw the line")
        draw_rect = create_drawing_action('Rect', "Draw the rect")
        draw_ellipse = create_drawing_action('Ellipse', "Draw the ellipse")

        Drawing_bar.addSeparator()

        draw_eraser = create_drawing_action('Eraser', "Draw the eraser")
        draw_fill = create_drawing_action('Fill', "Draw the fill")
        set_color = create_drawing_action('Color', "Set the color")

        Drawing_bar.addSeparator()

        vector_line = create_drawing_action('V-line', "Draw the vector line")
        vector_rect = create_drawing_action('V-rect', "Draw the vector rect")
        #
        # # 添加默认选中项（可选）
        # draw_pen.setChecked(True)

        self.addToolBar(Drawing_bar)

    def set_cursor_id(self,id):
        self.cursor_id = id

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

    def handle_dock_closed(self, top_level):
        """当 Dock 关闭时触发"""
        if not top_level:  # 当 Dock 不再是顶级窗口时（即被关闭）
            self.parent().del_canvas(self.canvas_id)

    def _remove_single_canvas(self, canvas_id):
        """删除单个画布"""
        # 从布局中移除并删除DockWidget
        for dock in self.findChildren(QDockWidget):
            if hasattr(dock, 'canvas_id') and dock.canvas_id == canvas_id:
                self.removeDockWidget(dock)
                dock.deleteLater()
                break

        # 从display_canvas列表中移除
        for i, canvas in enumerate(self.display_canvas):
            if canvas.canvas_id == canvas_id:
                del self.display_canvas[i]
                break

        return True

    def del_canvas(self,canvas_id = None):
        """删除画布
             None - 删除最后一个画布
              int - 删除指定ID的画布
               -1 - 删除所有画布
        """
        logging.info("del_canvas test works")
        if not self.display_canvas:
            return False
        # 删除最后添加的画布
        if canvas_id is None:
            canvas_id = self.display_canvas[-1].canvas_id
        # 删除所有画布
        elif canvas_id == -1: # 全部清除
            all_ids = [c.canvas_id for c in self.display_canvas]
            for cid in all_ids:
                self._remove_single_canvas(cid)
            return True

        # 删除单个canvas_id画布
        return self._remove_single_canvas(canvas_id)

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
            self.splitDockWidget(self.display_canvas[0], dock, Qt.Vertical)
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

    def set_tools(self,tool_name:str):
        self.display_canvas[self.cursor_id].set_drawing_tool(tool_name)




class SubImageDisplayWidget(QDockWidget):
    """子图像显示部件"""
    mouse_position_signal = pyqtSignal(int, int, int, float,float)
    mouse_clicked_signal = pyqtSignal(int, int)
    current_canvas_signal = pyqtSignal(int)
    draw_result_signal = pyqtSignal(str,object)
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
        self.drawing_tool = None
        self.drawing = False
        self.start_pos = None
        self.end_pos = None
        self.temp_item = None # 临时画布
        self.vector_rect_item = None # 矢量矩形绘制
        self.v_rect_roi = None # 矢量矩形蒙版结果（左上角坐标（x,y), 宽度, 高度）

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
        self.scene.addItem(self.draw_layer)

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
        self.start_button.setIconSize(QSize(16,16))
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(self.max_time_idx-1)
        self.time_label = QLabel(f"{self.current_time_idx}/{self.max_time_idx-1}")
        slider_layout.addWidget(self.start_button)
        slider_layout.addWidget(self.time_slider)
        slider_layout.addWidget(self.time_label)
        if self.data.is_temporary :
            layout.addLayout(slider_layout)
            self.time_slider.valueChanged.connect(self.update_time_slice)
        widget.setLayout(layout)
        self.setWidget(widget)

    def set_drawing_tool(self, tool):
        """设置当前绘图工具"""
        self.drawing_tool = tool
        self.drawing = False
        self.start_pos = None
        self.end_pos = None

        # 清除前序画板
        self.clear_vector_rect()
        if self.temp_item:
            self.scene.removeItem(self.temp_item)
            self.temp_item = None

    def clear_vector_rect(self):
        """清除之前绘制的矢量矩形"""
        if self.vector_rect_item:
            self.scene.removeItem(self.vector_rect_item)
            self.vector_rect_item = None
        if self.temp_item:
            self.scene.removeItem(self.temp_item)
            self.temp_item = None
        self.current_roi_rect = None

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
            if self.drawing_tool == 'V-rect':
                self.clear_vector_rect()
                self.drawing = True
                self.start_pos = self.graphics_view.mapToScene(event.pos())
                self.end_pos = self.start_pos

                # 创建临时绘图项
                if self.temp_item:
                    self.scene.removeItem(self.temp_item)
                rect = QRectF(self.start_pos, self.end_pos)
                self.temp_item = QGraphicsRectItem(rect)
                self.temp_item.setPen(QPen(Qt.yellow, 0.2, Qt.SolidLine,Qt.SquareCap ,Qt.MiterJoin))
                self.scene.addItem(self.temp_item)
            else:
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
            self.current_canvas_signal.emit(self.id)

        # 绘图模式
        if self.drawing and self.drawing_tool == 'V-rect' and self.temp_item:
            self.end_pos = self.graphics_view.mapToScene(event.pos())
            rect = QRectF(self.start_pos, self.end_pos).normalized()
            self.temp_item.setRect(rect)
        super().mouseMoveEvent(event)

    def mouse_release_event(self, event):
        """鼠标释放事件（结束拖动）"""
        if event.button() == Qt.MidButton:
            self.drag_start_pos = None
            self.graphics_view.setCursor(Qt.ArrowCursor)

        elif event.button() == Qt.LeftButton and self.drawing:
            # 完成绘图
            self.drawing = False

            if self.drawing_tool == 'V-rect' and self.temp_item:
                # 获取矩形坐标
                rect = self.temp_item.rect()
                x1 = int(rect.x())
                y1 = int(rect.y())
                width = int(rect.width())
                height = int(rect.height())

                # 确保坐标在图像范围内
                h, w = self.current_image.shape
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                width = min(width, w - x1)
                height = min(height, h - y1)

                # 创建ROI信息
                self.v_rect_roi = ((x1, y1), width, height)
                self.draw_result_signal.emit('v_rect',self.v_rect_roi)

                # 移除临时绘图项

                self.vector_rect_item = QGraphicsRectItem(QRectF(x1, y1, width, height))
                self.vector_rect_item.setPen(QPen(Qt.yellow, 1,Qt.SolidLine,Qt.SquareCap ,Qt.MiterJoin))
                self.vector_rect_item.setBrush(QBrush(QColor(255,255, 0, 128)))
                self.scene.addItem(self.vector_rect_item)

                self.scene.removeItem(self.temp_item)
                self.temp_item = None

                # 可选：在绘图层显示ROI区域
                if hasattr(self, 'top_pixmap'):
                    painter = QPainter(self.top_pixmap)
                    painter.setPen(QPen(Qt.red, 1,))
                    painter.drawRect(x1, y1, width, height)
                    painter.end()
                    self.draw_layer.setPixmap(self.top_pixmap)

        super(QGraphicsView, self.graphics_view).mouseReleaseEvent(event)

    def closeEvent(self, event):
        """重写关闭事件"""
        if self.parent() and hasattr(self.parent(), 'remove_canvas'):
            self.parent().remove_canvas(self.canvas_id)
        super().closeEvent(event)

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
        # 绘制层
        height, width = image_data.shape
        self.top_pixmap = QPixmap(width, height)
        self.top_pixmap.fill(Qt.transparent)
        self.draw_layer = self.scene.addPixmap(self.top_pixmap)
        self.draw_layer.setOpacity(self.draw_layer_opacity)
        self.draw_layer.setZValue(1)  # 确保绘图层在数据层之上


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