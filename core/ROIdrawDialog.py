import sys
import numpy as np
from math import atan2, pi, cos, sin
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QToolBar, QToolButton,
                             QLabel, QSlider, QColorDialog, QGraphicsView, QGraphicsScene,
                             QGraphicsPixmapItem, QButtonGroup, QSizePolicy, QDialogButtonBox, QWidget, QApplication)
from PyQt5.QtCore import Qt, QPoint, QRectF, QSize, QPointF
from PyQt5.QtGui import (QPixmap, QPainter, QPen, QBrush, QImage, QColor,
                         qRgb, qRed, qGreen, qBlue, QPainterPath)


class ROIdrawDialog(QDialog):
    def __init__(self, base_layer_array=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("双图层绘图工具")

        # 初始化变量
        self.drawing = False
        self.last_point = QPoint()
        self.current_tool = "pen"
        self.pen_size = 1
        self.pen_color = Qt.black
        self.fill_color = Qt.black
        self.top_layer_opacity = 0.8
        self.bottom_layer_opacity = 1.0
        self.angle_step = pi / 4

        # 如果没有提供基础图层，创建一个默认的
        if base_layer_array is None:
            base_layer_array = np.zeros((400, 600), dtype=np.uint8)

        self.base_layer_array = base_layer_array
        self.top_layer_array = np.zeros_like(base_layer_array, dtype=np.uint8)

        # 初始化UI
        self.init_ui()

        # 更新画布大小
        self.update_canvas_size()

    def init_ui(self):
        # 顶部工具栏
        self.top_toolbar = QToolBar("顶部工具栏")
        self.top_toolbar.setFloatable(True)
        self.top_toolbar.setMovable(True)
        self.init_top_toolbar()

        # 左侧工具栏
        self.left_toolbar = QToolBar("左侧工具栏")
        self.left_toolbar.setFloatable(True)
        self.left_toolbar.setMovable(True)
        self.left_toolbar.setOrientation(Qt.Vertical)
        self.init_left_toolbar()

        # 画布区域
        self.init_graphics_view()

        # 创建主窗口布局
        main_widget = QWidget()
        main_widget.setLayout(QVBoxLayout())

        # 添加工具栏和画布
        main_widget.layout().addWidget(self.top_toolbar)
        main_widget.layout().addWidget(self.graphics_view)

        # 底部确认取消按钮区域
        self.bottom_button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self
        )
        self.bottom_button_box.accepted.connect(self.accept)
        self.bottom_button_box.rejected.connect(self.reject)

        main_widget.layout().addWidget(self.bottom_button_box)

        # 设置主布局
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.left_toolbar)
        self.layout().addWidget(main_widget, 1)

    def init_graphics_view(self):
        """初始化图形视图和场景"""
        self.graphics_view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)

        # 设置抗锯齿和优化渲染
        self.graphics_view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.graphics_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        # 创建图层的图形项
        self.bottom_layer_item = QGraphicsPixmapItem()
        self.top_layer_item = QGraphicsPixmapItem()

        # 添加到场景
        self.scene.addItem(self.bottom_layer_item)
        self.scene.addItem(self.top_layer_item)

        # 设置鼠标事件
        self.graphics_view.setMouseTracking(True)
        self.graphics_view.viewport().installEventFilter(self)

    def init_top_toolbar(self):
        # 画笔大小
        pen_size_label = QLabel("画笔大小:")
        self.top_toolbar.addWidget(pen_size_label)

        self.pen_size_slider = QSlider(Qt.Horizontal)
        self.pen_size_slider.setRange(1, 20)
        self.pen_size_slider.setValue(self.pen_size)
        self.pen_size_slider.valueChanged.connect(self.set_pen_size)
        self.top_toolbar.addWidget(self.pen_size_slider)
        self.pen_size_value_label = QLabel(str(self.pen_size))
        self.top_toolbar.addWidget(self.pen_size_value_label)

        # 透明度调节 (顶部图层)
        top_opacity_label = QLabel("顶部透明度:")
        self.top_toolbar.addWidget(top_opacity_label)

        self.top_opacity_slider = QSlider(Qt.Horizontal)
        self.top_opacity_slider.setRange(0, 100)
        self.top_opacity_slider.setValue(int(self.top_layer_opacity * 100))
        self.top_opacity_slider.valueChanged.connect(self.set_top_layer_opacity)
        self.top_toolbar.addWidget(self.top_opacity_slider)
        self.top_opacity_value_label = QLabel(f"{self.top_layer_opacity:.1f}")
        self.top_toolbar.addWidget(self.top_opacity_value_label)

        # 透明度调节 (底部图层)
        bottom_opacity_label = QLabel("底部透明度:")
        self.top_toolbar.addWidget(bottom_opacity_label)

        self.bottom_opacity_slider = QSlider(Qt.Horizontal)
        self.bottom_opacity_slider.setRange(0, 100)
        self.bottom_opacity_slider.setValue(int(self.bottom_layer_opacity * 100))
        self.bottom_opacity_slider.valueChanged.connect(self.set_bottom_layer_opacity)
        self.top_toolbar.addWidget(self.bottom_opacity_slider)
        self.bottom_opacity_value_label = QLabel(f"{self.bottom_layer_opacity:.1f}")
        self.top_toolbar.addWidget(self.bottom_opacity_value_label)

        # 清除画板
        clear_btn = QToolButton()
        clear_btn.setText("清除顶部图层")
        clear_btn.clicked.connect(self.clear_top_layer)
        self.top_toolbar.addWidget(clear_btn)

    def init_left_toolbar(self):
        # 创建按钮组确保单选
        self.tool_button_group = QButtonGroup(self)

        # 画笔工具
        pen_btn = QToolButton()
        pen_btn.setText("画笔")
        pen_btn.setCheckable(True)
        pen_btn.setChecked(True)
        pen_btn.clicked.connect(lambda: self.set_tool("pen"))
        self.left_toolbar.addWidget(pen_btn)
        self.tool_button_group.addButton(pen_btn)

        # 直线工具
        line_btn = QToolButton()
        line_btn.setText("直线")
        line_btn.setCheckable(True)
        line_btn.clicked.connect(lambda: self.set_tool("line"))
        self.left_toolbar.addWidget(line_btn)
        self.tool_button_group.addButton(line_btn)

        # 矩形工具
        rect_btn = QToolButton()
        rect_btn.setText("矩形")
        rect_btn.setCheckable(True)
        rect_btn.clicked.connect(lambda: self.set_tool("rect"))
        self.left_toolbar.addWidget(rect_btn)
        self.tool_button_group.addButton(rect_btn)

        # 椭圆工具
        ellipse_btn = QToolButton()
        ellipse_btn.setText("椭圆")
        ellipse_btn.setCheckable(True)
        ellipse_btn.clicked.connect(lambda: self.set_tool("ellipse"))
        self.left_toolbar.addWidget(ellipse_btn)
        self.tool_button_group.addButton(ellipse_btn)

        # 填充工具
        fill_btn = QToolButton()
        fill_btn.setText("填充")
        fill_btn.setCheckable(True)
        fill_btn.clicked.connect(lambda: self.set_tool("fill"))
        self.left_toolbar.addWidget(fill_btn)
        self.tool_button_group.addButton(fill_btn)

        # 橡皮擦工具
        eraser_btn = QToolButton()
        eraser_btn.setText("橡皮擦")
        eraser_btn.setCheckable(True)
        eraser_btn.clicked.connect(lambda: self.set_tool("eraser"))
        self.left_toolbar.addWidget(eraser_btn)
        self.tool_button_group.addButton(eraser_btn)

        # 颜色选择
        color_btn = QToolButton()
        color_btn.setText("选择颜色")
        color_btn.clicked.connect(self.choose_color)
        self.left_toolbar.addWidget(color_btn)

        # 填充颜色选择
        fill_color_btn = QToolButton()
        fill_color_btn.setText("选择填充色")
        fill_color_btn.clicked.connect(self.choose_fill_color)
        self.left_toolbar.addWidget(fill_color_btn)

        # 添加分隔符
        self.left_toolbar.addSeparator()
        self.left_toolbar.setContentsMargins(0,2,0,2)

    def update_canvas_size(self):
        # 根据基础图层数组大小创建画布
        height, width = self.base_layer_array.shape

        # 创建底部图层QPixmap
        self.bottom_pixmap = self.array_to_pixmap(self.base_layer_array)
        self.bottom_layer_item.setPixmap(self.bottom_pixmap)
        self.bottom_layer_item.setOpacity(self.bottom_layer_opacity)

        # 创建顶部图层QPixmap (透明)
        self.top_pixmap = QPixmap(width, height)
        self.top_pixmap.fill(Qt.transparent)
        self.top_layer_item.setPixmap(self.top_pixmap)
        self.top_layer_item.setOpacity(self.top_layer_opacity)

        # 设置场景大小
        self.scene.setSceneRect(0, 0, width, height)

        # 自适应视图
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def array_to_pixmap(self, array):
        """将二维数组转换为QPixmap"""
        height, width = array.shape
        image = QImage(width, height, QImage.Format_ARGB32)
        array_to_show = (array * 255).astype(np.uint8)
        image = QImage(array_to_show.data, array_to_show.shape[1], array_to_show.shape[0],
                       array_to_show.shape[1], QImage.Format_Grayscale8)

        return QPixmap.fromImage(image)

    def set_tool(self, tool):
        self.current_tool = tool

    def set_pen_size(self, size):
        self.pen_size = size
        self.pen_size_value_label.setText(str(size))

    def set_top_layer_opacity(self, value):
        self.top_layer_opacity = value / 100.0
        self.top_layer_item.setOpacity(self.top_layer_opacity)
        self.top_opacity_value_label.setText(f"{self.top_layer_opacity:.1f}")

    def set_bottom_layer_opacity(self, value):
        self.bottom_layer_opacity = value / 100.0
        self.bottom_layer_item.setOpacity(self.bottom_layer_opacity)
        self.bottom_opacity_value_label.setText(f"{self.bottom_layer_opacity:.1f}")

    def choose_color(self):
        color = QColorDialog.getColor(self.pen_color, self, "选择画笔颜色")
        if color.isValid():
            self.pen_color = color

    def choose_fill_color(self):
        color = QColorDialog.getColor(self.fill_color, self, "选择填充颜色", QColorDialog.ShowAlphaChannel)
        if color.isValid():
            self.fill_color = color
            # 如果选择了完全透明，则设置填充为透明
            if color.alpha() == 0:
                self.fill_color = Qt.transparent

    def clear_top_layer(self):
        height, width = self.top_layer_array.shape
        self.top_pixmap = QPixmap(width, height)
        self.top_pixmap.fill(Qt.transparent)
        self.top_layer_item.setPixmap(self.top_pixmap)
        self.top_layer_array.fill(0)

    def eventFilter(self, source, event):
        """处理鼠标事件"""
        if source is self.graphics_view.viewport():
            if event.type() == event.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    self.handle_mouse_press(event)
                    return True
            elif event.type() == event.MouseMove:
                self.handle_mouse_move(event)
                return True
            elif event.type() == event.MouseButtonRelease:
                if event.button() == Qt.LeftButton:
                    self.handle_mouse_release(event)
                    return True
        return super().eventFilter(source, event)

    def handle_mouse_press(self, event):
        """处理鼠标按下事件"""
        scene_pos = self.graphics_view.mapToScene(event.pos())
        self.drawing = True
        self.last_point = scene_pos.toPoint()

        if self.current_tool == "fill":
            self.fill_at_point(scene_pos.toPoint())
            self.drawing = False


    def handle_mouse_move(self, event):
        """处理鼠标移动事件"""
        if not self.drawing:
            return

        scene_pos = self.graphics_view.mapToScene(event.pos())
        current_point = scene_pos.toPoint()
        # 创建临时绘图表面
        temp_pixmap = QPixmap(self.top_pixmap)
        self._draw_on_pixmap(temp_pixmap, self.last_point, current_point, is_preview=True)
        self.top_layer_item.setPixmap(temp_pixmap)

        if self.current_tool in ["pen", "eraser"]:
            self.top_pixmap = temp_pixmap
            self.last_point = current_point

    def handle_mouse_release(self, event):
        """处理鼠标释放事件"""
        if not self.drawing:
            return

        scene_pos = self.graphics_view.mapToScene(event.pos())
        current_point = scene_pos.toPoint()

        # 最终绘制
        self.top_pixmap = self._draw_on_pixmap(
            QPixmap(self.top_pixmap),
            self.last_point,
            current_point
        )
        self.top_layer_item.setPixmap(self.top_pixmap)
        self.drawing = False

        # 更新顶部图层数组
        self.update_top_layer_array()

    def _draw_on_pixmap(self, pixmap, from_point, to_point, is_preview=False):
        """通用绘图方法，支持预览和最终绘制"""
        painter = QPainter(pixmap)
        painter.setPen(QPen(self.pen_color, self.pen_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        # 检测 Shift 按键
        shift_pressed = QApplication.keyboardModifiers() == Qt.ShiftModifier

        if self.current_tool == "pen":
            painter.drawLine(from_point, to_point)

        elif self.current_tool == "eraser":
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.setPen(QPen(Qt.transparent, self.pen_size))
            painter.drawLine(from_point, to_point)

        elif self.current_tool == "line":
            if shift_pressed:
                # --- 角度约束逻辑（45°倍数）---
                dx = to_point.x() - from_point.x()
                dy = to_point.y() - from_point.y()
                length = (dx ** 2 + dy ** 2) ** 0.5  # 直线长度

                if length > 0:
                    angle = atan2(dy, dx)  # 原始角度（弧度）
                    constrained_angle = round(angle / self.angle_step) * self.angle_step  # 锁定到最近的45°倍数

                    # 修正终点坐标
                    to_point = QPointF(
                        from_point.x() + length * cos(constrained_angle),
                        from_point.y() + length * sin(constrained_angle)
                    ).toPoint()
            painter.drawLine(from_point, to_point)

        elif self.current_tool == "rect":
            rect = QRectF(from_point, to_point).normalized()
            if shift_pressed:  # 强制正方形
                size = min(rect.width(), rect.height())
                if to_point.x() >= from_point.x() and to_point.y() >= from_point.y():
                    rect = QRectF(from_point, from_point + QPointF(size, size))
                else:
                    rect = QRectF(from_point, from_point - QPointF(size, size))
            painter.drawRect(rect)

        elif self.current_tool == "ellipse":
            rect = QRectF(from_point, to_point).normalized()
            if shift_pressed:  # 强制圆形
                size = min(rect.width(), rect.height())
                if to_point.x() >= from_point.x() and to_point.y() >= from_point.y():
                    rect = QRectF(from_point, from_point + QPointF(size, size))
                else:
                    rect = QRectF(from_point, from_point - QPointF(size, size))
            painter.drawEllipse(rect)

        painter.end()
        return pixmap

    def fill_at_point(self, point):
        """在指定点进行填充"""
        # 创建临时图像
        image = self.top_pixmap.toImage()

        # 获取点击位置的颜色
        target_color = image.pixelColor(point)

        # 如果颜色已经是填充色，则不操作
        if target_color == self.fill_color:
            return

        # 执行填充算法
        self.flood_fill(image, point.x(), point.y(), target_color, self.fill_color)

        # 更新pixmap
        self.top_pixmap = QPixmap.fromImage(image)
        self.top_layer_item.setPixmap(self.top_pixmap)

        # 更新顶部图层数组
        self.update_top_layer_array()

    def flood_fill(self, image, x, y, target_color, fill_color):
        """洪水填充算法实现"""
        # 使用非递归方式实现，避免堆栈溢出
        pixels = [(x, y)]
        width = image.width()
        height = image.height()

        while pixels:
            x, y = pixels.pop()

            # 边界检查
            if x < 0 or x >= width or y < 0 or y >= height:
                continue

            current_color = image.pixelColor(x, y)

            # 检查是否需要填充
            if current_color != target_color:
                continue

            # 设置填充颜色
            image.setPixelColor(x, y, fill_color)

            # 添加相邻像素
            pixels.append((x + 1, y))
            pixels.append((x - 1, y))
            pixels.append((x, y + 1))
            pixels.append((x, y - 1))

    def update_top_layer_array(self):
        """将顶部图层QPixmap转换为二维数组"""
        image = self.top_pixmap.toImage()
        height, width = self.top_layer_array.shape

        for y in range(height):
            for x in range(width):
                if x < image.width() and y < image.height():
                    color = image.pixelColor(x, y)
                    # 按照透明度设置蒙版
                    if color.alpha() == 0:  # 完全透明
                        self.top_layer_array[y, x] = 0
                    else:
                        self.top_layer_array[y, x] = color.alpha() / 255

    def resizeEvent(self, event):
        """窗口大小改变时自适应视图"""
        super().resizeEvent(event)
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def get_top_layer_array(self):
        """获取顶部图层数组"""
        bool_mask = self.top_layer_array >128

        return self.top_layer_array.copy(), bool_mask

