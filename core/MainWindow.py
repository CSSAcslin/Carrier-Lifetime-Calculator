from datetime import datetime
from logging.handlers import RotatingFileHandler
from PyQt5 import sip
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QScrollArea,
                             QFileDialog, QSlider, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QStackedWidget, QDockWidget,
                             QStatusBar, QScrollBar
                             )
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QMetaObject
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from DataProcessor import DataProcessor
from ImageDisplayWidget import ImageDisplayWidget
from LifetimeCalculator import LifetimeCalculator, CalculationThread
from ResultDisplayWidget import ResultDisplayWidget
from ConsoleUtils import *
from ExtraDialog import *
import logging
from ROIdrawDialog import ROIdrawDialog
import resources_rc

class MainWindow(QMainWindow):
    """主窗口"""
    # 线程激活信号
    start_reg_cal_signal = pyqtSignal(dict, float, tuple, str, int, str)
    start_dis_cal_signal = pyqtSignal(dict, float, str)
    start_dif_cal_signal = pyqtSignal(dict, float, float)

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "carrier_lifetime.log")
        self.setup_menus()
        self.setup_logging()
        self.log_startup_message()

        # 参数初始化
        self.data = None
        self.time_points = None
        self.time_unit = 1.0
        self.space_unit = 1.0
        self.bool_mask = None
        self.idx = None
        self.vector_array = None
        self.plot_params = {
            'current_mode': 'heatmap',  # 'heatmap' 或 'curve'
            'line_style': '--',
            'line_width': 2,
            'marker_style': 's',
            'marker_size': 6,
            'color': '#1f77b4',
            'show_grid': False,
            'heatmap_cmap': 'jet',
            'contour_levels': 10,
            'set_axis':True,
            '_from_start_cal': False
        }
        self.cal_set_params = {
            'from_start_cal': False,
            'r_squared_min': 0.6,
            'peak_min': 0.0,
            'peak_max': 100.0,
            'tau_min': 1e-3,
            'tau_max': 1e3
        }
        # 状态控制
        self._is_calculating = False
        # 线程管理
        self.thread_open()
        # 信号连接
        self.signal_connect()

    def init_ui(self):
        self.setWindowTitle("载流子寿命分析工具")
        self.setGeometry(100, 50, 1600, 850)

        # 主部件和布局
        # main_widget = QWidget()
        # self.setCentralWidget(main_widget)

        # 左侧设置区域
        self.setup_left_panel()
        self.param_dock = QDockWidget("基础设置", self)
        self.param_dock.setWidget(self.left_panel)
        self.param_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.param_dock.setMinimumSize(300, 700)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.param_dock) # 加到左侧

        # 右侧图像区域
        self.image_display = ImageDisplayWidget(self)
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.addWidget(self.image_display)

        # 时间滑块
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_label = QLabel("时间点: 0/0")
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("时间序列:"))
        slider_layout.addWidget(self.time_slider)
        slider_layout.addWidget(self.time_label)
        image_layout.addLayout(slider_layout)
        self.image_dock = QDockWidget("图像显示", self)
        self.image_dock.setWidget(image_widget)
        self.image_dock.setMinimumSize(350, 350)
        self.image_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.image_dock)

        # 结果显示区域
        self.result_dock = QDockWidget("绘图结果", self)
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        # 垂直滑块添加
        right_layout_horizontal = QHBoxLayout()
        self.time_slider_vertical = QSlider(Qt.Vertical)
        self.time_slider_vertical.setRange(0, 0)
        self.time_slider_vertical.setVisible(False)
        self.result_display = ResultDisplayWidget()
        right_layout_horizontal.addWidget(self.time_slider_vertical)
        right_layout_horizontal.addWidget(self.result_display)
        result_layout.addLayout(right_layout_horizontal)
        self.result_dock.setWidget(result_widget)
        self.result_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.result_dock.setMinimumSize(350, 350)
        self.addDockWidget(Qt.RightDockWidgetArea, self.result_dock)
        self.splitDockWidget(self.param_dock, self.image_dock, Qt.Horizontal)
        self.splitDockWidget(self.image_dock, self.result_dock, Qt.Horizontal)
        self.resizeDocks([self.image_dock, self.result_dock], [650, 650], Qt.Horizontal)
        self.setup_status_bar()

        # 设置控制台
        self.setup_console()

    def setup_left_panel(self):
        """设置左侧面板"""
        self.left_panel = QWidget()
        self.left_panel_layout = QVBoxLayout()
        self.left_panel_layout.setContentsMargins(15,15,15,15)

    # 数据导入面板
        self.data_import = self.QGroupBoxCreator('数据导入')
        left_layout0 = QVBoxLayout()
        left_layout0.setSpacing(2)

        # 模式选择
        self.fuction_select = QComboBox()
        self.fuction_select.addItems(['请选择分析模式','FS-iSCAT','光热信号处理','EM-iSCAT'])
        left_layout0.addWidget(self.fuction_select)
        self.funtion_stack = QStackedWidget()
        nothing_group = self.QGroupBoxCreator(style="inner")
        nothing_layout = QVBoxLayout()
        nothing_layout.addWidget(QLabel("首先：请选择分析模式!"))
        nothing_group.setLayout(nothing_layout)
        self.funtion_stack.addWidget(nothing_group)

        # FS-iSCAT模式下的文件夹选择
        fs_iSCAT_group = self.QGroupBoxCreator(style="inner")
        tiff_layout = QVBoxLayout()
        self.group_selector = QComboBox()
        self.group_selector.addItems(['n', 'p'])
        self.tiff_folder_btn = QPushButton("选择TIFF文件夹")
        tiff_layout.addWidget(self.group_selector)
        tiff_layout.addWidget(self.tiff_folder_btn)
        fs_iSCAT_group.setLayout(tiff_layout)
        self.funtion_stack.addWidget(fs_iSCAT_group)

        # 光热信号处理模式下的文件夹选择
        PA_group = self.QGroupBoxCreator(style="inner")
        sif_layout = QVBoxLayout()
        method_label = QLabel("归一化方法:")         # 归一化方法选择
        self.method_combo = QComboBox()
        self.method_combo.addItems(["linear", "percentile", "sigmoid", "log", "clahe"])
        self.sif_folder_btn = QPushButton('选择SIF文件夹')
        sif_layout.addWidget(method_label)
        sif_layout.addWidget(self.method_combo)
        sif_layout.addWidget(self.sif_folder_btn)
        PA_group.setLayout(sif_layout)
        self.funtion_stack.addWidget(PA_group)

        # 文件类型为tiff
        EM_iSCAT_group = self.QGroupBoxCreator(style="inner")
        type_choose = QHBoxLayout()
        self.file_type_selector = QComboBox()
        self.file_type_selector.addItems(['avi格式', 'tiff格式'])
        file_types = QVBoxLayout()
        file_types.addWidget(self.file_type_selector)
        self.file_type_stack = QStackedWidget()
        avi_group = self.QGroupBoxCreator(style = "noborder") # avi 选择
        avi_layout = QVBoxLayout()
        self.avi_select_btn = QPushButton("选择avi文件")
        avi_layout.addWidget(self.avi_select_btn)
        avi_group.setLayout(avi_layout)
        self.file_type_stack.addWidget(avi_group)
        tiff_group = self.QGroupBoxCreator(style = "noborder") # tiff 选择
        tiff_layout = QVBoxLayout()
        self.EMtiff_folder_btn = QPushButton("选择TIFF文件夹")
        tiff_layout.addWidget(self.EMtiff_folder_btn)
        tiff_group.setLayout(tiff_layout)
        self.file_type_stack.addWidget(tiff_group)
        type_choose.addLayout(file_types)
        type_choose.addWidget(self.file_type_stack)
        EM_iSCAT_group.setLayout(type_choose)
        self.funtion_stack.addWidget(EM_iSCAT_group)
        # 总提示
        self.folder_path_label = QLabel("未选择文件夹")
        self.folder_path_label.setMaximumWidth(300)
        self.folder_path_label.setWordWrap(True)
        self.folder_path_label.setStyleSheet("font-size: 14px;")  # 后续还要改

        left_layout0.addWidget(self.funtion_stack)
        left_layout0.addSpacing(3)
        left_layout0.addWidget(self.folder_path_label)
        self.data_import.setLayout(left_layout0)

    # 参数设置
        self.parameter_panel = self.QGroupBoxCreator("参数设置")
        left_layout = QVBoxLayout()
        left_layout.setSpacing(2)
        # left_layout.setContentsMargins(1,7,1,7)
        time_step_layout = QHBoxLayout()
        time_step_layout.addWidget(QLabel("时间单位:"))
        self.time_step_input = QDoubleSpinBox()
        self.time_step_input.setMinimum(0.001)
        self.time_step_input.setMaximum(10000)
        self.time_step_input.setValue(1.0)
        self.time_step_input.setDecimals(3)
        time_step_layout.addWidget(self.time_step_input)
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(["ms", "μs", "ns", "ps", "fs"])
        self.time_unit_combo.setCurrentIndex(3)
        time_step_layout.addWidget(self.time_unit_combo)
        time_step_layout.addWidget(QLabel("/帧"))
        left_layout.addLayout(time_step_layout)
        left_layout.addWidget(QLabel("     (最小分辨率：.001 fs)"))
        left_layout.addSpacing(5)
        space_step_layout = QHBoxLayout()
        space_step_layout.addWidget(QLabel("空间单位:"))
        self.space_step_input = QDoubleSpinBox()
        self.space_step_input.setMinimum(0.001)
        self.space_step_input.setValue(1.0)
        self.space_step_input.setDecimals(3)
        space_step_layout.addWidget(self.space_step_input)
        self.space_unit_combo = QComboBox()
        self.space_unit_combo.addItems(["mm", "μm", "nm"])
        self.space_unit_combo.setCurrentIndex(1)
        space_step_layout.addWidget(self.space_unit_combo)
        space_step_layout.addWidget(QLabel("/像素"))
        left_layout.addLayout(space_step_layout)
        left_layout.addWidget(QLabel('     (最小分辨率：.001 nm)'))
        self.parameter_panel.setLayout(left_layout)

    # 分析总体设置
        self.modes_panel = self.QGroupBoxCreator("分析设置:")
        left_layout1 = QVBoxLayout()
        left_layout1.setContentsMargins(1, 2, 1, 2)
        self.between_stack = QStackedWidget()
        # 默认显示
        nothing_GROUP = self.QGroupBoxCreator(style="noborder")
        nothing_layout1 = QVBoxLayout()
        nothing_layout1.addWidget(QLabel("首先：请选择分析模式!"))
        nothing_GROUP.setLayout(nothing_layout1)
        self.between_stack.addWidget(nothing_GROUP)

        # fs_iSCAT下的功能选择
        fs_iSCAT_GROUP = self.QGroupBoxCreator(style="noborder")
        operation_layout = QVBoxLayout()
        # 寿命模型选择
        lifetime_layout = QHBoxLayout()
        lifetime_layout.addWidget(QLabel("寿命模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["单指数衰减", "双指数-仅区域"])
        lifetime_layout.addWidget(self.model_combo)
        # 区域分析设置
        # operation_layout.addSpacing(10)
        operation_mode_layout = QHBoxLayout()
        operation_mode_layout.addWidget(QLabel("模式:"))
        self.FS_mode_combo = QComboBox()
        self.FS_mode_combo.addItems(["载流子寿命热图", "选区寿命曲线","载流子扩散系数计算"])
        operation_mode_layout.addWidget(self.FS_mode_combo)
        operation_layout.addLayout(operation_mode_layout)
        self.FS_mode_stack = QStackedWidget()
        # 载流子寿命分布图参数板
        heatmap_group = self.QGroupBoxCreator(style = "inner")
        heatmap_layout = QVBoxLayout()
        heatmap_layout.addLayout(lifetime_layout)
        self.analyze_btn = QPushButton("开始分析")
        heatmap_layout.addWidget(self.analyze_btn)
        heatmap_group.setLayout(heatmap_layout)
        self.FS_mode_stack.addWidget(heatmap_group)
        # 特定区域寿命分析功能参数板
            # 区域分析参数
        self.region_shape_combo = QComboBox()
        self.region_shape_combo.addItems(["正方形", "圆形"])
        self.region_size_input = QSpinBox()
        self.region_size_input.setMinimum(1)
        self.region_size_input.setMaximum(50)
        self.region_size_input.setValue(5)
        self.analyze_region_btn = QPushButton("分析选定区域")
            # 区域坐标输入
        self.region_x_input = QSpinBox()
        self.region_y_input = QSpinBox()
        self.region_x_input.setMaximum(131)
        self.region_y_input.setMaximum(131)
            # 区域分析面板生成
        region_group = self.QGroupBoxCreator(style = "inner")
        region_layout = QVBoxLayout()
        lifetime_layout = QHBoxLayout()
        lifetime_layout.addWidget(QLabel("寿命模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["单指数衰减", "双指数-仅区域"])
        lifetime_layout.addWidget(self.model_combo)
        coord_layout = QHBoxLayout()
        coord_layout.addWidget(QLabel("中心X:"))
        coord_layout.addWidget(self.region_x_input)
        coord_layout.addWidget(QLabel("中心Y:"))
        coord_layout.addWidget(self.region_y_input)
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(QLabel("区域形状:"))
        shape_layout.addWidget(self.region_shape_combo)
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("区域大小:"))
        size_layout.addWidget(self.region_size_input)
        region_layout.addLayout(lifetime_layout)
        region_layout.addLayout(coord_layout)
        region_layout.addLayout(shape_layout)
        region_layout.addLayout(size_layout)
        region_layout.addWidget(self.analyze_region_btn)
        region_group.setLayout(region_layout)
        self.FS_mode_stack.addWidget(region_group)
        # 载流子扩散系数计算参数板
        diffusion_group = self.QGroupBoxCreator(style = "inner")
        diffusion_layout = QVBoxLayout()
        self.vector_signal_btn = QPushButton("1.计算ROI上全时信号强度")
        self.frame_input = QTextEdit()
        self.frame_input.setPlaceholderText("2.输入帧位（起始帧位为0），以逗号或分号分隔，范围用-\n输入all选取全部帧")
        self.frame_input.setFixedHeight(70)
        self.select_frames_btn = QPushButton("3.计算选定时刻信号强度")
        self.diffusion_coefficient_btn = QPushButton("4.展示方差演化图及扩散系数")
        diffusion_layout.addWidget(self.vector_signal_btn)
        diffusion_layout.addWidget(self.frame_input)
        diffusion_layout.addWidget(self.select_frames_btn)
        diffusion_layout.addWidget(self.diffusion_coefficient_btn)
        diffusion_group.setLayout(diffusion_layout)
        self.FS_mode_stack.addWidget(diffusion_group)
        operation_layout.addWidget(self.FS_mode_stack)
        fs_iSCAT_GROUP.setLayout(operation_layout)
        self.between_stack.addWidget(fs_iSCAT_GROUP)

        # 光热信号处理下的功能选择（因为功能冲突，此板块暂不启用，通过between_stack_change覆盖选取）
        PA_GROUP = self.QGroupBoxCreator(style="noborder")
        PA_layout1 = QVBoxLayout()
        # self.PA_mode_combo = QComboBox()
        # self.PA_mode_combo.addItems(["选区寿命指数衰减检测"])
        # PA_layout1.addWidget(self.PA_mode_combo)
        # self.PA_mode_stack = QStackedWidget()
        # # 区域分析参数
        # self.region_shape_combo = QComboBox()
        # self.region_shape_combo.addItems(["正方形", "圆形"])
        # self.region_size_input = QSpinBox()
        # self.region_size_input.setMinimum(1)
        # self.region_size_input.setMaximum(50)
        # self.region_size_input.setValue(5)
        # self.analyze_region_btn = QPushButton("分析选定区域")
        # # 区域坐标输入
        # self.region_x_input = QSpinBox()
        # self.region_y_input = QSpinBox()
        # self.region_x_input.setMaximum(131)
        # self.region_y_input.setMaximum(131)
        # # 区域分析面板生成
        # region_group = self.QGroupBoxCreator(style="inner")
        # region_layout = QVBoxLayout()
        # lifetime_layout = QHBoxLayout()
        # lifetime_layout.addWidget(QLabel("寿命模型:"))
        # self.model_combo = QComboBox()
        # self.model_combo.addItems(["单指数衰减", "双指数-仅区域"])
        # lifetime_layout.addWidget(self.model_combo)
        # coord_layout = QHBoxLayout()
        # coord_layout.addWidget(QLabel("中心X:"))
        # coord_layout.addWidget(self.region_x_input)
        # coord_layout.addWidget(QLabel("中心Y:"))
        # coord_layout.addWidget(self.region_y_input)
        # shape_layout = QHBoxLayout()
        # shape_layout.addWidget(QLabel("区域形状:"))
        # shape_layout.addWidget(self.region_shape_combo)
        # size_layout = QHBoxLayout()
        # size_layout.addWidget(QLabel("区域大小:"))
        # size_layout.addWidget(self.region_size_input)
        # region_layout.addLayout(lifetime_layout)
        # region_layout.addLayout(coord_layout)
        # region_layout.addLayout(shape_layout)
        # region_layout.addLayout(size_layout)
        # region_layout.addWidget(self.analyze_region_btn)
        # region_group.setLayout(region_layout)
        # self.PA_mode_stack.addWidget(region_group)
        # PA_layout1.addWidget(self.PA_mode_stack)
        PA_GROUP.setLayout(PA_layout1)
        self.between_stack.addWidget(PA_GROUP)

        # EM_iSCAT下的功能选择
        EM_iSCAT_GROUP = self.QGroupBoxCreator(style="noborder")
        EM_iSCAT_layout1 = QVBoxLayout()
        self.EM_mode_combo = QComboBox()
        self.EM_mode_combo.addItems(["未完成"])
        EM_iSCAT_layout1.addWidget(self.EM_mode_combo)
        self.EM_mode_stack = QStackedWidget()
        self.EM_mode_stack.addWidget(QLabel("未完成"))
        EM_iSCAT_layout1.addWidget(self.EM_mode_stack)
        EM_iSCAT_GROUP.setLayout(EM_iSCAT_layout1)
        self.between_stack.addWidget(EM_iSCAT_GROUP)

        left_layout1.addWidget(self.between_stack)
        self.modes_panel.setLayout(left_layout1)

        # 添加分析按钮和导出按钮
        data_save_layout = QHBoxLayout()
        self.export_image_btn = QPushButton("导出结果为图片")
        self.export_data_btn = QPushButton("导出结果为数据")
        data_save_layout.addWidget(self.export_image_btn)
        data_save_layout.addWidget(self.export_data_btn)

        self.left_panel_layout.addWidget(self.data_import)
        self.left_panel_layout.addWidget(self.parameter_panel)
        self.left_panel_layout.addWidget(self.modes_panel)
        self.left_panel_layout.addSpacing(15)
        self.left_panel_layout.addLayout(data_save_layout)
        self.left_panel.setLayout(self.left_panel_layout)

    def between_stack_change(self):
        if self.fuction_select.currentIndex() == 0: # nothing
            self.between_stack.setCurrentIndex(0)
            self.FS_mode_combo.setCurrentIndex(0)
        if self.fuction_select.currentIndex() == 1:  # FS-iSCAT
            self.between_stack.setCurrentIndex(1)
            self.FS_mode_combo.setCurrentIndex(0)
        if self.fuction_select.currentIndex() == 2:  # PA
            self.between_stack.setCurrentIndex(1)
            self.FS_mode_combo.setCurrentIndex(1)
        if self.fuction_select.currentIndex() == 3:  # FS-iSCAT
            self.between_stack.setCurrentIndex(3)


    def setup_menus(self):
        """加入菜单栏"""
        self.menu = self.menuBar()
        self.menu.addMenu('主窗口')

        # 控制台
        view_menu = self.menu.addMenu("控制台")
        toggle_console = view_menu.addAction("显示/隐藏控制台")
        toggle_console.triggered.connect(lambda: self.console_dock.setVisible(not self.console_dock.isVisible()))

        # 编辑菜单
        edit_menu = self.menu.addMenu("编辑")

        # 坏点处理功能
        bad_frame_edit = edit_menu.addAction("坏点处理")
        bad_frame_edit.triggered.connect(self.bad_frame_edit_dialog)

        # 计算设置功能
        data_select_edit = edit_menu.addAction("计算设置")
        data_select_edit.triggered.connect(self.calculation_set_edit_dialog)

        # 绘图设置调整
        plt_settings_edit = edit_menu.addAction("绘图设置")
        plt_settings_edit.triggered.connect(self.plt_settings_edit_dialog)

        # ROI绘图
        ROI_function = self.menu.addAction("ROI选取")
        ROI_function.triggered.connect(self.roi_select_dialog)

    @staticmethod
    def QGroupBoxCreator(title="",style="default"):
        # 全局Box样式定义
        group_box = QGroupBox(title)
        styles = {
            "default": """
            QGroupBox{
                border:1px solid gray;
                border-radius:5px;
                margin-top:5px;
                padding:15px;
                padding-left: 5px;
                padding-right: 5px;
            }
            QGroupBox::title{
                ubcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                color: #333333;
            }
            """,
            "inner":"""
            QGroupBox{
                border: 1px solid gray;
                border-radius: 5px;
                margin-top: 5px;
                padding: 5px;
                padding-left: 0px;
                padding-right: 0px;
            }""",
            "noborder":"""
            QGroupBox{
                border: 0;
                border-radius: 0px;
                margin-top: 0px;
            }"""
        }
        group_box.setStyleSheet(styles.get(style, styles["default"]))
        return group_box

    def setup_status_bar(self):
        """设置状态条"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # 状态文本
        self.status_label = QLabel("准备就绪")
        self.status_label.setFixedWidth(250)
        self.status_bar.addWidget(self.status_label)
        # 鼠标悬停显示
        self.mouse_pos_label = QLabel("鼠标位置: x= -, y= -, t= -; 值: -")
        self.mouse_pos_label.setFixedWidth(500)
        self.status_bar.addWidget(self.mouse_pos_label)
        self._handle_hover = self.make_hover_handler()
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(650)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                background: white;
                text-align: center;
                min-height: 18px;
                max-height: 18px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
            }
        """)
        self.status_bar.addWidget(self.progress_bar)

        # 状态指示灯 (红绿灯)
        self.status_light = QLabel()
        self.status_light.setPixmap(QPixmap(":/icons/green_light.png").scaled(16, 16))
        self.status_bar.addPermanentWidget(self.status_light)

    def update_status(self, status, is_working=False):
        """更新状态条的显示"""
        self.status_label.setText(status)
        light = "green_light.png" if not is_working else "yellow_light.png"
        self.status_light.setPixmap(QPixmap(f":/icons/{light}").scaled(16, 16))

    def setup_console(self):
        """设置控制台停靠窗口"""
        self.console_dock = QDockWidget("控制台", self)
        self.console_dock.setObjectName("ConsoleDock")

        # 创建控制台部件
        self.console_widget = ConsoleWidget(self)
        self.command_processor = CommandProcessor(self)

        self.console_dock.setWidget(self.console_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console_dock)
        self.splitDockWidget(self.result_dock, self.console_dock, Qt.Vertical)
        self.resizeDocks([self.result_dock, self.console_dock], [550, 300], Qt.Vertical)
        # 设置控制台特性
        self.console_dock.setMinimumWidth(400)
        self.console_dock.setMinimumHeight(200)
        self.console_dock.setFeatures(QDockWidget.DockWidgetMovable |
                                      QDockWidget.DockWidgetFloatable |
                                      QDockWidget.DockWidgetClosable)

    def setup_logging(self):
        """配置日志系统"""
        # 确保日志目录存在
        log_dir = os.path.dirname(self.log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 设置轮转文件处理器 (每个文件最大5MB，保留3个备份)
        file_handler = RotatingFileHandler(
            self.log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))

        # 设置控制台处理器
        console_handler = ConsoleHandler(self)

        # 配置根日志记录器
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()  # 清除现有处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        sys.stdout = StreamLogger(logging.INFO)
        sys.stderr = StreamLogger(logging.ERROR)

    def log_to_console(self, message):
        """将消息输出到控制台"""
        self.console_widget.console_output.append(message)
        self.console_widget.console_output.verticalScrollBar().setValue(
            self.console_widget.console_output.verticalScrollBar().maximum()
        )

    def log_startup_message(self):
        """记录程序启动消息"""
        startup_msg = f"""\n
============================================
载流子寿命分析工具启动
启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
日志位置: {self.log_file}
程序版本: 1.8.0
============================================
        """
        logging.info(startup_msg.strip())
        logging.info("程序已进入准备状态，等待用户操作...（第一次计算可能较慢）")

    def thread_open(self):
        """计算线程相关 以及信号槽连接都放在这里了"""
        self.thread = QThread()
        self.cal_thread = CalculationThread()
        self.cal_thread.moveToThread(self.thread)
        # 计算状态更新
        self.start_reg_cal_signal.connect(self.cal_thread.region_analyze)
        self.start_dis_cal_signal.connect(self.cal_thread.distribution_analyze)
        self.start_dif_cal_signal.connect(self.cal_thread.diffusion_calculation)
        self.cal_thread.calculating_progress_signal.connect(self.update_progress)
        self.cal_thread.result_data_signal.connect(self.result_display.display_lifetime_curve)
        self.cal_thread.result_map_signal.connect(self.result_display.display_distribution_map)
        self.cal_thread.diffusion_coefficient_signal.connect(self.result_display.display_diffusion_coefficient)
        self.cal_thread.stop_thread_signal.connect(self.stop_thread)
        self.cal_thread.cal_time.connect(lambda ms: logging.info(f"耗时: {ms}毫秒"))
        self.cal_thread.cal_running_status.connect(self.btn_safety)

    def signal_connect(self):
        # 连接参数区域按钮
        self.fuction_select.currentIndexChanged.connect(self.funtion_stack.setCurrentIndex)
        self.fuction_select.currentIndexChanged.connect(self.between_stack_change)
        self.file_type_selector.currentIndexChanged.connect(self.file_type_stack.setCurrentIndex)
        self.tiff_folder_btn.clicked.connect(self.load_tiff_folder)
        self.sif_folder_btn.clicked.connect(self.load_sif_folder)
        self.avi_select_btn.clicked.connect(self.load_avi_file)
        self.EMtiff_folder_btn.clicked.connect(self.load_tiff_folder_EM)
        self.analyze_region_btn.clicked.connect(self.region_analyze_start)
        self.analyze_btn.clicked.connect(self.distribution_analyze_start)
        self.FS_mode_combo.currentIndexChanged.connect(self.FS_mode_stack.setCurrentIndex)
        # self.PA_mode_combo.currentIndexChanged.connect(self.PA_mode_stack.setCurrentIndex)
        self.EM_mode_combo.currentIndexChanged.connect(self.EM_mode_stack.setCurrentIndex)
        self.vector_signal_btn.clicked.connect(self.vectorROI_signal_show)
        self.select_frames_btn.clicked.connect(self.vectorROI_selection)
        self.diffusion_coefficient_btn.clicked.connect(self.result_display.plot_variance_evolution)
        self.export_image_btn.clicked.connect(self.export_image)
        self.export_data_btn.clicked.connect(self.export_data)
        # 鼠标移动
        self.image_display.mouse_position_signal.connect(self._handle_hover)
        self.image_display.mouse_clicked_signal.connect(self._handle_click)
        # 时间滑块
        self.time_slider.valueChanged.connect(self.update_time_slice)
        self.time_slider_vertical.valueChanged.connect(self.update_result_display)
        # 连接控制台信号
        self.command_processor.terminate_requested.connect(self.stop_calculation)
        self.command_processor.save_config_requested.connect(self.save_config)
        self.command_processor.load_config_requested.connect(self.load_config)
        self.command_processor.clear_result_requested.connect(self.clear_result)
        # 结果区域信号
        self.result_display.tab_type_changed.connect(self._handle_result_tab)

    '''上面是初始化预设，下面是功能响应'''
    def load_tiff_folder(self):
        """加载TIFF文件夹(FS-iSCAT)"""
        self.time_unit = float(self.time_step_input.value())
        folder_path = QFileDialog.getExistingDirectory(self, "选择TIFF图像文件夹")
        self.data_processor = DataProcessor(folder_path)
        if folder_path:
            logging.info(folder_path)
            self.folder_path_label.setText("已加载TIFF文件夹")
            current_group = self.group_selector.currentText()

            # 读取文件夹中的所有tiff文件
            tiff_files = self.data_processor.load_and_sort_tiff(current_group)

            if not tiff_files:
                self.folder_path_label.setText("文件夹中没有目标TIFF文件")
                return

            # 读取所有图像
            self.data = self.data_processor.process_tiff(tiff_files)

            if not self.data:
                self.folder_path_label.setText("无法读取TIFF文件")
                return

            logging.info('成功加载TIFF数据')
            # 设置时间滑块
            self.time_slider.setMaximum(len(self.data['images']) - 1)
            self.time_label.setText(f"时间点: 0/{len(self.data['images']) - 1}")

            # 显示第一张图像
            self.update_time_slice(0,True)
            self.time_slider.setValue(0)

            # 根据图像大小调节region范围
            self.region_x_input.setMaximum(self.data['images'].shape[1])
            self.region_y_input.setMaximum(self.data['images'].shape[2])

    def load_sif_folder(self):
        '''加载SIF文件夹'''
        folder_path = QFileDialog.getExistingDirectory(self, "选择SIF图像文件夹")
        self.data_processor = DataProcessor(folder_path,self.method_combo.currentText())
        if folder_path:
            logging.info(folder_path)
            self.folder_path_label.setText("已加载SIF文件夹")

            # 读取文件夹中的所有sif文件
            check_sif = self.data_processor.load_and_sort_sif()

            if not check_sif:
                self.folder_path_label.setText("文件夹中没有目标SIF文件")
                logging.warning("请确认选择的文件格式是否匹配")
                return

            # 读取所有图像
            self.data = self.data_processor.process_sif()

            if not self.data:
                self.folder_path_label.setText("无法读取sif文件")
                return

            logging.info('成功加载SIF数据')
            # 设置时间滑块
            self.time_slider.setMaximum(len(self.data['images']) - 1)
            self.time_label.setText(f"时间点: 0/{len(self.data['images']) - 1}")

            # 显示第一张图像
            self.update_time_slice(0,True)
            self.time_slider.setValue(0)

            # 根据图像大小调节region范围
            self.region_x_input.setMaximum(self.data['images'].shape[1])
            self.region_y_input.setMaximum(self.data['images'].shape[2])
        pass

    def load_avi_file(self):
        """加载avi文件"""
        pass

    def load_tiff_folder_EM(self):
        """加载TIFF文件夹(FS-iSCAT)"""
        pass

    def make_hover_handler(self):
        args = {'x': None, 'y': None, 't': None, 'value': None}
        def _handle_hover(x=None, y=None, t=None, value=None):
            """鼠标位置显示"""
            # 更新传入的参数（未传入的保持原值）
            if x is not None: args['x'] = x
            if y is not None: args['y'] = y
            if t is not None: args['t'] = t
            if value is not None:
                args['value'] = value
            else:
                args['value'] = self.data['images'][args['t'], args['y'], args['x']]
            if args['x'] is None or args['y'] is None:
                return

            # 更新显示
            self.mouse_pos_label.setText(
                f"鼠标位置: x={args['x']}, y={args['y']}, t={args['t']}; 归一值: {args['value']:.2f}, 原始值：{self.data['data_origin'][args['t'], args['y'], args['x']]:.6e}")

        return _handle_hover

    def _handle_click(self, x, y):
        """处理图像点击事件"""
        if self.FS_mode_combo.currentIndex() == 1 :  # 区域分析模式 or self.PA_mode_combo.currentIndex() == 0
            self.region_x_input.setValue(x)
            self.region_y_input.setValue(y)

    def _handle_result_tab(self, tab_type):
        """特殊标签页类型处理"""
        if tab_type == 'roi':
            # 如果是roi结果
            self.time_slider_vertical.setVisible(True)
            self.time_slider_vertical.setMaximum(self.data['data_origin'].shape[0] - 1)
            self.time_slider_vertical.setValue(0)
        else:
            if self.time_slider_vertical.isVisible():
                self.time_slider_vertical.setVisible(False)

    def bad_frame_edit_dialog(self):
        """显示坏点处理对话框"""
        if not hasattr(self, 'data') or self.data is None:
            logging.warning("无数据，请先加载数据文件")
            return

        dialog = BadFrameDialog(self)
        self.update_status("坏点修复ing", True)
        if dialog.exec_():
            # 更新图像显示
            self.update_time_slice(0)
            self.time_slider.setValue(0)
            logging.info(f"坏点处理完成，修复了 {len(dialog.bad_frames)} 个坏帧")
        self.update_status("准备就绪", False)

    def calculation_set_edit_dialog(self):
        """计算设置调整"""
        if not hasattr(self, 'data') or self.data is None:
            logging.warning("无数据，请先加载数据文件")
            return
        self.update_status("计算设置ing", True)
        dialog = CalculationSetDialog(self.cal_set_params)
        if dialog.exec_():
            self.update_time_slice(0)
            self.time_slider.setValue(0)
            self.cal_set_params = dialog.params
            LifetimeCalculator.set_cal_parameters(self.cal_set_params)
            # 同步修改绘图设置并传参
            self.plot_params['_from_start_cal'] = self.cal_set_params['from_start_cal']
            self.result_display.update_plot_settings(self.plot_params, update=False)
            logging.info("设置已更新，请重新绘图")
        self.update_status("准备就绪", False)

    def plt_settings_edit_dialog(self):
        """绘图设置"""
        dialog = PltSettingsDialog(params=self.plot_params)
        self.update_status("绘图设置ing", True)
        if dialog.exec_():
            # 将参数传递给ResultDisplayWidget
            self.result_display.update_plot_settings(dialog.params)
            self.plot_params = dialog.params
            logging.info("绘图已更新")
        self.update_status("准备就绪", False)

    def roi_select_dialog(self):
        """ROI选取功能"""
        if not hasattr(self, 'data') or self.data is None:
            logging.warning("无数据，请先加载数据文件")
            return
        roi_dialog = ROIdrawDialog(base_layer_array=self.data['images'][self.idx],parent=self)
        self.update_status("ROI绘制ing", True)
        if roi_dialog.exec_() == QDialog.Accepted:
            if roi_dialog.action_type == "mask":
                self.mask, self.bool_mask = roi_dialog.get_top_layer_array()
                logging.info(f'成功绘制ROI，大小{self.mask.shape[0]}×{self.mask.shape[1]}')
                data_amend = self.data_processor.amend_data(self.data, self.bool_mask)
                self.data.update(data_amend)
                self.update_time_slice(self.idx)
            elif roi_dialog.action_type == "vector":
                self.vector_array = roi_dialog.vector_line.getPixelValues(self.data,self.space_unit,self.time_unit)
                logging.info(f'成功绘制ROI，大小{self.vector_array.shape}')


        self.update_status("准备就绪", False)

    def update_time_slice(self, idx, first_create = False):
        """更新时间切片显示"""
        self.idx = idx
        if self.data is not None and 0 <= idx < len(self.data['images']):
            self.time_label.setText(f"时间点: {idx}/{len(self.data['images']) - 1}")
            self.image_display.current_image = self.data['images'][idx]
            if first_create:
                self.image_display.display_image(self.data['images'][idx],idx)
            else:
                self.image_display.update_display_idx(self.data['images'][idx],idx)
                self._handle_hover(t = idx)

    def update_progress(self, current, total=None):
        """更新进度条"""
        if total is not None:
            self.progress_bar.setMaximum(total)

        self.progress_bar.setValue(current)
        self.console_widget.update_progress(current, total)

        if current == 1:
            # self.update_status("计算中...", True)
            pass
        elif current >= self.progress_bar.maximum():
            self.update_status("计算完成")
            self.progress_bar.reset()

    def vectorROI_signal_show(self):
        """向量选取信号全部展示"""
        if not hasattr(self, 'data') or self.data is None:
            logging.warning("无数据，请先加载数据文件")
            return
        if self.vector_array is None :
            logging.warning("未选取向量直线ROI")
            return
        elif self.data['data_origin'].shape[0] == self.vector_array.shape[0]:
            # self.time_slider_vertical.setVisible(True)
            # self.time_slider_vertical.setMaximum(self.data['data_origin'].shape[0] - 1)
            # self.time_slider_vertical.setValue(0)
            self.update_result_display(0,reuse_current = False)
            return
        else:
            logging.error("数据长度不匹配")
            return

    def update_result_display(self,idx,reuse_current=True):
        if self.vector_array is not None and 0 <= idx < self.vector_array.shape[0]:
            frame_data = self.vector_array[idx]
            self.result_display.display_roi_series(
                positions=frame_data[:, 0],
                intensities=frame_data[:, 1],
                fig_title=f"ROI信号强度 (帧:{idx})",
                reuse_current = reuse_current

            )

    def vectorROI_selection(self):
        """向量选取信号选择展示"""
        frames = self.parse_frame_input()
        if not frames :
            logging.warning("请输入选取的帧数")
        elif not hasattr(self, 'vector_array'):
            logging.warning("请选择矢量直线绘制ROI选区")
            return
        elif frames is None:
            logging.warning("帧数无效，请重新输入")
            return

        # 检查帧数是否有效
        invalid_frames = [f for f in frames if f < 0 or f > self.max_frame]
        if invalid_frames:
            QMessageBox.warning(
                self, "帧数超出范围",
                f"有效帧数范围: 0-{self.max_frame}\n无效帧: {invalid_frames}"
            )
            logging.warning("请输入有效帧数")
            frames = [f for f in frames if 0 <= f <= self.max_frame]
            if not frames:
                return
        else:
            logging.info(f"输入帧数{frames}，帧数有效可以处理")

        # 收集选定帧的数据
        self.vectorROI_data = {f: self.vector_array[f] for f in frames}

        # 信号拟合和绘制
        self.diffusion_calculation_start()

        # # 自动显示方差演化图
        # self.display_diffusion_coefficient()

    def parse_frame_input(self):
        """解析用户输入的帧数"""
        text = self.frame_input.toPlainText()
        self.max_frame = self.vector_array.shape[0] - 1
        # 替换所有分隔符为逗号
        if text == 'all':
            return list(range(self.max_frame + 1))
        if not text:
            QMessageBox.warning(self, "输入为空", "请输入有效的帧数选择")
            return None

        text = text.replace(';', ',').replace('，', ',')
        frames = set()
        parts = text.split(',')
        try:
            for part in parts:
                if not part:
                    continue
                # 处理范围输入 (e.g., "5-10", "20-25")
                if '-' in part:
                    range_parts = part.split('-')
                    if len(range_parts) != 2:
                        raise ValueError(f"无效的范围格式: {part}")

                    start = int(range_parts[0])
                    end = int(range_parts[1])

                    if start > end:
                        raise ValueError(f"起始帧({start})不能大于结束帧({end})")

                    # 确保范围在有效区间内
                    if start < 0 or end > self.max_frame:
                        raise ValueError(f"范围 {part} 超出有效帧范围 (0-{self.max_frame})")

                    frames.update(range(start, end + 1))

                # 处理单帧数字
                else:
                    frame = int(part)
                    if frame < 0 or frame > self.max_frame:
                        raise ValueError(f"帧号 {frame} 超出有效范围 (0-{self.max_frame})")
                    frames.add(frame)
            return sorted(frames)
        except ValueError as e:
            QMessageBox.warning(
                self,
                "输入错误",
                f"无效输入: {str(e)}\n\n正确格式示例:\n"
                "• 单帧: 5\n"
                "• 序列: 1,3,5,7\n"
                "• 范围: 10-15,20-25\n"
                "• 混合: 1,3-5,7,9-10\n"
                f"• 所有帧: all\n\n有效帧范围: 0-{self.max_frame}"
            )
            logging.warning(f"帧数输入错误: {str(e)} - 输入内容: {text}")
            return None

    def region_analyze_start(self):
        """分析选定区域载流子寿命"""
        if self.data is None or not hasattr(self, 'time_points'):
            return logging.warning('无数据载入')
        # self.time_slider_vertical.setVisible(False)
        # 如果线程没了，要开启
        if not self.is_thread_active("thread"):
            self.thread_open()
        # 如果有线程在运算，要提示（不过目前不需要，保留语句）
        if self.cal_thread and self.thread.isRunning():
            logging.warning("已有计算任务正在运行")
            return

        self.thread.start()
        self.update_status('计算进行中...', True)
        self.time_unit = float(self.time_step_input.value())
        center = (self.region_y_input.value(), self.region_x_input.value())
        shape = 'square' if self.region_shape_combo.currentText() == "正方形" else 'circle'
        size = self.region_size_input.value()
        model_type = 'single' if self.model_combo.currentText() == "单指数衰减" else 'double'
        self.start_reg_cal_signal.emit(self.data,self.time_unit,center,shape,size,model_type)

    def distribution_analyze_start(self):
        """分析载流子寿命"""
        if self.data is None:
            return logging.warning('无数据载入')
        # self.time_slider_vertical.setVisible(False)
        # 如果线程没了，要创建
        if not self.is_thread_active("thread"):
            self.thread_open()


        self.thread.start()
        self.update_status('长时计算进行中...', True)
        self.time_unit = float(self.time_step_input.value())
        model_type = 'single' if self.model_combo.currentText() == "单指数衰减" else 'double'
        self.start_dis_cal_signal.emit(self.data,self.time_unit,model_type)

    def diffusion_calculation_start(self):
        """扩散系数计算"""
        if self.data is None:
            return logging.warning('无数据载入')
        elif self.vectorROI_data is None:
            return  logging.warning("无有效ROI数据")
        # self.time_slider_vertical.setVisible(False)
        # 如果线程没了，要创建
        if not self.is_thread_active("thread"):
            self.thread_open()

        self.thread.start()
        self.update_status('计算进行中...', True)
        self.time_unit = float(self.time_step_input.value())
        self.space_unit = float(self.space_step_input.value())
        self.start_dif_cal_signal.emit(self.vectorROI_data,self.time_unit, self.space_unit)

    def is_thread_active(self, thread_name: str) -> bool:
        """检查指定名称的线程是否存在且正在运行"""
        # :param thread_name: 线程对象的变量名（str）
        # :return: True（线程存在且运行中）/ False（线程不存在或已结束）

        if hasattr(self, thread_name):
            thread = getattr(self, thread_name)  # 动态获取线程对象
            if isinstance(thread, QThread) and not sip.isdeleted(thread):
                return thread.isRunning()
        return False

    def btn_safety(self, cal_run=False):
        """关闭按钮的功能"""
        if cal_run:
            self.analyze_btn.setEnabled(False)
            self.analyze_region_btn.setEnabled(False)
        elif not cal_run:
            self.analyze_btn.setEnabled(True)
            self.analyze_region_btn.setEnabled(True)
        return

    def stop_thread(self):
        """彻底删除线程（反正关闭也不能重启）后续线程多了加入选择关闭的能力"""
        self.thread.quit()  # 请求退出
        self.thread.wait()  # 等待结束
        self.thread.deleteLater()  # 标记删除
        logging.info("计算线程关闭")

    def export_image(self):
        """导出热图为图片"""
        current_index = self.result_display.currentIndex()
        if current_index < 0:
            QMessageBox.warning(self, "导出失败", "没有可导出的图像")
            return

        tab = self.result_display.widget(current_index)
        canvas = tab.findChild(FigureCanvas)

        if not canvas:
            QMessageBox.warning(self, "导出失败", "未找到图像画布")
            return

        try:
            path, _ = QFileDialog.getSaveFileName(
                self, "保存图像", "", "PNG(*.png);;JPEG(*.jpg);;TIFF图像 (*.tif *.tiff);;所有文件(*.*)"
            )

            if path:
                canvas.figure.savefig(path, dpi=300)
                QMessageBox.information(self, "导出成功", f"图像已保存至:\n{path}")
                logging.info(f"导出成功,图像已保存至:{path}")
        except:
            logging.info("数据未保存")

        # if hasattr(self.result_display, 'current_data'):
        #     file_path, _ = QFileDialog.getSaveFileName(
        #         self, "保存图像", "", "PNG图像 (*.png);;JPEG图像 (*.jpg *.jpeg);;TIFF图像 (*.tif *.tiff)")
        #
        #     if file_path:
        #         # 从matplotlib保存图像
        #         self.result_display.figure.savefig(file_path, dpi=300, bbox_inches='tight')
        #     logging.info("图片已保存")

    def export_data(self):
        """导出寿命数据"""
        if hasattr(self.result_display, 'current_dataframe'):
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存数据", "", "CSV文件 (*.csv);;文本文件 (*.txt)")

            if file_path:
                # 保存为CSV或TXT
                if file_path.lower().endswith('.csv'):
                    try:
                        self.result_display.current_dataframe.to_csv(file_path, index=False, header=False)
                        logging.info("数据已保存")
                    except:
                        logging.info("数据未保存")
                else:
                    try:
                        self.result_display.current_dataframe.to_csv(file_path, sep='\t', index=False, header=False)
                        logging.info("数据已保存")
                    except:
                        logging.info("数据未保存")
            else:
                logging.info("数据未保存")
                return

    '''以下控制台命令更新'''
    def stop_calculation(self):
        """终止当前计算"""
        # 这里需要实现终止计算的逻辑
        # 可以通过设置标志位或直接终止计算线程
        logging.warning("计算终止请求已接收，正在停止...")
        self.cal_thread.stop()
        # 实际终止逻辑需要根据你的计算实现来添加

    def save_config(self):
        """保存当前配置(留空暂不实现)"""
        logging.info("正在保存当前配置...")

    def load_config(self, preset_name):
        """加载预设参数(留空暂不实现)"""
        logging.info(f"正在加载预设参数: {preset_name}")

    def clear_result(self):
        self.result_display.clear()


class StreamLogger(object):
    """重定向标准输出到日志系统"""

    def __init__(self, log_level):
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            logging.log(self.log_level, line.rstrip())

    def flush(self):
        pass

if __name__ == "__main__":
    app = QApplication([])
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(':/LifeCalor.ico'))
    window = MainWindow()
    window.setWindowIcon(QIcon(':/LifeCalor.ico'))
    window.show()
    app.exec_()