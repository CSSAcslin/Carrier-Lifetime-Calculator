from datetime import datetime
from logging.handlers import RotatingFileHandler
from PyQt5 import sip
from PyQt5.QtGui import QPixmap, QIcon, QFontDatabase
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QScrollArea,
                             QFileDialog, QSlider, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QStackedWidget, QDockWidget,
                             QStatusBar, QScrollBar
                             )
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QMetaObject, QElapsedTimer
from astropy.utils.console import ProgressBar
from fontTools.ttx import process
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from DataProcessor import DataProcessor, MassDataProcessor
from ImageDisplayWindow import *
from LifetimeCalculator import LifetimeCalculator, CalculationThread
from ResultDisplayWidget import ResultDisplayWidget
from ConsoleUtils import *
from ExtraDialog import *
import logging
from ROIdrawDialog import ROIdrawDialog
import resources_rc
from DataManager import *

class MainWindow(QMainWindow):
    """主窗口"""
    # 线程激活信号
    # cal
    start_reg_cal_signal = pyqtSignal(Data, float, tuple, str, int, str)
    start_dis_cal_signal = pyqtSignal(Data, float, str)
    start_dif_cal_signal = pyqtSignal(dict, float, float,float,str)
    # mass
    load_avi_EM_signal = pyqtSignal(str)
    load_tif_EM_signal = pyqtSignal(str)
    pre_process_signal = pyqtSignal(Data,int,bool)
    stft_quality_signal = pyqtSignal(ProcessedData,float, int, int, int, int, str)
    stft_python_signal = pyqtSignal(ProcessedData,float, int, int, int, int, str)
    cwt_quality_signal = pyqtSignal(ProcessedData,float, int, int, str)
    cwt_python_signal = pyqtSignal(ProcessedData,float, int, int, str, float)
    mass_export_signal = pyqtSignal(np.ndarray,str,str,str)
    atam_signal = pyqtSignal(ProcessedData)
    tDgf_signal = pyqtSignal(ProcessedData,int,float,bool)


    def __init__(self):
        super().__init__()

        # 参数初始化
        self.data = None
        self.processed_data = None
        self.time_points = None
        self.time_unit = 1.0
        self.space_unit = 1.0
        self.bool_mask = None
        self.idx = None
        self.vector_array = None

        # 参数初始化
        self.init_params()

        # 界面加载
        self.init_ui()
        self.log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "carrier_lifetime.log")
        self.setup_menus()
        self.setup_logging()
        self.log_startup_message()

        # 进度条与计时器
        self.elapsed_timer = QElapsedTimer()
        self.last_time = 0 # 记录运算的时间
        self.last_progress = 0 # 记录进度
        self.last_percent = -1 # 记录百分比进度
        self.cached_remaining = "计算中..." # 记录剩余时长

        # 状态控制
        self._is_calculating = False
        # 信号连接
        self.signal_connect()

    def init_params(self):
        """初始化参数库"""
        self.main_params = {
            'time_step': 1.000,
            'space_step': 1.000,
            'region_size': 5,
            'bg_nums': 300,
        }
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
            'set_axis': True,
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
        self.EM_params = {
            'EM_fps': 360,
            'target_freq': 30.0,
            'type': None,
            'stft_window_size': 128,  # 加窗大小
            'stft_noverlap': 120,  # 重叠点数
            'stft_window_type': 'hann',
            'stft_total_scales': 10,  # 频率分辨率
            'stft_scale_range': 5,  # 取频率范围（以target frequency为中心）
            'custom_nfft': 360,  # 变换长度，默认为调制频率
            'cwt_type': 'morl',
            'cwt_total_scales': 256,  # 频率处理点数
            'cwt_scale_range': 10.0,  # 取频率范围（以target frequency为中心）
            'scs_thr': 2.5, # 单通道阈值
            'scs_zoom': 2 , # 单通道选区放大倍数
            'thr_known' : False, # 是否已知阈值
        }

    def init_ui(self):
        self.setWindowTitle("成像数据分析工具箱")
        self.setGeometry(100, 50, 1700, 900)

        # 主部件和布局
        # main_widget = QWidget()
        # self.setCentralWidget(main_widget)

        # 左侧设置区域
        self.setup_left_panel()
        self.param_dock = QDockWidget("基础设置", self)
        self.param_dock.setWidget(self.left_panel)
        self.param_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.param_dock.setMinimumSize(300, 700)
        self.param_dock.setMaximumWidth(350)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.param_dock) # 加到左侧

        # 右侧图像区域
        self.image_display = ImageDisplayWindow(self)
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)


        image_layout.addWidget(self.image_display)

        # 时间滑块
        # self.time_slider = QSlider(Qt.Horizontal)
        # self.time_slider.setMinimum(0)
        # self.time_slider.setMaximum(0)
        # self.time_label = QLabel("时间点: 0/0")
        # slider_layout = QHBoxLayout()
        # slider_layout.addWidget(QLabel("时间序列:"))
        # slider_layout.addWidget(self.time_slider)
        # slider_layout.addWidget(self.time_label)
        # image_layout.addLayout(slider_layout)
        self.image_dock = QDockWidget("图像显示", self)
        self.image_dock.setWidget(image_widget)
        self.image_dock.setMinimumSize(700, 600)
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
        self.resizeDocks([self.image_dock, self.result_dock], [800, 600], Qt.Horizontal)
        self.setup_status_bar()

        # 设置控制台
        self.setup_console()

    def setup_left_panel(self):
        """设置左侧面板"""
        button_style_sheet = """
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 10px;
            font-weight: 500;
            min-width: 100px;
        }
        
        QPushButton:hover {
            background-color: #388E3C;
        }
        
        QPushButton:pressed {
            background-color: #2E7D32;
        }
        
        QPushButton:disabled {
            background-color: #A5D6A7;
            color: #E8F5E9;
        }
        
        QPushButton:focus {
            outline: 1px solid #81C784;
            border-radius: 4px;
            padding: 5px 10px;
            outline-offset: 1px;
        }"""
        self.left_panel = QWidget()
        self.left_panel_layout = QVBoxLayout()
        self.left_panel_layout.setContentsMargins(15,15,15,15)

    # 数据导入面板
        self.data_import = self.QGroupBoxCreator('数据导入')
        left_layout0 = QVBoxLayout()
        left_layout0.setSpacing(2)

        # 模式选择
        self.fuction_select = QComboBox()
        self.fuction_select.addItems(['请选择分析模式','超快成像动态分析','EM-iSCAT','科学分析（Simulation）'])
        left_layout0.addWidget(self.fuction_select)
        self.funtion_stack = QStackedWidget()
        nothing_group = self.QGroupBoxCreator(style="inner")
        nothing_layout = QVBoxLayout()
        nothing_layout.addWidget(QLabel("首先：请选择分析模式!"))
        nothing_group.setLayout(nothing_layout)
        self.funtion_stack.addWidget(nothing_group)

        # FS&PA模式下的文件夹选择
        fs_iSCAT_group = self.QGroupBoxCreator(style="inner")
        type_choose1 = QVBoxLayout()
        self.file_type_selector1 = QComboBox()
        self.file_type_selector1.addItems(['tif格式', 'sif格式'])
        self.file_type_stack1 = QStackedWidget()
        fs_group = self.QGroupBoxCreator(style="noborder") # tif
        tiff_layout = QHBoxLayout()
        self.group_selector = QComboBox()
        self.group_selector.addItems(['n', 'p', '不区分'])
        self.tiff_folder_btn = QPushButton("选择TIFF文件夹")
        tiff_layout.addWidget(self.group_selector)
        tiff_layout.addWidget(self.tiff_folder_btn)
        fs_group.setLayout(tiff_layout)
        self.file_type_stack1.addWidget(fs_group)
        # 光热信号处理模式下的文件夹选择
        PA_group = self.QGroupBoxCreator(style="noborder") # sif
        sif_layout = QVBoxLayout()
        sif_layout_inner = QHBoxLayout()
        method_label = QLabel("归一化方法:")         # 归一化方法选择
        self.method_combo = QComboBox()
        self.method_combo.addItems(["linear", "percentile", "sigmoid", "log", "clahe"])
        self.sif_folder_btn = QPushButton('选择SIF文件夹')
        sif_layout_inner.addWidget(method_label)
        sif_layout_inner.addWidget(self.method_combo)
        sif_layout.addLayout(sif_layout_inner)
        sif_layout.addWidget(self.sif_folder_btn)
        PA_group.setLayout(sif_layout)
        self.file_type_stack1.addWidget(PA_group)
        type_choose1.addWidget(self.file_type_selector1)
        type_choose1.addWidget(self.file_type_stack1)
        fs_iSCAT_group.setLayout(type_choose1)
        self.funtion_stack.addWidget(fs_iSCAT_group)

        # 文件类型为tiff
        EM_iSCAT_group = self.QGroupBoxCreator(style="inner")
        type_choose = QHBoxLayout()
        file_types = QVBoxLayout()
        self.file_type_selector = QComboBox()
        self.file_type_selector.addItems(['avi格式', 'tiff格式'])
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

        # 科学分析模块
        Sim_group = self.QGroupBoxCreator(style="inner")
        sim_layout = QVBoxLayout()
        self.text_box = QTextEdit()
        self.text_box.setPlaceholderText("输入Python代码或拖入.py文件")
        self.text_box.setMaximumHeight(40)
        self.text_box.setMinimumHeight(20)
        sim_layout.addWidget(self.text_box)
        self.code_button = QPushButton('执行代码')
        sim_layout.addWidget(self.code_button)
        Sim_group.setLayout(sim_layout)
        self.funtion_stack.addWidget(Sim_group)

        # 总提示
        # self.folder_path_label = QLabel("未选择文件夹")
        # self.folder_path_label.setMaximumWidth(300)
        # self.folder_path_label.setWordWrap(True)
        # # self.folder_path_label.setStyleSheet("font-size: 14px;")  # 后续还要改

        left_layout0.addWidget(self.funtion_stack)
        # left_layout0.addSpacing(3)
        # left_layout0.addWidget(self.folder_path_label)
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
        self.time_step_input.setValue(self.main_params['time_step'])
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
        self.space_step_input.setValue(self.main_params['space_step'])
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
        self.modes_panel = self.QGroupBoxCreator("分析设置")
        left_layout1 = QVBoxLayout()
        left_layout1.setContentsMargins(1, 0, 1, 0)
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
        self.FS_mode_combo.addItems(["选区寿命热图", "指数衰减寿命曲线","载流子扩散系数计算"])
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
        self.region_size_input.setValue(self.main_params['region_size'])
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

    # EM_iSCAT下的功能选择
        EM_iSCAT_GROUP = self.QGroupBoxCreator(style="noborder")
        EM_iSCAT_layout = QVBoxLayout()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # 关键设置
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll_content = QWidget()
        scroll_content.setStyleSheet(""" QWidget {background-color: white; }""")
        EM_iSCAT_layout1 = QVBoxLayout(scroll_content)
        preprocess_set_layout = QHBoxLayout()
        preprocess_set_layout.addWidget(QLabel("背景帧数："))
        self.bg_nums_input = QSpinBox()
        self.bg_nums_input.setMinimum(1)
        self.bg_nums_input.setMaximum(9999)
        self.bg_nums_input.setValue(self.main_params['bg_nums'])
        preprocess_set_layout.addWidget(self.bg_nums_input)
        self.preprocess_data_btn = QPushButton("数据预处理")
        preprocess_set_layout2 = QHBoxLayout()
        preprocess_set_layout2.addWidget(QLabel("是否显示结果："))
        self.show_stft_check = QCheckBox()
        self.show_stft_check.setChecked(False)
        preprocess_set_layout2.addWidget(self.show_stft_check,alignment=Qt.AlignRight)
        EM_iSCAT_layout1.addLayout(preprocess_set_layout)
        EM_iSCAT_layout1.addWidget(self.preprocess_data_btn)
        EM_iSCAT_layout1.addSpacing(4)
        EM_iSCAT_layout1.addLayout(preprocess_set_layout2)
        EM_iSCAT_layout1.addSpacing(4)
        process_set_layout = QHBoxLayout()
        self.EM_mode_combo = QComboBox()
        self.EM_mode_combo.addItems(["stft短时傅里叶","cwt连续小波变换"])
        process_set_layout.addWidget(QLabel("变换方法："))
        process_set_layout.addWidget(self.EM_mode_combo)
        EM_iSCAT_layout1.addLayout(process_set_layout)
        self.EM_mode_stack = QStackedWidget()
        # stft 短时傅里叶变换
        stft_GROUP = self.QGroupBoxCreator(style="inner")
        stft_layout = QVBoxLayout()
        stft_GROUP.setLayout(stft_layout)
        process_set_layout1 = QHBoxLayout()
        process_set_layout1.addWidget(QLabel("处理方法："))
        self.stft_program_select = QComboBox()
        self.stft_program_select.addItems(["python", "julia（未实现）"])
        process_set_layout1.addWidget(self.stft_program_select)
        self.stft_window_select = QComboBox()
        self.stft_window_select.addItems(["汉宁窗(hann)", "汉明窗(hanming)","gabor变换(gaussian)","矩形窗","blackman",'blackman-harris'])
        process_set_layout2 = QHBoxLayout()
        process_set_layout2.addWidget(QLabel("窗选择："))
        process_set_layout2.addWidget(self.stft_window_select)
        self.stft_quality_btn = QPushButton("stft质量评价（功率密度谱）")
        self.stft_process_btn = QPushButton("执行短时傅里叶变换")
        stft_layout.addLayout(process_set_layout1)
        stft_layout.addLayout(process_set_layout2)
        stft_layout.addWidget(self.stft_quality_btn)
        stft_layout.addWidget(self.stft_process_btn)
        self.EM_mode_stack.addWidget(stft_GROUP)
        # cwt 小波变换
        cwt_GROUP = self.QGroupBoxCreator(style='inner')
        cwt_layout = QVBoxLayout()
        cwt_GROUP.setLayout(cwt_layout)
        cwt_set_layout1 = QHBoxLayout()
        cwt_set_layout1.addWidget(QLabel("处理方法："))
        self.cwt_program_select = QComboBox()
        self.cwt_program_select.addItems(["python","julia"])
        cwt_set_layout1.addWidget(self.cwt_program_select)
        self.cwt_quality_btn = QPushButton("cwt质量检验（功率谱）")
        self.cwt_process_btn = QPushButton("执行连续小波变换")
        cwt_layout.addLayout(cwt_set_layout1)
        cwt_layout.addWidget(self.cwt_quality_btn)
        cwt_layout.addWidget(self.cwt_process_btn)
        self.EM_mode_stack.addWidget(cwt_GROUP)
        EM_iSCAT_layout1.addWidget(self.EM_mode_stack)

        EM_iSCAT_layout2 = QHBoxLayout()
        self.after_process_select = QComboBox()
        self.after_process_select.addItems(["全细胞电生理分析","单通道电生理分析"])
        EM_iSCAT_layout2.addWidget(QLabel("后处理方法："))
        EM_iSCAT_layout2.addWidget(self.after_process_select)
        self.after_process_stack = QStackedWidget()
        whole_cell_GROUP = self.QGroupBoxCreator(style='inner')
        whole_cell_layout = QVBoxLayout()
        self.EM_output_btn = QPushButton("时频变换结果导出")
        self.roi_signal_btn = QPushButton("选区信号均值变化")
        whole_cell_layout.addWidget(self.EM_output_btn)
        whole_cell_layout.addWidget(self.roi_signal_btn)
        whole_cell_GROUP.setLayout(whole_cell_layout)
        self.after_process_stack.addWidget(whole_cell_GROUP)

        single_channel_GROUP = self.QGroupBoxCreator(style='inner')
        single_channel_layout = QVBoxLayout()
        self.atam_btn = QPushButton("累计时间振幅图")
        self.tDgf_btn = QPushButton("选区二维高斯拟合")
        single_channel_layout.addWidget(self.atam_btn)
        single_channel_layout.addWidget(self.tDgf_btn)
        single_channel_GROUP.setLayout(single_channel_layout)
        self.after_process_stack.addWidget(single_channel_GROUP)

        EM_iSCAT_layout1.addLayout(EM_iSCAT_layout2)
        EM_iSCAT_layout1.addWidget(self.after_process_stack)

        EM_iSCAT_layout1.addStretch(1)
        scroll_area.setWidget(scroll_content)
        EM_iSCAT_layout.addWidget(scroll_area)
        EM_iSCAT_GROUP.setLayout(EM_iSCAT_layout)
        self.between_stack.addWidget(EM_iSCAT_GROUP)

        left_layout1.addWidget(self.between_stack)
        # left_layout1.addStretch(1)
        self.modes_panel.setLayout(left_layout1)

        # 添加分析按钮和导出按钮
        data_save_layout = QHBoxLayout()
        self.export_image_btn = QPushButton("导出结果为图片")
        self.export_data_btn = QPushButton("导出结果为数据")
        self.export_data_btn.setStyleSheet(button_style_sheet)
        self.export_image_btn.setStyleSheet(button_style_sheet)
        data_save_layout.addWidget(self.export_image_btn)
        data_save_layout.addWidget(self.export_data_btn)

        self.left_panel_layout.addWidget(self.data_import)
        self.left_panel_layout.addWidget(self.parameter_panel)
        self.left_panel_layout.addWidget(self.modes_panel, stretch=1)
        self.left_panel_layout.addSpacing(15)
        self.left_panel_layout.addLayout(data_save_layout)
        self.left_panel.setLayout(self.left_panel_layout)

    def between_stack_change(self):
        if self.fuction_select.currentIndex() == 0: # nothing
            self.between_stack.setCurrentIndex(0)
            self.FS_mode_combo.setCurrentIndex(0)
        if self.fuction_select.currentIndex() == 1:  # FS-iSCAT & PA
            self.between_stack.setCurrentIndex(1)
            self.FS_mode_combo.setCurrentIndex(1)
            self.cal_thread_open()
            self.stop_thread(type=1)
            self.update_status('准备就绪')
        if self.fuction_select.currentIndex() == 2:  # ES-iSCAT
            self.between_stack.setCurrentIndex(2)
            self.EM_thread_open()
            self.stop_thread(type=0)
            self.update_status('准备就绪')

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

        # 编辑菜单-坏点处理功能
        bad_frame_edit = edit_menu.addAction("坏点处理")
        bad_frame_edit.triggered.connect(self.bad_frame_edit_dialog)

        # 编辑菜单-计算设置功能
        data_select_edit = edit_menu.addAction("计算设置")
        data_select_edit.triggered.connect(self.calculation_set_edit_dialog)

        # 编辑菜单-绘图设置调整
        plt_settings_edit = edit_menu.addAction("绘图设置")
        plt_settings_edit.triggered.connect(self.plt_settings_edit_dialog)

        # ROI绘图
        ROI_function = self.menu.addAction("ROI选取")
        ROI_function.triggered.connect(self.roi_select_dialog)

        # 历史数据管理
        data_menu = self.menu.addMenu('历史数据')
        # 清除历史
        data_history_clear = data_menu.addAction('历史清除')
        data_history_clear.triggered.connect(self.data_history_clear)
        # 数据导入历史查看
        data_history_view = data_menu.addAction('历史导入查看')
        data_history_view.triggered.connect(self.data_history_view) # 临时
        # 数据处理历史查看
        process_history_view = data_menu.addAction('历史处理查看')
        process_history_view.triggered.connect(self.process_history_view)


    @staticmethod
    def QGroupBoxCreator(title="",style="default"):
        # 全局Box样式定义
        group_box = QGroupBox(title)
        styles = {
            "default": """
            QGroupBox{
                border:1px solid #aaaaaa;
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
                color: #2E7D32;
                font-weight: 1000;
            }
            """,
            "inner":"""
            QGroupBox{
                border: 1px solid #aaaaaa;
                border-radius: 5px;
                margin-top: 5px;
                padding: 5px;
                padding-left: 0px;
                padding-right: 0px;
            }""",
            "noborder":"""
            QGroupBox{
                border: 0px;
                border-radius: 0px;
                margin: 0px;
                padding:0px;
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
        self.mouse_pos_label = QLabel("光标位置: x= -, y= -, t= -; 值: -")
        self.mouse_pos_label.setFixedWidth(500)
        self.status_bar.addWidget(self.mouse_pos_label)
        self._handle_hover = self.make_hover_handler()
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedWidth(650)
        self.status_bar.addWidget(self.progress_bar)

        # 状态指示灯 (红绿灯)
        self.status_light = QLabel()
        self.status_light.setPixmap(QPixmap(":/icons/green_light.png").scaled(16, 16))
        self.status_bar.addPermanentWidget(self.status_light)

    def update_status(self, status, working_status='idle'):
        """更新状态条的显示"""
        self.status_label.setText(status)
        if working_status == 'idle' : # idle
            light = "green_light.png"
        elif working_status == 'working' :
            light = "yellow_light.png"
        elif working_status == 'warning':
            light = "red_light.png"
            logging.warning(status)
        elif working_status == 'error':
            QMessageBox.warning(self,'错误！',status)
            logging.error(status)
            light = "green_light.png"
        else:
            light = "red_light.png"
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
成像数据分析工具箱启动
启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
日志位置: {self.log_file}
程序版本: 1.9.6
============================================
        """
        logging.info(startup_msg.strip())
        logging.info("程序已进入准备状态，等待用户操作...（第一次计算可能较慢）")

    def cal_thread_open(self):
        """计算线程相关 以及信号槽连接都放在这里了"""
        self.calc_thread = QThread()
        self.cal_thread = CalculationThread()
        self.cal_thread.moveToThread(self.calc_thread)
        # 计算状态更新
        self.start_reg_cal_signal.connect(self.cal_thread.region_analyze)
        self.start_dis_cal_signal.connect(self.cal_thread.distribution_analyze)
        self.start_dif_cal_signal.connect(self.cal_thread.diffusion_calculation)
        self.cal_thread.calculating_progress_signal.connect(self.update_progress)
        self.cal_thread.processed_result.connect(self.processed_result)
        self.cal_thread.stop_thread_signal.connect(self.stop_thread)
        self.cal_thread.cal_running_status.connect(self.btn_safety)
        self.cal_thread.update_status.connect(self.update_status)

    def signal_connect(self):
        # 连接参数区域按钮
        self.fuction_select.currentIndexChanged.connect(self.funtion_stack.setCurrentIndex)
        self.fuction_select.currentIndexChanged.connect(self.between_stack_change)
        self.file_type_selector1.currentIndexChanged.connect(self.file_type_stack1.setCurrentIndex)
        self.file_type_selector.currentIndexChanged.connect(self.file_type_stack.setCurrentIndex)
        self.tiff_folder_btn.clicked.connect(self.load_tiff_folder)
        self.sif_folder_btn.clicked.connect(self.load_sif_folder)
        self.avi_select_btn.clicked.connect(self.load_avi)
        self.EMtiff_folder_btn.clicked.connect(self.load_tiff_folder_EM)
        self.analyze_region_btn.clicked.connect(self.region_analyze_start)
        self.analyze_btn.clicked.connect(self.distribution_analyze_start)
        self.FS_mode_combo.currentIndexChanged.connect(self.FS_mode_stack.setCurrentIndex)
        # self.PA_mode_combo.currentIndexChanged.connect(self.PA_mode_stack.setCurrentIndex)
        self.EM_mode_combo.currentIndexChanged.connect(self.EM_mode_stack.setCurrentIndex)
        self.preprocess_data_btn.clicked.connect(self.pre_process_EM)
        self.stft_process_btn.clicked.connect(self.process_EM_stft)
        self.stft_quality_btn.clicked.connect(self.quality_EM_stft)
        self.cwt_quality_btn.clicked.connect(self.quality_EM_cwt)
        self.cwt_process_btn.clicked.connect(self.process_EM_cwt)
        self.EM_output_btn.clicked.connect(self.export_EM_data)
        self.atam_btn.clicked.connect(self.process_atam)
        self.tDgf_btn.clicked.connect(self.process_tDgf)
        self.roi_signal_btn.clicked.connect(self.roi_signal_avg)
        self.vector_signal_btn.clicked.connect(self.vectorROI_signal_show)
        self.select_frames_btn.clicked.connect(self.vectorROI_selection)
        self.diffusion_coefficient_btn.clicked.connect(self.result_display.plot_variance_evolution)
        self.export_image_btn.clicked.connect(self.export_image)
        self.export_data_btn.clicked.connect(self.export_data)
        # 鼠标移动
        # self.image_display.mouse_position_signal.connect(self._handle_hover)
        # self.image_display.mouse_clicked_signal.connect(self._handle_click)
        self.image_display.add_canvas_signal.connect(self.add_new_canvas)
        # 时间滑块
        # self.time_slider.valueChanged.connect(self.image_display.update_time_slice)
        self.time_slider_vertical.valueChanged.connect(self.update_result_display)
        # 连接控制台信号
        self.command_processor.terminate_requested.connect(self.stop_calculation)
        self.command_processor.save_config_requested.connect(self.save_config)
        self.command_processor.load_config_requested.connect(self.load_config)
        self.command_processor.clear_result_requested.connect(self.clear_result)
        # 结果区域信号
        self.result_display.tab_type_changed.connect(self._handle_result_tab)

    def canvas_signal_connect(self):
        for canvas in self.image_display.display_canvas:
            canvas.mouse_position_signal.connect(self._handle_hover)
            canvas.mouse_clicked_signal.connect(self._handle_click)
            canvas.current_canvas_signal.connect(self.image_display.set_cursor_id)
            canvas.draw_result_signal.connect(self.draw_result)

    '''上面是初始化预设，下面是功能响应'''
    def add_new_canvas(self, assign_data = None):
        """新建图像显示画布"""
        if self.data is None and self.processed_data is None:
            logging.warning('请先导入或处理数据')
            return
        # if self.image_display is []:  走不到这里
        #     logging.warning("请先导入数据")

        dialog = DataViewAndSelectPop(datadict=self.get_data_all(), processed_datadict=self.get_processed_data_all(),add_canvas=True)
        if dialog.exec_():
            selected_timestamp = dialog.get_selected_timestamp()
            data_display = None
            if self.data is not None:
                for data in self.data.history:
                    if data.timestamp == selected_timestamp:
                        data_display = ImagingData.create_image(data)
                        logging.info("数据选择成功（原初）")
                        continue
            if self.processed_data is not None:
                for data in self.processed_data.history:
                    if data.timestamp == selected_timestamp:
                        data_display = ImagingData.create_image(data)
                        logging.info("数据选择成功（处理）")
                        continue
            if data_display is not None:
                self.image_display.add_canvas(data_display)
            else:
                QMessageBox.warning(self,"数据错误","数据已经遗失（不可能错误）")
                return

        self.canvas_signal_connect()
        # self.image_display.update_time_slice(0, True)

        self.update_status("准备就绪", 'idle')

    def get_data_all(self) ->  List[Dict[str, Any]]:
        Data_list = []
        if self.data is None:
            return []
        # 直接读取历史数据
        for data in self.data.history:
            Data_list.append({
                "type": 'Data',
                "name": data.name,
                "序号": data.serial_number,
                "导入格式": data.format_import,
                "数据大小": data.datashape,
                "timestamp": data.timestamp,
            })
            Data_list.reverse()
        return Data_list

    def get_processed_data_all(self) ->  List[Dict[str, Any]]:
        ProcessedData_list = []
        if self.processed_data is None:
            return []
        # 直接读取历史数据
        for processed in self.processed_data.history:
                ProcessedData_list.append({
                    "type": "ProcessedData",
                    "name": processed.name,
                    "处理类型": processed.type_processed,
                    "数据大小": processed.datashape,
                    "数据源": self._find_parent_name(processed.timestamp_inherited),
                    "timestamp": processed.timestamp,
                })
        ProcessedData_list.reverse()
        return ProcessedData_list

    def _find_parent_name(self, timestamp: float) -> Optional[str]:
        """通过时间戳查找父数据名称"""
        # 首先在原始数据中查找
        for data in list(self.data.history):
            if data.timestamp == timestamp:
                return data.name

        # 然后在处理数据中查找
        for processed in list(self.processed_data.history):
            if processed.timestamp == timestamp:
                return processed.name

        return None

    def load_image(self,data_type = 'original',other_params:str = None):
        """图像加载，后面会进一步修改"""
        if len(self.image_display.display_canvas) == 0 : # 初次创建
            # self.add_new_canvas()
            if data_type == 'original':
                self.imaging_main = ImagingData.create_image(self.data)
            self.image_display.add_canvas(self.imaging_main)
            totalframes = self.imaging_main.totalframes
            # self.time_slider.setMaximum(totalframes - 1)
            # self.time_label.setText(f"时间点: 0/{totalframes - 1}")
            self.canvas_signal_connect()
        else:
            if data_type == 'original':
                # imports_done = self.other_imports
                msg_box = QMessageBox()
                msg_box.setWindowTitle("画布操作")
                msg_box.setText("请选择是否要覆盖当前画布或新建画布")

                # 添加标准按钮
                overwrite_btn = msg_box.addButton("覆盖", QMessageBox.ActionRole)
                new_btn = msg_box.addButton("新建", QMessageBox.ActionRole)
                hide_btn = msg_box.addButton("隐藏", QMessageBox.ActionRole)
                msg_box.exec_()

                # 返回结果
                if msg_box.clickedButton() == overwrite_btn:
                    self.image_display.del_canvas(-1)
                    self.imaging_main = ImagingData.create_image(self.data)
                    totalframes = self.imaging_main.totalframes
                    # self.time_slider.setMaximum(totalframes - 1)
                    # self.time_label.setText(f"时间点: 0/{totalframes - 1}")
                    self.add_new_canvas()
                elif msg_box.clickedButton() == new_btn:
                    self.add_new_canvas('latest')
                elif msg_box.clickedButton() == hide_btn:
                    return False

        # 显示第一张图像
        # self.image_display.update_time_slice(0, True)
        # self.time_slider.setValue(0)

        # 根据图像大小调节region范围
        self.region_x_input.setMaximum(self.data.datashape[1])
        self.region_y_input.setMaximum(self.data.datashape[2])

    def other_imports(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("画布操作")
        msg_box.setText("请选择是否要覆盖当前画布或新建画布")

        # 添加标准按钮
        overwrite_btn = msg_box.addButton("覆盖", QMessageBox.ActionRole)
        new_btn = msg_box.addButton("新建", QMessageBox.ActionRole)
        hide_btn = msg_box.addButton("隐藏", QMessageBox.ActionRole)
        msg_box.exec_()

        # 返回结果
        if msg_box.clickedButton() == overwrite_btn:
            return "overwrite"
        elif msg_box.clickedButton() == new_btn:
            self.add_new_canvas('latest')
            return "new"
        elif msg_box.clickedButton() == hide_btn:
            return "hide"
        return None

    def load_tiff_folder(self):
        """加载TIFF文件夹(FS-iSCAT)"""
        self.time_unit = float(self.time_step_input.value())
        folder_path = QFileDialog.getExistingDirectory(self, "选择TIFF图像文件夹")
        self.data_processor = DataProcessor(folder_path)
        if folder_path:
            logging.info(folder_path)
            self.update_status("已加载TIFF文件夹",'idle')
            current_group = self.group_selector.currentText()

            # 读取文件夹中的所有tiff文件
            tiff_files = self.data_processor.load_and_sort_tiff(current_group)

            if not tiff_files or tiff_files == []:
                self.update_status("文件夹中没有目标TIFF文件",'warning')
                QMessageBox.warning(self,"数据导入","文件夹中没有目标TIFF文件")
                return

            # 读取所有图像
            self.data = self.data_processor.process_tiff(tiff_files)

            if not self.data or (self.data.format_import != 'tif' and self.data.format_import != 'tiff'):
                self.update_status("无法读取TIFF文件",'warning')
                return

            logging.info('成功加载TIFF数据')
            # 设置时间滑块
            self.load_image()

    def load_sif_folder(self):
        '''加载SIF文件夹'''
        folder_path = QFileDialog.getExistingDirectory(self, "选择SIF图像文件夹")
        self.data_processor = DataProcessor(folder_path,self.method_combo.currentText())
        if folder_path:
            logging.info(folder_path)
            self.update_status("已加载SIF文件夹",'idle')

            # 读取文件夹中的所有sif文件
            check_sif = self.data_processor.load_and_sort_sif()

            if not check_sif:
                self.update_status("文件夹中没有目标SIF文件",'warning')
                QMessageBox.warning(self,'数据导入',"文件夹中没有目标SIF文件,请确认选择的文件格式是否匹配")
                return

            # 读取所有图像
            self.data = self.data_processor.process_sif()

            if not self.data or self.data.format_import != 'sif':
                self.update_status("无法读取sif文件",'warning')
                return

            logging.info('成功加载SIF数据')
            # 设置时间滑块
            self.load_image()
        pass

    def EM_thread_open(self):
        """加载EM文件的线程开启"""
        # 初始化数据处理线程

        self.avi_thread = QThread()
        self.mass_data_processor = MassDataProcessor()
        self.mass_data_processor.moveToThread(self.avi_thread)

        self.mass_data_processor.mass_finished.connect(self.loaded_EM)
        self.mass_data_processor.processing_progress_signal.connect(self.update_progress)
        self.mass_data_processor.processed_result.connect(self.processed_result)
        self.load_avi_EM_signal.connect(self.mass_data_processor.load_avi)
        self.load_tif_EM_signal.connect(self.mass_data_processor.load_tiff)
        self.pre_process_signal.connect(self.mass_data_processor.pre_process)
        self.stft_python_signal.connect(self.mass_data_processor.python_stft)
        self.stft_quality_signal.connect(self.mass_data_processor.quality_stft)
        self.cwt_quality_signal.connect(self.mass_data_processor.quality_cwt)
        self.cwt_python_signal.connect(self.mass_data_processor.python_cwt)
        self.mass_export_signal.connect(self.mass_data_processor.export_EM_data)
        self.atam_signal.connect(self.mass_data_processor.accumulate_amplitude)
        self.tDgf_signal.connect(self.mass_data_processor.twoD_gaussian_fit)

        # self.avi_thread.start()

    def load_avi(self):
        """加载avi读取线程传递函数"""
        self.status_label.setText("正在处理数据...")
        self.avi_thread.start()
        file_types = "AVI视频文件 (*.avi);;所有文件 (*)"

        # 获取文件路径
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择AVI视频文件",
            "",  # 起始目录
            file_types
        )

        if not file_path:
            # self.folder_path_label.setText("未选择文件")
            logging.info("用户取消选择")
            return

        logging.info(f"已选择AVI文件: {file_path}")
        self.update_status("正在加载AVI文件...",'working')

        self.load_avi_EM_signal.emit(file_path)

    def load_tiff_folder_EM(self):
        """加载TIFF文件夹(FS-iSCAT)"""
        self.status_label.setText("正在处理数据...")
        self.avi_thread.start()
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "选择TIFF图像序列文件夹",
            ""
        )

        if not folder_path:
            # self.folder_path_label.setText("未选择文件夹")
            logging.info("用户取消选择")
            return

        for f in os.listdir(folder_path):
            if f.lower().endswith(('.tif', '.tiff')):
                self.load_tif_EM_signal.emit(folder_path)
                self.update_status("正在加载tiff文件...",'working')
                return
            else:
                self.update_status("文件夹中没有TIFF文件",'warning')
                return

    def loaded_EM(self, result):
        self.data = result

        if not self.data or not (self.data.format_import != 'avi' or self.data.format_import != 'tiff'):
            self.update_status("无法读取文件",'warning')
            return
        else:
            self.update_status("已加载文件",'idle')
            logging.info("文件加载成功")
        # 设置时间滑块
        self.load_image()

    def make_hover_handler(self):
        args = {'x': None, 'y': None, 't': None, 'value': None, 'origin': None}
        def _handle_hover(x=None, y=None, t=None, value=None, origin=None):
            """鼠标位置显示"""
            # 更新传入的参数（未传入的保持原值）
            if x is not None: args['x'] = x
            if y is not None: args['y'] = y
            if t is not None: args['t'] = t
            if value is not None:
                args['value'] = value
            else:
                args['value'] = self.data.image_import[args['t'], args['y'], args['x']]
            if args['x'] is None or args['y'] is None:
                return
            if origin is not None: args['origin'] = origin

            # 更新显示
            self.mouse_pos_label.setText(
                f"光标位置: x={args['x']}, y={args['y']}, t={args['t']}; 归一值: {args['value']}, 原始值：{args['origin']:.3f}")

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
            self.time_slider_vertical.setMaximum(self.data.timelength - 1)
            self.time_slider_vertical.setValue(0)
        elif tab_type == 'pre-scs':
            self.time_slider_vertical.setVisible(True)
            self.time_slider_vertical.setMaximum(int(self.processed_data.out_processed['max_signal'].max()*10))
        else:
            if self.time_slider_vertical.isVisible():
                self.time_slider_vertical.setVisible(False)

    def bad_frame_edit_dialog(self):
        """显示坏点处理对话框"""
        if self.data is None or self.processed_data is None:
            logging.warning("无数据，请先加载数据文件")
            return

        dialog = BadFrameDialog(self)
        self.update_status("坏点修复ing", 'working')
        if dialog.exec_():
            # 更新图像显示
            # self.time_label.setText(self.image_display.update_time_slice(0))
            # self.time_slider.setValue(0)
            logging.info(f"坏点处理完成，修复了 {len(dialog.bad_frames)} 个坏帧")
        self.update_status("准备就绪", 'idle')

    def calculation_set_edit_dialog(self):
        """计算设置调整"""
        if self.data is None or self.processed_data is None:
            logging.warning("无数据，请先加载数据文件")
            return
        self.update_status("计算设置ing", 'working')
        dialog = CalculationSetDialog(self.cal_set_params)
        if dialog.exec_():
            # self.time_label.setText(self.image_display.update_time_slice(0))
            # self.time_slider.setValue(0)
            self.cal_set_params = dialog.params
            LifetimeCalculator.set_cal_parameters(self.cal_set_params)
            # 同步修改绘图设置并传参
            self.plot_params['_from_start_cal'] = self.cal_set_params['from_start_cal']
            self.result_display.update_plot_settings(self.plot_params, update=False)
            logging.info("设置已更新，请重新绘图")
        self.update_status("准备就绪", 'idle')

    def plt_settings_edit_dialog(self):
        """绘图设置"""
        dialog = PltSettingsDialog(params=self.plot_params)
        self.update_status("绘图设置ing", 'working')
        if dialog.exec_():
            # 将参数传递给ResultDisplayWidget
            self.result_display.update_plot_settings(dialog.params)
            self.plot_params = dialog.params
            logging.info("绘图已更新")
        self.update_status("准备就绪", 'idle')

    def roi_select_dialog(self):
        """ROI选取功能"""
        if self.data is None or self.processed_data is None:
            logging.warning("无数据，请先加载数据文件")
            return
        roi_dialog = ROIdrawDialog(base_layer_array=self.data.data_origin[0],parent=self)
        self.update_status("ROI绘制ing", 'working')
        if roi_dialog.exec_() == QDialog.Accepted:
            if roi_dialog.action_type == "mask":
                self.mask, self.bool_mask = roi_dialog.get_top_layer_array()
                logging.info(f'成功绘制ROI，大小{self.mask.shape[0]}×{self.mask.shape[1]}')
                if self.fuction_select.currentIndex() == 3:
                    pass
                else:
                    data_amend = self.data_processor.amend_data(self.data, self.bool_mask)
                    self.data.update(data_amend)
                    # self.time_label.setText(self.image_display.update_time_slice(self.idx))
            elif roi_dialog.action_type == "vector":
                self.vector_array = roi_dialog.vector_line.getPixelValues(self.data,self.space_unit,self.time_unit)
                logging.info(f'成功绘制ROI，大小{self.vector_array.shape}')


        self.update_status("准备就绪", 'idle')

    def start_calculation(self):
        """开始计算时调用此方法"""
        self.elapsed_timer.start()
        self.last_time = 0
        self.last_progress = 0
        self.last_percent = -1
        self.cached_remaining = "计算中..."

    def update_progress(self, current, total=None):
        """更新进度条"""
        if total is not None:
            self.progress_bar.setMaximum(total)

        self.progress_bar.setValue(current)
        # 计算当前进度
        current_percent = current / self.progress_bar.maximum() * 100 if self.progress_bar.maximum() > 0 else 0
        elapsed_ms = self.elapsed_timer.elapsed()
        elapsed_sec = elapsed_ms / 1000.0

        # 只有当进度变化超过1%时才更新剩余时间
        if int(current_percent) > self.last_percent:

            # 计算剩余时间（仅当进度变化超过1%时）
            if current > self.last_progress and elapsed_ms > self.last_time:
                # 计算速度
                progress_diff = current - self.last_progress
                time_diff = (elapsed_ms - self.last_time) / 1000.0
                speed = progress_diff / time_diff if time_diff > 0 else 0

                # 更新记录点
                self.last_progress = current
                self.last_time = elapsed_ms

                # 计算并缓存剩余时间
                if speed > 0 and total is not None:
                    remaining_sec = (total - current) / speed
                    self.cached_remaining = self.format_time(remaining_sec)

            # 更新百分比记录
            self.last_percent = int(current_percent)

        # 格式化时间显示
        elapsed_str = self.format_time(elapsed_sec)

        # 更新进度条格式
        self.progress_bar.setFormat(
            f"进度: {current}/{self.progress_bar.maximum()} "
            f"({current_percent:.1f}%) | "
            f"已用: {elapsed_str} | 预计剩余: {self.cached_remaining}"
        )

        # self.console_widget.update_progress(current, total)

        if current == 1:
            self.start_calculation() # 启动计时器
        elif current >= self.progress_bar.maximum():
            logging.info(f"计算完成，总耗时{elapsed_str}")
            self.update_status("进程任务完成,准备就绪",'idle')
            self.progress_bar.reset()
        elif current == -1:
            self.progress_bar.reset()

    def vectorROI_signal_show(self):
        """向量选取信号全部展示"""
        if not hasattr(self, 'data') or self.data is None:
            logging.warning("无数据，请先加载数据文件")
            return
        if self.vector_array is None :
            logging.warning("未选取向量直线ROI")
            return
        elif self.data.timelength == self.vector_array.shape[0]:
            # self.time_slider_vertical.setVisible(True)
            # self.time_slider_vertical.setMaximum(self.data['data_origin'].shape[0] - 1)
            # self.time_slider_vertical.setValue(0)
            self.update_result_display(0,reuse_current = False)
            return
        else:
            logging.error("数据长度不匹配")
            return

    def update_result_display(self,idx,reuse_current=True):
        """目前有两个地方用到垂直滚动条"""
        data = self.processed_data
        if self.vector_array is not None and 0 <= idx < self.vector_array.shape[0]:
            frame_data = self.vector_array[idx]
            self.result_display.display_roi_series(
                positions=frame_data[:, 0],
                intensities=frame_data[:, 1],
                fig_title=f"ROI信号强度 (帧:{idx})",
                reuse_current = reuse_current

            )
        elif self.processed_data.type_processed == 'Single_channel_signal' and not data.out_processed['thr_known']:
            thr = idx/10
            for m in range(self.processed_data.framesize):
                if data.out_processed['max_signal'][m] > thr:
                    data.data_processed[m] = data.out_processed['amplitudes'][m]
                else:
                    data.data_processed[m] = data.out_processed['mean_signal'][m]
            data.out_processed['thr'] = thr
            self.result_display.single_channel(data,False,
                reuse_current = reuse_current)
        else:
            logging.debug("结果垂直滚动条失去更新源，不可能错误")

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
        # 如果线程没了，要开启
        if not self.is_thread_active("calc_thread"):
            self.cal_thread_open()
        # 如果有线程在运算，要提示（不过目前不需要，保留语句）
        if self.cal_thread and self.calc_thread.isRunning():
            logging.warning("已有计算任务正在运行")
            return

        self.calc_thread.start()
        self.update_status('计算进行中...', 'working')
        self.time_unit = float(self.time_step_input.value())
        center = (self.region_y_input.value(), self.region_x_input.value())
        shape = 'square' if self.region_shape_combo.currentText() == "正方形" else 'circle'
        size = self.region_size_input.value()
        model_type = 'single' if self.model_combo.currentText() == "单指数衰减" else 'double'
        self.start_reg_cal_signal.emit(self.data,self.time_unit,center,shape,size,model_type)
        return None

    def distribution_analyze_start(self):
        """分析载流子寿命"""
        if self.data is None:
            return logging.warning('无数据载入')
        # self.time_slider_vertical.setVisible(False)
        # 如果线程没了，要创建
        if not self.is_thread_active("calc_thread"):
            self.cal_thread_open()


        self.calc_thread.start()
        self.update_status('长时计算进行中...', 'working')
        self.time_unit = float(self.time_step_input.value())
        model_type = 'single' if self.model_combo.currentText() == "单指数衰减" else 'double'
        self.start_dis_cal_signal.emit(self.data,self.time_unit,model_type)
        return None

    def diffusion_calculation_start(self):
        """扩散系数计算"""
        if self.data is None:
            return logging.warning('无数据载入')
        elif self.vectorROI_data is None:
            return  logging.warning("无有效ROI数据")
        # self.time_slider_vertical.setVisible(False)
        # 如果线程没了，要创建
        if not self.is_thread_active("calc_thread"):
            self.cal_thread_open()

        self.calc_thread.start()
        self.update_status('计算进行中...', 'working')
        self.time_unit = float(self.time_step_input.value())
        self.space_unit = float(self.space_step_input.value())
        self.start_dif_cal_signal.emit(self.vectorROI_data,self.time_unit, self.space_unit, self.data.timestamp,self.data.name)
        return None

    def pre_process_EM(self):
        """EM的数据预处理"""
        if self.data is None:
            logging.warning("请先加载数据")
            return
        if not hasattr(self.data, 'data_origin'):
            logging.error("数据加载有误，请重新加载数据")
            QMessageBox.warning(self,'数据错误',"数据加载有误，请重新加载数据")
            return
        if not self.is_thread_active("avi_thread"):
            # self.EM_thread_open()
            pass
        # 如果有线程在运算，要提示（不过目前不需要，保留语句）
        if not self.avi_thread.isRunning():
            self.avi_thread.start()
        self.pre_process_signal.emit(self.data, self.bg_nums_input.value(), True)

    def quality_EM_stft(self):
        if self.processed_data is None:
            logging.warning("没有数据可以计算，请先加载并预处理数据")
            return
        if "EM_pre_processed" == self.processed_data.type_processed:
            data = self.processed_data
        else:
            data = next(
                (data for data in reversed(self.processed_data.history) if data.type_processed == "EM_pre_processed"),None)
        if data is not None:
            # 窗函数选择转义
            window_dict = ['hann', 'hamming', 'gaussian', 'boxcar','blackman','blackmanharris']
            self.EM_params['stft_window_type'] = window_dict[self.stft_window_select.currentIndex()]
            dialog = STFTComputePop(self.EM_params,'quality')
            self.update_status("STFT计算ing", 'working')
            if dialog.exec_():
                self.EM_params['target_freq'] = dialog.target_freq_input.value()
                self.EM_fps = dialog.fps_input.value()
                self.EM_params['stft_window_size'] = dialog.window_size_input.value()
                self.EM_params['stft_noverlap'] = dialog.noverlap_input.value()
                self.EM_params['custom_nfft'] = dialog.custom_nfft_input.value()
                if not self.avi_thread.isRunning():
                    self.avi_thread.start()
                self.stft_quality_signal.emit(data,
                                              self.EM_params['target_freq'],self.EM_fps,
                                             self.EM_params['stft_window_size'],
                                             self.EM_params['stft_noverlap'],
                                             self.EM_params['custom_nfft'],
                                             self.EM_params['stft_window_type'])
                self.EM_params['EM_fps']=self.EM_fps
                # self.stft_quality_btn.setEnabled(False)
        else:
            logging.warning("查找不到预处理数据，请先对数据进行预处理")
            self.update_status("准备就绪")
            return

    def process_EM_stft(self):
        """EM的数据处理"""
        if self.stft_program_select.currentIndex() == 0:
            type = "python"
        if self.stft_program_select.currentIndex() == 1:
            type = "julia"
        if self.processed_data is None:
            logging.warning("没有数据可以计算，请先加载并预处理数据")
            return
        if "stft_quality" == self.processed_data.type_processed:
            data = self.processed_data
        else:
            data = next(
                (data for data in reversed(self.processed_data.history) if data.type_processed == "stft_quality"),
                None)
        if data is not None:
            self.update_status("STFT计算ing", 'working')
            # 窗函数选择转义
            window_dict = ['hann', 'hamming', 'gaussian', 'boxcar','blackman','blackmanharris']
            self.EM_params['stft_window_type'] = window_dict[self.stft_window_select.currentIndex()]
            if not self.avi_thread.isRunning():
                self.avi_thread.start()
            self.stft_python_signal.emit(data,
                                         self.EM_params['target_freq'], self.EM_fps,
                                         self.EM_params['stft_window_size'],
                                         self.EM_params['stft_noverlap'],
                                         self.EM_params['custom_nfft'],
                                         self.EM_params['stft_window_type'])
            self.stft_process_btn.setEnabled(False)
        else:
            logging.warning("查找不到数据，请先对数据进行预处理和质量评价")
            self.update_status("准备就绪")
            return

    def quality_EM_cwt(self):
        if self.processed_data is None:
            logging.warning("没有数据可以计算，请先加载并预处理数据")
            return
        if "EM_pre_processed" == self.processed_data.type_processed:
            data = self.processed_data
        else:
            data = next(
                (data for data in reversed(self.processed_data.history) if data.type_processed == "EM_pre_processed"),
                None)
        if data is not None:
            dialog = CWTComputePop(self.EM_params,'quality')

            if dialog.exec_():
                self.EM_params['target_freq'] = dialog.target_freq_input.value()
                self.EM_fps = dialog.fps_input.value()
                self.EM_params['cwt_total_scales'] = dialog.cwt_size_input.value()
                self.EM_params['cwt_type'] = dialog.wavelet.currentText()
                if not self.avi_thread.isRunning():
                    self.avi_thread.start()
                self.cwt_quality_signal.emit(data,
                                             self.EM_params['target_freq'],
                                             self.EM_fps,
                                             self.EM_params['cwt_total_scales'],
                                             self.EM_params['cwt_type'])
                # self.cwt_quality_btn.setEnabled(False)
                self.EM_params['EM_fps'] = self.EM_fps
        else:
            logging.warning("查找不到预处理数据，请先对数据进行预处理")
            self.update_status("准备就绪")
            return

    def process_EM_cwt(self):
        """小波变换"""
        if self.processed_data is None:
            logging.warning("没有数据可以计算，请先加载并预处理数据")
            return
        if "cwt_quality" == self.processed_data.type_processed:
            data = self.processed_data
        else:
            data = next(
                (data for data in reversed(self.processed_data.history) if data.type_processed == "cwt_quality"),
                None)
        if data is not None:
            dialog = CWTComputePop(self.EM_params, 'signal')
            if dialog.exec_():
                self.update_status("CWT计算ing", 'working')
                self.EM_params['EM_fps'] = dialog.fps_input.value()
                self.EM_fps = self.EM_params['EM_fps']
                self.EM_params['target_freq'] = dialog.target_freq_input.value()
                self.EM_params['cwt_type'] = dialog.wavelet.currentText()
                self.EM_params['cwt_total_scales'] = dialog.cwt_size_input.value()
                self.EM_params['cwt_scale_range'] = dialog.cwt_scale_range.value()
                if not self.avi_thread.isRunning():
                    self.avi_thread.start()
                self.cwt_python_signal.emit(data,
                                            self.EM_params['target_freq'],
                                            self.EM_fps,
                                            self.EM_params['cwt_total_scales'],
                                            self.EM_params['cwt_type'],
                                            self.EM_params['cwt_scale_range'])
                self.cwt_process_btn.setEnabled(False)
        else:
            logging.warning("查找不到质量评价数据，请先对数据进行质量评价")
            self.update_status("准备就绪", 'idle')
            return

    def roi_signal_avg(self):
        """计算选区信号平均值并显示"""
        if not hasattr(self,"bool_mask") or self.bool_mask is None:
            QMessageBox.warning(self,"警告","请先绘制ROI")
            return
        if self.processed_data.type_processed == 'ROI_stft' or self.processed_data.type_processed == 'ROI_cwt':
            data = self.processed_data
        else:
            data = next(
                (data for data in reversed(self.processed_data.history) if data.type_processed == "ROI_cwt" or data.type_processed == "ROI_stft"),
                None)
        if data is not None:
            pass
        else:
            logging.warning("无变换后数据，请先处理数据")
            return

        mask = self.bool_mask
        if mask.dtype == bool:
            EM_masked = np.where(mask[np.newaxis,:,:],self.processed_data.data_processed,0)

            total_valid_pixels = np.sum(mask)
            if total_valid_pixels == 0:
                QMessageBox.warning(self, "蒙版错误", "布尔蒙版中没有True像素")
                return
            # 计算每个时间点上的平均值
            average_series = np.sum(EM_masked, axis=(1, 2)) / total_valid_pixels
        else:
            raise TypeError(f"不支持的蒙版类型: {mask.dtype}")

        # self.data['EM_masked'] = EM_masked
        self.result_display.plot_time_series(self.processed_data.out_processed['time_series'] , average_series)

    def process_atam(self):
        if self.data and self.processed_data is None :
            logging.warning('请先导入数据')
            return
        if self.processed_data.type_processed == 'ROI_stft' or 'ROI_cwt':
            data = self.processed_data
        else:
            data = next(
                (data for data in reversed(self.processed_data.history) if
                 data.type_processed == "ROI_cwt" or data.type_processed == "ROI_stft"),
                None)
        if data is not None:
            self.atam_signal.emit(self.processed_data)
            self.atam_btn.setEnabled(False)
            return True
        else:
            QMessageBox.warning(self,"数据错误","不支持的数据类型，请确认前序处理是否正确")
            return False

    def process_tDgf(self):
        """单通道二维高斯拟合以及信号显示"""
        if self.data is None and self.processed_data is None:
            logging.warning('请先导入数据')
            return
        if self.processed_data.type_processed == 'ROI_stft' or 'ROI_cwt':
            data = self.processed_data
        else:
            data = next(
                (data for data in reversed(self.processed_data.history) if
                 data.type_processed == "ROI_cwt" or data.type_processed == "ROI_stft"),
                None)
        if data is not None:
            dialog = SCSComputePop(self.EM_params)
            if dialog.exec_():
                self.update_status("单通道计算ing", 'working')
                self.EM_params['scs_thr'] = dialog.thr_input.value()
                self.EM_params['thr_known'] = dialog.thr_known_check.isChecked()
                self.EM_params['scs_zoom'] = dialog.zoom_input.value()
                if not self.avi_thread.isRunning():
                    self.avi_thread.start()
                self.tDgf_signal.emit(data,
                                      self.EM_params['scs_zoom'],
                                      self.EM_params['scs_thr'],
                                      self.EM_params['thr_known'])
                self.tDgf_btn.setEnabled(False)
        else:
            QMessageBox.warning(self,"数据错误","不支持的数据类型，请确认前序处理是否正确")
            self.update_status("准备就绪", 'idle')
            return

    def processed_result(self, data:ProcessedData):
        """处理过后的数据都来这里重整再分配"""
        if isinstance(data, ProcessedData):
            pass
        else:
            self.cwt_quality_btn.setEnabled(True)
            self.stft_quality_btn.setEnabled(True)
            self.stft_process_btn.setEnabled(True)
            self.cwt_process_btn.setEnabled(True)
            self.tDgf_btn.setEnabled(True)
            QMessageBox.warning(self,"运算错误","data['error")
            return False
        self.processed_data = data
        # 各处理后响应
        process_type = self.processed_data.type_processed
        match process_type:
            case "ROI_lifetime":
                self.result_display.display_lifetime_curve(self.processed_data)
                pass
            case 'lifetime_distribution':
                self.result_display.display_distribution_map(self.processed_data)
                pass
            # 中间还有一个取向量ROI的，先不管他
            case 'diffusion':
                self.result_display.display_diffusion_coefficient(self.processed_data)
                pass
            case 'heat_transfer':
                pass
            case 'EM_pre_processed':
                pass
            case 'stft_quality':
                self.stft_quality_btn.setEnabled(True)
                logging.info("请稍等，出图会有点慢")
                self.result_display.quality_avg(self.processed_data)
            case 'cwt_quality':
                self.cwt_quality_btn.setEnabled(True)
                logging.info("请稍等，出图会有点慢")
                self.result_display.quality_avg(self.processed_data)
            case 'ROI_stft':
                result = self.processed_data.data_processed
                self.stft_process_btn.setEnabled(True)
                if self.show_stft_check.isChecked():
                    self.data.image_import = (result - np.min(result)) / (np.max(result) - np.min(result)) # 要改
                    self.load_image()
                pass
            case 'ROI_cwt':
                result = self.processed_data.data_processed
                self.cwt_process_btn.setEnabled(True)
                if self.show_stft_check.isChecked():
                    self.data.image_import = (result - np.min(result)) / (np.max(result) - np.min(result))  # 要改
                    self.load_image()
                pass
            case 'Accumulated_time_amplitude_map':
                self.atam_btn.setEnabled(True)
                pass
            case 'Single_channel_signal':
                self.tDgf_btn.setEnabled(True)
                if self.processed_data.out_processed['thr_known']:
                    self.result_display.single_channel(self.processed_data,True)
                else:
                    thr = int(self.processed_data.out_processed['thr'])
                    self.time_slider_vertical.setVisible(True)
                    self.time_slider_vertical.setMaximum(int(self.processed_data.out_processed['max_signal'].max()*10))
                    # self.time_slider_vertical.setValue(thr*10)
                    self.update_result_display(thr*10, reuse_current=False)

    def draw_result(self,draw_type:str,canvas_id:int,result):
        """canvas绘图结果处理"""
        timestamp = self.image_display.display_canvas[canvas_id].data.timestamp_inherited
        draw_data = None
        if self.data is not None:
            for data in self.data.history:
                if data.timestamp == timestamp:
                    draw_data = data
                    break
        if self.processed_data is not None and draw_data is None:
            draw_data = next(data for data in self.processed_data.history if data.timestamp == timestamp)
        if draw_type == "v_rect":
            # ((x, y), width, height)
            x,y,w,h = result[0][0],result[0][1],result[1],result[2]
            if w == 0 or h == 0:
                return
            if hasattr(draw_data,'type_processed') and draw_data.type_processed == 'Accumulated_time_amplitude_map':
                try:
                    source_data = next(data for data in self.processed_data.history if data.timestamp == draw_data.timestamp_inherited)
                except: # 如果不行就从data里找
                    source_data = next(
                        data for data in self.data.history if data.timestamp == draw_data.timestamp_inherited)
                if isinstance(source_data, Data):
                    roi_data = source_data.data_origin[:,y:y+h,x:x+w]
                elif isinstance(source_data, ProcessedData):
                    roi_data = source_data.data_processed[:,y:y+h,x:x+w]
                else:
                    logging.error("roi应用错误（不可能错误）")
                    roi_data = None

                self.processed_data = ProcessedData(draw_data.timestamp,
                                                f"{draw_data.name}@ROIed",
                                                "Roi_applied",
                                                time_point=draw_data.time_point,
                                                data_processed=roi_data,
                                                out_processed=draw_data.out_processed,
                                                ROI_applied=True)

    '''其他功能'''
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

    def stop_thread(self,type = 0):
        """彻底删除线程（反正关闭也不能重启）后续线程多了加入选择关闭的能力"""
        if type == 0 and self.is_thread_active("calc_thread"):
            try:
                self.calc_thread.quit()  # 请求退出
                self.calc_thread.wait()  # 等待结束
                self.calc_thread.deleteLater()  # 标记删除
                logging.info("计算线程关闭")
            except Exception as e:
                logging.error(f"线程退出错误{e}")
        if type == 1 and hasattr(self,"avi_thread") and self.is_thread_active("avi_thread"):
            try:
                self.avi_thread.quit()
                self.avi_thread.wait()
                self.avi_thread.deleteLater()
                logging.info("大数据处理线程关闭")
            except Exception as e:
                logging.error(f"线程退出错误{e}")

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
        if self.result_display.current_dataframe is not None:
            dialog = DataSavingPop(self)
            file_path = None
            self.update_status("数据导出ing", 'working')
            if dialog.exec_():
                isfiting = dialog.fitting_check.isChecked()
                hasheader = dialog.index_check.isChecked()
                extra_check = dialog.extra_check.isChecked()
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "保存数据", "", "CSV文件 (*.csv);;文本文件 (*.txt)")
        else:
            logging.warning("没有数据可以导出")
            self.update_status("准备就绪")
            return

        if file_path:
            df = self.result_display.current_dataframe
            if not isfiting:
                if self.result_display.current_mode == 'curve':
                    df = df.loc[:, df.columns != 'fit_curve']
                elif self.result_display.current_mode == 'diff':
                    df = df.loc[:, df.columns.get_level_values(1) != '拟合曲线']
                elif self.result_display.current_mode == 'heatmap':
                    pass
                elif self.result_display.current_mode == 'roi':
                    pass
                elif self.result_display.current_mode == 'var':
                    pass
                elif self.result_display.current_mode == 'series':
                    pass
            # 保存为CSV或TXT
            if file_path.lower().endswith('.csv'):
                try:
                    df.to_csv(file_path, index=False, header=hasheader)
                    logging.info("数据已保存")
                except:
                    logging.info("数据未保存")
            else:
                try:
                    df.to_csv(file_path, sep='\t', index=False, header=hasheader)
                    logging.info("数据已保存")
                except:
                    logging.info("数据未保存")
            self.update_status("准备就绪", 'idle')
        else:
            logging.info("数据未保存")
            self.update_status("准备就绪", 'idle')
            return

    def export_EM_data(self,result):
        """时频变换后目标频率下的结果导出"""
        if self.processed_data is not None:
            if self.processed_data.type_processed == 'ROI_stft' or 'ROI_cwt':
                dialog = MassDataSavingPop()
                if dialog.exec_():
                    directory = dialog.directory
                    prefix = dialog.text_edit.text().strip()
                    filetype = dialog.type_combo.currentText()
                    self.mass_export_signal.emit(self.processed_data.data_processed,directory,prefix,filetype)
                return
            else:
                QMessageBox.warning(self,'提示','请先变换处理数据')
        else:
            logging.warning('请先加载并处理数据')
            return

    def data_history_view(self):
        """查看历史数据"""
        if self.data is None :
            logging.warning('请先导入数据')
            return

        dialog = DataViewAndSelectPop(datadict=self.get_data_all())
        if dialog.exec_():
            selected_timestamp = dialog.get_selected_timestamp()
            self.data = self.data.find_history(selected_timestamp)
            logging.info(f"当前数据焦点已更新至{self.data.name}")

    def process_history_view(self):
        """查看历史数据-处理"""
        if self.processed_data is None:
            logging.warning('请先处理数据')
            return
        # if self.image_display is []:  走不到这里
        #     logging.warning("请先导入数据")

        dialog = DataViewAndSelectPop(processed_datadict=self.get_processed_data_all())
        if dialog.exec_():
            selected_timestamp = dialog.get_selected_timestamp()
            self.processed_data = self.processed_data.find_history(selected_timestamp)
            logging.info(f"当前数据焦点已更新至{self.processed_data.name}")

    def data_history_clear(self):
        """历史数据清除（所有）"""
        if Data is not None:
            Data.clear_history()
            logging.info('导入数据已清除')
        else:
            logging.warning('没有数据可清除')
        if ProcessedData is not None:
            ProcessedData.clear_history()
            logging.info('处理数据已清除')


    '''以下控制台命令更新'''
    def stop_calculation(self):
        """终止当前计算"""
        # 这里需要实现终止计算的逻辑
        # 可以通过设置标志位或直接终止计算线程
        logging.warning("计算终止请求已接收，正在停止...")
        if hasattr(self, 'cal_thread'):
            # self.cal_thread.stop()
            self.stop_thread(0)
        if hasattr(self, 'avi_thread'):
            # self.avi_thread.stop()
            self.stop_thread(1)
        return
        # 实际终止逻辑需要根据你的计算实现来添加

    def save_config(self):
        """保存当前配置(留空暂不实现)"""
        logging.info("正在保存当前配置...")

    def load_config(self, preset_name):
        """加载预设参数(留空暂不实现)"""
        logging.info(f"正在加载预设参数: {preset_name}")

    def clear_result(self):
        self.result_display.clear()

    @staticmethod
    def format_time(seconds):
        """将秒数格式化为 HH:MM:SS"""
        if seconds < 0:
            return "--:--:--"
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


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
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/NotoSansSC-VF.ttf")  # 如：思源黑体、阿里巴巴普惠体
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/calibril.ttf")  # 如：Roboto、Fira Code
    QSS = """
    QWidget {
        background-color: #0a192f;  /* 深蓝背景 */
        color: #e6f1ff;             /* 浅蓝文字 */
        font-family: "Fira Code", "Microsoft YaHei"; /* 优先使用科技感英文字体 */
        font-size: 12px;
        border: none;
    }

    /* 按钮样式 */
    QPushButton {
        background-color: #112240;   /* 深蓝按钮 */
        border: 1px solid #64ffda;   /* 科技感青色边框 */
        border-radius: 4px;          /* 圆角 */
        padding: 8px 16px;
        color: #64ffda;              /* 青色文字 */
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #0c2a4d;   /* 悬停加深 */
        border-color: #00ffcc;       /* 悬停高亮 */
    }
    QPushButton:pressed {
        background-color: #020c1b;   /* 按下效果 */
    }

    /* 输入框样式 */
    QLineEdit, QTextEdit {
        background-color: #0a192f;
        border: 1px solid #233554;   /* 深蓝边框 */
        border-radius: 3px;
        padding: 6px;
        color: #a8b2d1;              /* 灰蓝文字 */
        selection-background-color: #64ffda; /* 选中文本背景 */
    }
    QLineEdit:focus, QTextEdit:focus {
        border-color: #64ffda;       /* 聚焦高亮 */
    }

    /* 标签与标题 */
    QLabel {
        color: #ccd6f6;              /* 亮蓝文字 */
        font-size: 14px;
    }
    QLabel#title {                   /* 通过 objectName 定制 */
        font-size: 18px;
        font-weight: bold;
        color: #64ffda;
    }

    /* 复选框/单选框 */
    QCheckBox, QRadioButton {
        color: #ccd6f6;
        spacing: 5px;                /* 图标与文字间距 */
    }
    QCheckBox::indicator, QRadioButton::indicator {
        width: 16px;
        height: 16px;
        border: 1px solid #64ffda;
        border-radius: 3px;
    }
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {
        background-color: #64ffda;   /* 选中状态 */
    }

    /* 进度条 */
    QProgressBar {
        border: 1px solid #233554;
        border-radius: 3px;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #64ffda;   /* 进度填充色 */
    }
    """
    QSS1 ="""
    /* ========== QMenuBar 样式 ========== */
QMenuBar {
    background-color: #ffffff;
    border-top: 3px solid #4CAF50;
    padding: 6px;
    margin-bottom: 3px;
    font-weight: 475;
}

QMenuBar::item {
    background: transparent;
    padding: 4px 8px;
    border-radius: 2px;
    color: #2E7D32;
    margin: 0px 2px 4px 2px;
}

QMenuBar::item:selected {
    background-color: #E8F5E9;
}

QMenuBar::item:pressed {
    background-color: #C8E6C9;
}
QDockWidget {
    border: 1px solid #388E3C;
    border-radius: 4px;
}


/* 标题栏样式 */
QDockWidget::title {
    background-color: #C8E6C9;
    text-align: center;
    padding: 2px;
}

QDockWidget::title:hover {
    background-color: #4CAF50;
    color: white;
}


/* 内容区域样式 */
QDockWidget > QWidget {
    background-color: #fefefe;
    border: 2px solid #C8E6C9;
    border-top: none;

}

/* 分隔线样式 */
QDockWidget::separator {
    background-color: #C8E6C9;
    width: 1px;
    height: 1px;
}

/* 当停靠窗口浮动时的样式 */
QDockWidget[floating="true"] {
    border: 2px solid #C8E6C9;
}
/* ========== 状态栏渐变 ========== */
QStatusBar {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #C8E6C9, stop:1 #fefefe);
}
/* ========== 进度条 ========== */
QProgressBar {
                border: 1px solid #C8E6C9;
                border-radius: 3px;
                background: white;
                text-align: center;
                min-height: 18px;
                max-height: 18px;
            }
            QProgressBar::chunk {
                background-color: #C8E6C9;
                width: 10px;
            }
/* ========== 工具栏 ========== */
QToolBar {
    background-color: white;
    border: 2px outset #C8E6C9;
    padding: 2px;
    spacing: 6px; /* 工具按钮间距 */
}
QToolBox::tab:selected { /* italicize selected tabs */
    font: italic;
    color: white;
}
QToolBar::separator {
    background-color: #C8E6C9;
    width: 1px;
    margin: 4px 2px;
}

QToolBar QToolButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 0px;
    min-width: 24px;
}

QToolBar QToolButton:hover {
    background-color: #E8F5E9;
    border-color: #C8E6C9;
}

QToolBar QToolButton:checked {
    background-color: #C8E6C9;
    border-color: #4CAF50;
}

QToolBar QToolButton:pressed {
    background-color: #A5D6A7;
}

/* ========== QToolBox 样式 ========== */
QToolBox {
    background-color: white;
}

QToolBox::tab {
    background-color: #E8F5E9;
    color: #2E7D32;
    border: 1px solid #C8E6C9;
    border-radius: 4px;
    margin-bottom: 4px;
    padding: 8px;
    font-weight: 500;
}

QToolBox::tab:selected {
    background-color: #4CAF50;
    color: white;
}

QToolBox::tab:hover {
    background-color: #C8E6C9;
}

QToolBox > QWidget {
    background-color: white;
    border: 1px solid #C8E6C9;
    border-radius: 4px;
}
/* ========== QButton 样式 ========== */
QPushButton {
    background-color: white;
    border: 1px solid #9ad19a;
    border-radius: 4px;
    padding: 3px 4px 3px 4px;
}
QPushButton:hover {
    border-color: #4CAF50;
}
QPushButton:pressed {
    border: 2px solid #4CAF50;
    background-color: #F1F8E9;
}

QPushButton:disabled {
    background-color: #F5F5F5;
    color: #BDBDBD;
}

/* ========== QComboBox 样式 ========== */
QComboBox {
    background-color: white;
    border: 1px solid #9ad19a;
    border-radius: 4px;
    padding: 2px 2px 2px 4px; 

    min-width: 50px;
    selection-background-color: #E8F5E9;
}

QComboBox:hover {
    border-color: #4CAF50;
}

QComboBox:focus {
    border: 2px solid #4CAF50;
    background-color: #F1F8E9;
}

QComboBox:disabled {
    background-color: #F5F5F5;
    color: #BDBDBD;
}

/* 下拉箭头样式 */
QComboBox::drop-down {
    width: 24px;
    border-left: 1px solid #C8E6C9;
    border-radius: 0 2px 2px 0;
}

QComboBox::down-arrow {
}


/* 下拉菜单样式 */
QComboBox QAbstractItemView {
    background-color: white;
    border: 1px solid #C8E6C9;
    selection-background-color: #E8F5E9;
    selection-color: #2E7D32;
    outline: 0;  /* 移除选中项的虚线框 */
    padding: 4px;
}

QComboBox QAbstractItemView::item {
    height: 28px;
    padding: 0 8px;
}

QComboBox QAbstractItemView::item:hover {
    background-color: #C8E6C9;
}
    QTableWidget {
        gridline-color: #d0d0d0; /* 网格线颜色 */
        background-color: white; /* 背景色 */
    }
    /* 表头样式 */
    QHeaderView::section {
        background-color: #e0e0e0;
        padding: 4px;
        border: 1px solid #c0c0c0;
        font-weight: bold;
    }
    /* 单元格样式 */
    QTableWidget::item {
        padding: 3px;
        border: none; /* 去除默认边框，使用网格线 */
    }
    /* 选中单元格的样式 */
    QTableWidget::item:focus {
        background-color: #C8E6C9; /* 获得焦点时的背景色 */
        color: black;
    }
    /* 另一种设置选中行样式的方法（如果设置了SelectionBehavior为SelectRows） */
    QTableWidget::item:selected {
        background-color: #4CAF50; /* 选中项背景色 */
        color: white;
    }
    /* 鼠标悬停在单元格上的样式 */
    QTableWidget::item:hover {
        background-color: #C8E6C9; /* 悬停颜色 */
    }
    QScrollArea {
        border: none;
        background-color: white;
    }
    QScrollBar:vertical {
        width: 12px;
        background: #f0f0f0;
    }
    QScrollBar::handle:vertical {
        background: #C8E6C9;
        min-height: 20px;
    }
    QScrollBar::handle:vertical:hover {
        background: #A5D6A7;      /* 悬停时稍深 */
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    """
    # 应用全局样式
    app.setStyle('Fusion')
    app.setStyleSheet(QSS1)
    # app.setFont(QFont("Noto Sans"))
    app.setWindowIcon(QIcon(':/LifeCalor.ico'))
    window = MainWindow()
    window.setWindowIcon(QIcon(':/LifeCalor.ico'))
    window.show()
    app.exec_()