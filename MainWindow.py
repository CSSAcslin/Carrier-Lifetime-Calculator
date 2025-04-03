import numpy as np
from scipy.stats import pearsonr
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QScrollArea,
                             QFileDialog, QSlider, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QStackedWidget, QDockWidget
                             )
from PyQt5.QtCore import Qt, pyqtSignal

from DataProcessor import DataProcessor
from ImageDisplayWidget import ImageDisplayWidget
from LifetimeCalculator import LifetimeCalculator
from ResultDisplayWidget import ResultDisplayWidget
from ConsoleUtils import *

class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "carrier_lifetime.log")
        self.setup_logging()
        self.log_startup_message()
        # 参数初始化
        self.data = None
        self.time_points = None
        self.time_unit = 1.0
        self.space_unit = 1.0
        # 信号连接
        self.image_display.mouse_position_signal.connect(self._handle_hover)
        self.image_display.mouse_clicked_signal.connect(self._handle_click)


    def init_ui(self):
        self.setWindowTitle("载流子寿命分析工具")
        self.setGeometry(100, 100, 1100, 900)
        # 主部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout(main_widget)

        # 左侧参数设置区域
        self.setup_parameter_panel()
        main_layout.addWidget(self.parameter_panel, stretch=1)

        # 右侧区域 (图像和结果)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 图像显示区域
        self.image_display = ImageDisplayWidget(self)
        right_layout.addWidget(self.image_display, stretch=2)

        # 时间滑块
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.valueChanged.connect(self.update_time_slice)
        self.time_label = QLabel("时间点: 0/0")

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("时间序列:"))
        slider_layout.addWidget(self.time_slider)
        slider_layout.addWidget(self.time_label)

        right_layout.addLayout(slider_layout)

        # 结果显示区域
        self.result_display = ResultDisplayWidget()
        right_layout.addWidget(self.result_display, stretch=2)
        main_layout.addWidget(right_panel, stretch=3)

        # 设置控制台
        self.setup_console()

    def setup_parameter_panel(self):
        """设置参数面板"""
        self.parameter_panel = self.QGroupBoxCreator("参数设置")
        left_layout = QVBoxLayout()
        left_layout.setSpacing(20)
        left_layout.setContentsMargins(1,20,1,20)

        # 文件夹选择
        folder_choose = QHBoxLayout()
        self.folder_btn = QPushButton("选择TIFF文件夹")
        self.folder_btn.clicked.connect(self.load_tiff_folder)
        self.folder_path_label = QLabel("未选择文件夹")
        self.folder_path_label.setMaximumWidth(300)
        self.folder_path_label.setWordWrap(True)
        self.folder_path_label.setStyleSheet("font-size: 10px;") #后续还要改
        self.group_selector = QComboBox()
        self.group_selector.addItems(['n', 'p'])

        folder_choose.addWidget(QLabel("数据源:"))
        folder_choose.addWidget(self.folder_btn)
        folder_choose.addWidget(self.group_selector)
        left_layout.addLayout(folder_choose)
        left_layout.addWidget(self.folder_path_label)


        # 时间参数
        time_set = self.QGroupBoxCreator("时间参数:")

        time_layout = QVBoxLayout()
        time_start_layout = QHBoxLayout()
        time_start_layout.addWidget(QLabel("起始时间:"))
        self.time_start_input = QDoubleSpinBox()
        self.time_start_input.setMinimum(0)
        self.time_start_input.setValue(0)
        time_start_layout.addWidget(self.time_start_input)
        time_start_layout.addWidget(QLabel("帧"))
        time_layout.addLayout(time_start_layout)

        time_step_layout = QHBoxLayout()
        time_step_layout.addWidget(QLabel("时间单位:"))
        self.time_step_input = QDoubleSpinBox()
        self.time_step_input.setMinimum(0.001)
        self.time_step_input.setValue(1.0)
        time_step_layout.addWidget(self.time_step_input)
        time_step_layout.addWidget(QLabel("ps/帧"))
        time_layout.addLayout(time_step_layout)
        time_layout.addWidget(QLabel("     (最小分辨率：1 fs)"))
        time_set.setLayout(time_layout)
        left_layout.addWidget(time_set)

        # 空间参数
        space_set = self.QGroupBoxCreator("空间参数:")
        space_layout = QVBoxLayout()
        space_step_layout = QHBoxLayout()
        space_step_layout.addWidget(QLabel("空间单位:"))
        self.space_step_input = QDoubleSpinBox()
        self.space_step_input.setMinimum(0.001)
        self.space_step_input.setValue(1.0)
        space_step_layout.addWidget(self.space_step_input)
        space_step_layout.addWidget(QLabel("μm/像素"))
        space_layout.addLayout(space_step_layout)
        space_layout.addWidget(QLabel('     (最小分辨率：1 nm)'))
        # 鼠标悬停显示
        self.mouse_pos_label = QLabel("鼠标位置: x= -, y= -, t= -\n值: -")
        self.mouse_pos_label.setStyleSheet("background-color: #f0f0f0; padding-top: 5px;")
        space_layout.addWidget(self.mouse_pos_label)
        space_set.setLayout(space_layout)
        left_layout.addWidget(space_set)

        # 分析总体设置
        operation_set = self.QGroupBoxCreator("分析设置:")
        operation_layout = QVBoxLayout()
        # 寿命模型选择
        operation_layout.addWidget(QLabel("\n寿命模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["单指数衰减", "双指数衰减"])
        operation_layout.addWidget(self.model_combo)
        # 区域分析设置
        # operation_layout.addSpacing(10)
        operation_layout.addWidget(QLabel("\n分析模式:"))
        self.function_combo = QComboBox()
        self.function_combo.addItems(["载流子热图分析", "特定区域寿命分析"])
        operation_layout.addWidget(self.function_combo)
        # 区域分析参数
        self.region_shape_combo = QComboBox()
        self.region_shape_combo.addItems(["正方形", "圆形"])
        self.region_size_input = QSpinBox()
        self.region_size_input.setMinimum(1)
        self.region_size_input.setMaximum(50)
        self.region_size_input.setValue(5)
        self.analyze_region_btn = QPushButton("分析选定区域")
        self.analyze_region_btn.clicked.connect(self.region_analyze)
        # 区域坐标输入
        self.region_x_input = QSpinBox()
        self.region_y_input = QSpinBox()
        self.region_x_input.setMaximum(131)
        self.region_y_input.setMaximum(131) #这里后面要改成根据图像大小调节
        # 载流子寿命分布图参数板
        self.function_stack = QStackedWidget()
        heatmap_group = self.QGroupBoxCreator(style = "inner")
        heatmap_layout = QVBoxLayout()
        self.analyze_btn = QPushButton("开始分析")
        self.analyze_btn.clicked.connect(self.distribution_analyze)
        heatmap_layout.addWidget(self.analyze_btn)
        heatmap_group.setLayout(heatmap_layout)
        self.function_stack.addWidget(heatmap_group)
        # 特定区域寿命分析功能参数板
        region_group = self.QGroupBoxCreator(style = "inner")
        region_layout = QVBoxLayout()
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
        region_layout.addLayout(coord_layout)
        region_layout.addLayout(shape_layout)
        region_layout.addLayout(size_layout)
        region_layout.addWidget(self.analyze_region_btn)
        region_group.setLayout(region_layout)
        self.function_stack.addWidget(region_group)
        operation_layout.addWidget(self.function_stack)
        self.function_combo.currentIndexChanged.connect(self.function_stack.setCurrentIndex)

        operation_set.setLayout(operation_layout)
        left_layout.addWidget(operation_set)
        # 添加分析按钮和导出按钮
        left_layout.addSpacing(20)
        data_save_layout = QHBoxLayout()
        self.export_image_btn = QPushButton("导出结果为图片")
        self.export_data_btn = QPushButton("导出结果为数据")
        data_save_layout.addWidget(self.export_image_btn)
        data_save_layout.addWidget(self.export_data_btn)
        left_layout.addLayout(data_save_layout)
        self.parameter_panel.setLayout(left_layout)

        # 连接导出按钮
        self.export_image_btn.clicked.connect(self.export_image)
        self.export_data_btn.clicked.connect(self.export_data)

    def QGroupBoxCreator(self,title="",style="default"):
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
            }"""
        }
        group_box.setStyleSheet(styles.get(style, styles["default"]))
        return group_box

    def setup_console(self):
        """设置控制台停靠窗口"""
        self.console_dock = QDockWidget("控制台", self)
        self.console_dock.setObjectName("ConsoleDock")

        # 创建控制台部件
        self.console_widget = ConsoleWidget(self)
        self.command_processor = CommandProcessor(self)

        # 连接信号
        self.command_processor.terminate_requested.connect(self.stop_calculation)
        self.command_processor.save_config_requested.connect(self.save_config)
        self.command_processor.load_config_requested.connect(self.load_config)

        self.console_dock.setWidget(self.console_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console_dock)

        # 设置控制台特性
        self.console_dock.setMinimumHeight(200)
        self.console_dock.setFeatures(QDockWidget.DockWidgetMovable |
                                      QDockWidget.DockWidgetFloatable |
                                      QDockWidget.DockWidgetClosable)

        # 添加菜单项
        view_menu = self.menuBar().addMenu("视图")
        toggle_console = view_menu.addAction("显示/隐藏控制台")
        toggle_console.triggered.connect(lambda: self.console_dock.setVisible(not self.console_dock.isVisible()))

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

    def log_to_console(self, message):
        """将消息输出到控制台"""
        self.console_widget.console_output.append(message)
        self.console_widget.console_output.verticalScrollBar().setValue(
            self.console_widget.console_output.verticalScrollBar().maximum()
        )

    def log_startup_message(self):
        """记录程序启动消息"""
        startup_msg = f"""
        ============================================
        载流子寿命分析工具启动
        启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        日志文件: {self.log_file}
        程序版本: 1.0.0
        ============================================
        """
        logging.info(startup_msg.strip())
        logging.info("程序已进入准备状态，等待用户操作...")

    def _handle_hover(self, x, y, t, value):
        """更新鼠标位置显示"""
        if hasattr(self, 'time_points') and self.time_points is not None:
            time_val = self.time_points[t]
        else:
            time_val = t

        self.mouse_pos_label.setText(
            f"鼠标位置: x={x}, y={y}, t={time_val}\n值: {value:.2f}")

    def _handle_click(self, x, y):
        """处理图像点击事件"""
        if self.function_combo.currentIndex() == 1:  # 区域分析模式
            self.region_x_input.setValue(x)
            self.region_y_input.setValue(y)

    def load_tiff_folder(self):
        """加载TIFF文件夹"""
        self.time_unit = float(self.time_step_input.value())
        folder_path = QFileDialog.getExistingDirectory(self, "选择TIFF图像文件夹")
        loader = DataProcessor(folder_path)
        if folder_path:
            self.folder_path_label.setText(folder_path)
            current_group = self.group_selector.currentText()

            # 读取文件夹中的所有tiff文件
            tiff_files = loader.load_and_sort_files(current_group)

            if not tiff_files:
                self.folder_path_label.setText("文件夹中没有目标TIFF文件")
                return

            # 读取所有图像
            self.data = loader.process_files(tiff_files, self.time_start_input, self.time_unit)

            if not self.data:
                self.folder_path_label.setText("无法读取TIFF文件")
                return

            # 设置时间滑块
            self.time_slider.setMaximum(len(self.data['images']) - 1)
            self.time_label.setText(f"时间点: 0/{len(self.data['images']) - 1}")

            # 显示第一张图像
            self.update_time_slice(0)

    def update_time_slice(self, idx):
        """更新时间切片显示"""
        if self.data is not None and 0 <= idx < len(self.data['images']):
            self.time_label.setText(f"时间点: {idx}/{len(self.data['images']) - 1}")
            # zoom = self.image_display.zoom_spinbox.value()
            self.image_display.current_image = self.data['images'][idx]
            self.image_display.display_image(self.data['images'][idx])


    def region_analyze(self):
        """分析选定区域"""
        if self.data is None or not hasattr(self, 'time_points'):
            return

        # 获取参数
        self.time_unit = float(self.time_step_input.value())
        self.time_points = self.data['time_points']
        center = (self.region_y_input.value(), self.region_x_input.value())
        shape = 'square' if self.region_shape_combo.currentText() == "正方形" else 'circle'
        size = self.region_size_input.value()
        model_type = 'single' if self.model_combo.currentText() == "单指数衰减" else 'double'

        # 执行区域分析
        lifetime, fit_curve, mask, phy_signal, r_squared = LifetimeCalculator.analyze_region(
            self.data, self.time_points, center, shape, size, model_type)

        # 显示结果
        self.result_display.display_lifetime_curve(phy_signal, lifetime, r_squared, fit_curve,self.time_points, self.data['boundary'])



    def distribution_analyze(self):
        """分析载流子寿命"""
        if self.data is None:
            return

        # 获取时间点
        self.time_unit = float(self.time_step_input.value())
        self.time_points = self.data['time_points'] * self.time_unit
        self.data_type = self.data['data_type']
        self.value_mean_max = np.abs(self.data['data_mean'])
        self.loading_bar = pyqtSignal(int, int)
        self.loading_bar.connect(ConsoleWidget.update_progress)

        # 获取模型类型
        model_type = 'single' if self.model_combo.currentText() == "单指数衰减" else 'double'

        # 计算每个像素的寿命
        height, width = self.data['data_origin'].shape[1], self.data['data_origin'].shape[2]
        lifetime_map = np.zeros((height, width))
        logging.info("开始计算载流子寿命...")
        l=0 #进度条
        total_l = height * width
        for i in range(height):
            for j in range(width):
                time_series = self.data['data_origin'][:, i, j]
                # 用皮尔逊系数判断噪音(滑动窗口法)
                window_size = min(10, len(self.time_points) // 2)
                pr = []
                for k in range(len(time_series) - window_size):
                    window = time_series[k:k + window_size]
                    time_window = self.time_points[k:k + window_size]
                    r, _ = pearsonr(time_window, window)
                    pr.append(r)
                    if abs(r) >= 0.8:
                        _, lifetime, r_squared, _ = LifetimeCalculator.calculate_lifetime(self.data_type, time_series, self.time_points, model_type)
                        continue
                    else:
                        pass
                if np.all(np.abs(pr) < 0.8):
                    lifetime = np.nan
                else:
                    pass
                lifetime_map[i, j] = lifetime if not np.isnan(lifetime) else 0
                self.loading_bar.emit(l+1, total_l)
        logging.info("计算完成!")
        # 显示结果
        smoothed_map = LifetimeCalculator.apply_custom_kernel(lifetime_map)
        self.result_display.display_distribution_map(smoothed_map)

    def export_image(self):
        """导出热图为图片"""
        if hasattr(self.result_display, 'current_data'):
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存图像", "", "PNG图像 (*.png);;JPEG图像 (*.jpg *.jpeg);;TIFF图像 (*.tif *.tiff)")

            if file_path:
                # 从matplotlib保存图像
                self.result_display.figure.savefig(file_path, dpi=300, bbox_inches='tight')

    def export_data(self):
        """导出寿命数据"""
        if hasattr(self.result_display, 'current_data'):
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存数据", "", "CSV文件 (*.csv);;文本文件 (*.txt)")

            if file_path:
                # 保存为CSV或TXT
                if file_path.lower().endswith('.csv'):
                    self.result_display.current_data.to_csv(file_path, index=False, header=False)
                else:
                    self.result_display.current_data.to_csv(file_path, sep='\t', index=False, header=False)


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()