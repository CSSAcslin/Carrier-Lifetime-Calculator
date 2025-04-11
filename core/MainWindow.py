from datetime import datetime
from logging.handlers import RotatingFileHandler

from PyQt5 import sip
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QScrollArea,
                             QFileDialog, QSlider, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QStackedWidget, QDockWidget, QStatusBar
                             )
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QMetaObject

from DataProcessor import DataProcessor
from ImageDisplayWidget import ImageDisplayWidget
from LifetimeCalculator import LifetimeCalculator, CalculationThread
from ResultDisplayWidget import ResultDisplayWidget
from ConsoleUtils import *
from ExtraDialog import BadFrameDialog
import logging
import resources_rc

class MainWindow(QMainWindow):
    """主窗口"""
    # 线程激活信号
    start_reg_cal_signal = pyqtSignal(dict, float, tuple, str, int, str)
    start_dis_cal_signal = pyqtSignal(dict, float, str)

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
        # 状态控制
        self._is_calculating = False
        # 线程管理
        self.thread_open()
        # 信号连接
        self.signal_connect()

    def init_ui(self):
        self.setWindowTitle("载流子寿命分析工具")
        self.setGeometry(100, 20, 1300, 850)

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

        self.setup_status_bar()

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
        self.folder_path_label = QLabel("未选择文件夹")
        self.folder_path_label.setMaximumWidth(300)
        self.folder_path_label.setWordWrap(True)
        self.folder_path_label.setStyleSheet("font-size: 14px;") #后续还要改
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

        # 坏点处理动作
        bad_frame_action = edit_menu.addAction("坏点处理")
        bad_frame_action.triggered.connect(self.show_bad_frame_dialog)

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
        self.status_label.setFixedWidth(300)
        self.status_bar.addWidget(self.status_label)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(800)
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
        self.addDockWidget(Qt.RightDockWidgetArea, self.console_dock)

        # 设置控制台特性
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
        程序版本: 1.4.1
        ============================================
        """
        logging.info(startup_msg.strip())
        logging.info("程序已进入准备状态，等待用户操作...")

    def thread_open(self):
        """计算线程相关 以及信号槽连接都放在这里了"""
        self.thread = QThread()
        self.cal_thread = CalculationThread()
        self.cal_thread.moveToThread(self.thread)
        # 计算状态更新
        self.start_reg_cal_signal.connect(self.cal_thread.region_analyze)
        self.start_dis_cal_signal.connect(self.cal_thread.distribution_analyze)
        self.cal_thread.calculating_progress_signal.connect(self.update_progress)
        self.cal_thread.result_data_signal.connect(self.result_display.display_lifetime_curve)
        self.cal_thread.result_map_signal.connect(self.result_display.display_distribution_map)
        self.cal_thread.stop_thread_signal.connect(self.stop_thread)
        self.cal_thread.cal_time.connect(lambda ms: logging.info(f"耗时: {ms}毫秒"))
        self.cal_thread.cal_running_status.connect(self.btn_safety)

    def signal_connect(self):
        # 连接参数区域按钮
        self.folder_btn.clicked.connect(self.load_tiff_folder)
        self.analyze_region_btn.clicked.connect(self.region_analyze_start)
        self.analyze_btn.clicked.connect(self.distribution_analyze_start)
        self.function_combo.currentIndexChanged.connect(self.function_stack.setCurrentIndex)
        self.export_image_btn.clicked.connect(self.export_image)
        self.export_data_btn.clicked.connect(self.export_data)
        # 鼠标移动
        self.image_display.mouse_position_signal.connect(self._handle_hover)
        self.image_display.mouse_clicked_signal.connect(self._handle_click)
        # 时间滑块
        self.time_slider.valueChanged.connect(self.update_time_slice)
        # 连接控制台信号
        self.command_processor.terminate_requested.connect(self.stop_calculation)
        self.command_processor.save_config_requested.connect(self.save_config)
        self.command_processor.load_config_requested.connect(self.load_config)
        self.command_processor.clear_result_requested.connect(self.clear_result)

    '''上面是初始化预设，下面是功能响应'''
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
        self.data_processor = DataProcessor(folder_path)
        if folder_path:
            logging.info(folder_path)
            self.folder_path_label.setText("已加载文件夹")
            current_group = self.group_selector.currentText()

            # 读取文件夹中的所有tiff文件
            tiff_files = self.data_processor.load_and_sort_files(current_group)

            if not tiff_files:
                self.folder_path_label.setText("文件夹中没有目标TIFF文件")
                return

            # 读取所有图像
            self.data = self.data_processor.process_files(tiff_files, self.time_start_input, self.time_unit)

            if not self.data:
                self.folder_path_label.setText("无法读取TIFF文件")
                return

            # 设置时间滑块
            self.time_slider.setMaximum(len(self.data['images']) - 1)
            self.time_label.setText(f"时间点: 0/{len(self.data['images']) - 1}")

            # 显示第一张图像
            self.update_time_slice(0)

    def show_bad_frame_dialog(self):
        """显示坏点处理对话框"""
        if not hasattr(self, 'data') or self.data is None:
            logging.warning("无数据，请先加载数据文件")
            return

        dialog = BadFrameDialog(self)
        if dialog.exec_():
            # 更新图像显示
            self.update_time_slice(0)
            logging.info(f"坏点处理完成，修复了 {len(dialog.bad_frames)} 个坏帧")

    def update_time_slice(self, idx):
        """更新时间切片显示"""
        if self.data is not None and 0 <= idx < len(self.data['images']):
            self.time_label.setText(f"时间点: {idx}/{len(self.data['images']) - 1}")
            # zoom = self.image_display.zoom_spinbox.value()
            self.image_display.current_image = self.data['images'][idx]
            self.image_display.display_image(self.data['images'][idx])

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

    def region_analyze_start(self):
        """分析选定区域载流子寿命"""
        if self.data is None or not hasattr(self, 'time_points'):
            return logging.warning('无数据载入')
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
        # 如果线程没了，要创建
        if not self.is_thread_active("thread"):
            self.thread_open()


        self.thread.start()
        self.update_status('长时计算进行中...', True)
        self.time_unit = float(self.time_step_input.value())
        model_type = 'single' if self.model_combo.currentText() == "单指数衰减" else 'double'
        self.start_dis_cal_signal.emit(self.data,self.time_unit,model_type)

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
        if hasattr(self.result_display, 'current_data'):
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存图像", "", "PNG图像 (*.png);;JPEG图像 (*.jpg *.jpeg);;TIFF图像 (*.tif *.tiff)")

            if file_path:
                # 从matplotlib保存图像
                self.result_display.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            logging.info("图片已保存")

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
            logging.info("数据已保存")

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

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()