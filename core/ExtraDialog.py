import logging
from cProfile import label
from symtable import Class
from typing import List

from PyQt5.QtGui import QColor, QIntValidator, QFont
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QRadioButton, QSpinBox, QLineEdit, QPushButton,
                             QLabel, QMessageBox, QFormLayout, QDoubleSpinBox, QColorDialog, QComboBox, QCheckBox,
                             QFileDialog, QWhatsThis, QTextBrowser, QTableWidget, QDialogButtonBox, QTableWidgetItem,
                             QHeaderView, QAbstractItemView, QTabWidget, QWidget)
from PyQt5.QtCore import Qt, QEvent


# 坏帧处理对话框
class BadFrameDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("坏帧处理")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setMinimumWidth(400)

        self.init_ui()
        self.bad_frames = []

    def init_ui(self):
        layout = QVBoxLayout()

        # 检测方法选择
        method_group = QGroupBox("坏帧检测方法")
        method_layout = QHBoxLayout()

        self.auto_radio = QRadioButton("自动检测")
        self.auto_radio.setChecked(True)
        self.manual_radio = QRadioButton("手动输入")

        method_layout.addWidget(self.auto_radio)
        method_layout.addWidget(self.manual_radio)
        method_group.setLayout(method_layout)

        # 自动检测参数
        auto_group = QGroupBox("自动检测参数")
        auto_layout = QVBoxLayout()

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 10)
        self.threshold_spin.setValue(3)
        self.threshold_spin.setSuffix("σ")

        auto_layout.addWidget(QLabel("敏感度 (标准差倍数):"))
        auto_layout.addWidget(self.threshold_spin)
        auto_group.setLayout(auto_layout)

        # 手动输入
        manual_group = QGroupBox("手动选择坏帧")
        manual_layout = QVBoxLayout()

        self.frame_input = QLineEdit()
        self.frame_input.setPlaceholderText("输入坏帧位置，用逗号分隔 (如: 12,25,30)")

        manual_layout.addWidget(QLabel("坏帧位置:"))
        manual_layout.addWidget(self.frame_input)
        manual_group.setLayout(manual_layout)

        # 处理方法选择
        process_group = QGroupBox("坏帧处理方法")

        # 帧平均参数
        avg_group = QGroupBox("修复参数")
        avg_layout = QVBoxLayout()

        self.n_frames_spin = QSpinBox()
        self.n_frames_spin.setRange(1, 10)
        self.n_frames_spin.setValue(2)
        self.n_frames_spin.setSuffix("帧")

        avg_layout.addWidget(QLabel("前后平均帧数:"))
        avg_layout.addWidget(self.n_frames_spin)
        avg_group.setLayout(avg_layout)

        # 按钮
        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("应用修复")
        self.apply_btn.clicked.connect(self.apply_fix)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)

        # 组装布局
        layout.addWidget(method_group)
        layout.addWidget(auto_group)
        layout.addWidget(manual_group)
        layout.addWidget(avg_group)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # 初始状态
        self.auto_radio.toggled.connect(self.update_ui_state)
        self.update_ui_state()

    def event(self, event):
        if event.type() == QEvent.EnterWhatsThisMode:
            QWhatsThis.leaveWhatsThisMode()
            self.show_custom_help()  # 调用自定义弹窗方法
            return True
        return super().event(event)

    def show_custom_help(self):
        """显示自定义非模态帮助对话框"""
        help_title = "功能帮助说明"
        help_content = """
        <h3>主要功能说明</h3>
        <ul>
            <li><b>功能1</b>: 描述内容...</li>
            <li><b>功能2</b>: 描述内容...</li>
            <li><b>高级选项</b>: 点击<a href="https://example.com">这里</a>查看详情</li>
        </ul>
        <p><i>注：本帮助窗口不会阻塞主界面操作</i></p>
        """

        # 创建并显示自定义对话框
        help_dialog = CustomHelpDialog(help_title, help_content, self)
        help_dialog.show()  # 非阻塞显示

    def update_ui_state(self):
        """根据选择的方法更新UI状态"""
        auto_selected = self.auto_radio.isChecked()
        self.threshold_spin.setEnabled(auto_selected)
        self.frame_input.setEnabled(not auto_selected)

    def get_bad_frames(self) -> List[int] :
        """获取用户选择的坏帧列表"""
        if self.auto_radio.isChecked():
            return self.parent().data_processor.detect_bad_frames_auto(
                self.parent().data.origin,
                self.threshold_spin.value()
            )
        else:
            try:
                return [int(x.strip()) for x in self.frame_input.text().split(",") if x.strip()]
            except ValueError:
                QMessageBox.warning(self, "输入错误", "请输入有效的帧号，用逗号分隔")
                return []

    def apply_fix(self):
        """应用修复并关闭对话框"""
        self.bad_frames = self.get_bad_frames()
        if not self.bad_frames:
            QMessageBox.information(self, "无坏帧", "未检测到需要修复的坏帧")
            return

        n_frames = self.n_frames_spin.value()

        # 修复数据
        fixed_data = self.parent().data_processor.fix_bad_frames(
            self.parent().data.data_origin,
            self.bad_frames,
            n_frames
        )

        # 重新处理显示数据
        data_amend = self.parent().data_processor.amend_data(fixed_data)

        self.parent().data.update(data_amend)

        self.accept()
        QMessageBox.information(self, "修复完成", f"已修复 {len(self.bad_frames)} 个坏帧")
        logging.info(f"已修复 {len(self.bad_frames)} 个坏帧")

# 计算设置对话框
class CalculationSetDialog(QDialog):
    def __init__(self,params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("计算设置")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setMinimumWidth(350)

        # 默认参数
        self.params = params

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 计算设置组
        cal_set_group = QGroupBox('计算方法修改')
        cal_set_layout = QFormLayout()
        self.from_start_cal = QCheckBox()
        self.from_start_cal.setChecked(self.params['from_start_cal'])
        cal_set_layout.addRow(QLabel('从头拟合(默认为从最大值拟合)'),self.from_start_cal)
        cal_set_group.setLayout(cal_set_layout)

        # R方设置组
        r2_group = QGroupBox("拟合质量筛选")
        r2_layout = QFormLayout()

        self.r2_spin = QDoubleSpinBox()
        self.r2_spin.setRange(0.0, 1.0)
        self.r2_spin.setValue(self.params['r_squared_min'])
        self.r2_spin.setSingleStep(0.01)
        self.r2_spin.setDecimals(3)

        r2_layout.addRow(QLabel("R²最小值:"), self.r2_spin)
        r2_group.setLayout(r2_layout)

        # 信号范围组
        peak_group = QGroupBox("信号幅值范围")
        peak_layout = QFormLayout()

        self.peak_min_spin = QDoubleSpinBox()
        self.peak_min_spin.setRange(-1e8, 1e2)
        self.peak_min_spin.setValue(self.params['peak_min'])
        self.peak_min_spin.setSingleStep(0.1)

        self.peak_max_spin = QDoubleSpinBox()
        self.peak_max_spin.setRange(-1e2, 1e8)
        self.peak_max_spin.setValue(self.params['peak_max'])
        self.peak_max_spin.setSingleStep(0.1)

        peak_layout.addRow(QLabel("最小值:"), self.peak_min_spin)
        peak_layout.addRow(QLabel("最大值:"), self.peak_max_spin)
        peak_group.setLayout(peak_layout)

        # 寿命范围组
        tau_group = QGroupBox("寿命τ值范围 (ps)")
        tau_layout = QFormLayout()

        self.tau_min_spin = QDoubleSpinBox()
        self.tau_min_spin.setRange(0e-6, 1e6)
        self.tau_min_spin.setValue(self.params['tau_min'])
        self.tau_min_spin.setSingleStep(1e-3)
        self.tau_min_spin.setDecimals(6)

        self.tau_max_spin = QDoubleSpinBox()
        self.tau_max_spin.setRange(1e-6, 1e10)
        self.tau_max_spin.setValue(self.params['tau_max'])
        self.tau_max_spin.setSingleStep(1e2)
        self.tau_max_spin.setDecimals(6)

        tau_layout.addRow(QLabel("最小值:"), self.tau_min_spin)
        tau_layout.addRow(QLabel("最大值:"), self.tau_max_spin)
        tau_group.setLayout(tau_layout)

        # 按钮组
        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("应用")
        self.apply_btn.clicked.connect(self.apply_settings)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)

        # 组装布局
        layout.addWidget(cal_set_group)
        layout.addWidget(r2_group)
        layout.addWidget(peak_group)
        layout.addWidget(tau_group)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def apply_settings(self):
        """收集参数并关闭对话框"""
        self.params = {
            'from_start_cal': self.from_start_cal.isChecked(),
            'r_squared_min': self.r2_spin.value(),
            'peak_min': self.peak_min_spin.value(),
            'peak_max': self.peak_max_spin.value(),
            'tau_min': self.tau_min_spin.value(),
            'tau_max': self.tau_max_spin.value()
        }
        self.accept()

# 绘图设置对话框
class PltSettingsDialog(QDialog):
    def __init__(self, params,parent=None):
        super().__init__(parent)
        self.setWindowTitle("绘图设置")
        self.setMinimumWidth(400)
        self.setMinimumHeight(500)

        # 默认参数
        self.params = params
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 绘图类型选择
        type_group = QGroupBox("绘图类型")
        type_layout = QHBoxLayout()

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["寿命热图", "区域寿命曲线"])
        self.plot_type_combo.currentIndexChanged.connect(self.update_ui)

        type_layout.addWidget(QLabel("绘图模式:"))
        type_layout.addWidget(self.plot_type_combo)
        type_layout.addStretch()
        type_group.setLayout(type_layout)

        # 通用绘图设置
        common_group = QGroupBox("通用设置")
        common_layout = QFormLayout()

        self.color_btn = QPushButton()
        self.color_btn.setStyleSheet(f"background-color: {self.params['color']}")
        self.color_btn.clicked.connect(self.choose_color)

        self.grid_check = QCheckBox()
        self.grid_check.setChecked(self.params['show_grid'])

        self.axis_set = QCheckBox()
        self.axis_set.setChecked(self.params['set_axis'])

        common_layout.addRow(QLabel("线条颜色:"), self.color_btn)
        common_layout.addRow(QLabel("显示网格:"), self.grid_check)
        common_layout.addRow(QLabel('设置轴范围'), self.axis_set)
        common_group.setLayout(common_layout)

        # 曲线图特有设置
        self.curve_group = QGroupBox("曲线图设置")
        curve_layout = QFormLayout()

        self.line_style_combo = QComboBox()
        self.line_style_combo.addItems(["实线 -", "虚线 --", "点线 :", "点划线 -."])
        self.line_style_combo.setCurrentIndex(1)

        self.line_width_spin = QSpinBox()
        self.line_width_spin.setRange(1, 10)
        self.line_width_spin.setValue(self.params['line_width'])

        self.marker_combo = QComboBox()
        self.marker_combo.addItems(["无", "圆形 o", "方形 s", "三角形 ^", "星号 *"])
        self.marker_combo.setCurrentIndex(2)

        self.marker_size_spin = QSpinBox()
        self.marker_size_spin.setRange(1, 20)
        self.marker_size_spin.setValue(self.params['marker_size'])

        curve_layout.addRow(QLabel("线条样式:"), self.line_style_combo)
        curve_layout.addRow(QLabel("线条宽度:"), self.line_width_spin)
        curve_layout.addRow(QLabel("标记样式:"), self.marker_combo)
        curve_layout.addRow(QLabel("标记大小:"), self.marker_size_spin)
        self.curve_group.setLayout(curve_layout)

        # 热图特有设置
        self.heatmap_group = QGroupBox("热图设置")
        heatmap_layout = QFormLayout()

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["Jet", "Plasma", "Inferno", "Magma", "Viridis"])

        self.contour_spin = QSpinBox()
        self.contour_spin.setRange(0, 50)
        self.contour_spin.setValue(self.params['contour_levels'])
        self.contour_spin.setSpecialValueText("无等高线")

        heatmap_layout.addRow(QLabel("颜色映射:"), self.cmap_combo)
        heatmap_layout.addRow(QLabel("等高线级别:"), self.contour_spin)
        self.heatmap_group.setLayout(heatmap_layout)

        # 按钮组
        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("应用")
        self.apply_btn.clicked.connect(self.apply_settings)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)

        # 组装布局
        layout.addWidget(type_group)
        layout.addWidget(common_group)
        layout.addWidget(self.curve_group)
        layout.addWidget(self.heatmap_group)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.update_ui()  # 初始化UI状态

    def update_ui(self):
        """根据绘图类型显示/隐藏相关设置"""
        is_curve = self.plot_type_combo.currentText() == "区域寿命曲线"
        self.curve_group.setVisible(is_curve)
        self.heatmap_group.setVisible(not is_curve)

    def choose_color(self):
        """选择颜色"""
        color = QColorDialog.getColor(QColor(self.params['color']), self)
        if color.isValid():
            self.params['color'] = color.name()
            self.color_btn.setStyleSheet(f"background-color: {self.params['color']}")

    def apply_settings(self):
        """收集参数并关闭对话框"""
        self.params = {
            'current_mode': 'curve' if self.plot_type_combo.currentText() == "区域寿命曲线" else 'heatmap',
            'line_style': ['-', '--', ':', '-.'][self.line_style_combo.currentIndex()],
            'line_width': self.line_width_spin.value(),
            'marker_style': ['', 'o', 's', '^', '*'][self.marker_combo.currentIndex()],
            'marker_size': self.marker_size_spin.value(),
            'color': self.params['color'],
            'show_grid': self.grid_check.isChecked(),
            'heatmap_cmap': ['jet', 'plasma', 'inferno', 'magma', 'viridis'][self.cmap_combo.currentIndex()],
            'contour_levels': self.contour_spin.value(),
            'set_axis':self.axis_set.isChecked()
        }
        self.accept()

# 数据保存弹窗
class DataSavingPop(QDialog):
    def __init__(self,parent = None):
        super().__init__(parent)
        self.setWindowTitle("数据保存")
        self.setMinimumWidth(300)
        self.setMinimumHeight(200)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 创建水平布局容器来放置标签和复选框
        def create_checkbox_row(label_text, checkbox):
            row_layout = QHBoxLayout()
            label = QLabel(label_text)
            row_layout.addWidget(label)
            row_layout.addStretch()  # 添加弹性空间使复选框靠右
            row_layout.addWidget(checkbox)
            return row_layout

        # 是否拟合
        self.fitting_check = QCheckBox()
        self.fitting_check.setChecked(True)
        # 是否加标题
        self.index_check = QCheckBox()
        self.index_check.setChecked(False)
        # 是否显示额外信息（未完成）
        self.extra_check = QCheckBox()
        self.extra_check.setChecked(False)

        # 添加各个复选框行
        layout.addLayout(create_checkbox_row("是否导出拟合数据:", self.fitting_check))
        layout.addLayout(create_checkbox_row("是否导出标题:(utf-8编码)", self.index_check))
        layout.addLayout(create_checkbox_row("是否显示额外信息（未完成）:", self.extra_check))

        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("导出")
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

# 计算stft参数弹窗
class STFTComputePop(QDialog):
    def __init__(self,params,case,parent = None):
        super().__init__(parent)
        self.setWindowTitle("短时傅里叶变换")
        self.setMinimumWidth(300)
        self.setMinimumHeight(200)
        self.params = params
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        self.target_freq_input = QDoubleSpinBox()
        self.target_freq_input.setRange(0.1, 10000)
        self.target_freq_input.setValue(self.params['target_freq'])
        self.target_freq_input.setSuffix(" Hz")

        self.fps_input = QSpinBox()
        self.fps_input.setRange(10,99999)
        self.fps_input.setValue(self.params['EM_fps'])

        self.window_size_input = QSpinBox()
        self.window_size_input.setRange(1, 65536)
        self.window_size_input.setValue(self.params['stft_window_size'])

        self.noverlap_input = QSpinBox()
        self.noverlap_input.setRange(0, 65536)
        self.noverlap_input.setValue(self.params['stft_noverlap'])

        self.custom_nfft_input = QSpinBox()
        self.custom_nfft_input.setRange(0, 65536)
        self.custom_nfft_input.setValue(self.params['custom_nfft'])

        layout.addRow(QLabel("目标频率"),self.target_freq_input)
        layout.addRow(QLabel("采样帧率"),self.fps_input)
        layout.addRow(QLabel("窗口大小"),self.window_size_input)
        layout.addRow(QLabel("窗口重叠"),self.noverlap_input)
        layout.addRow(QLabel("变换长度"),self.custom_nfft_input)


        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("执行STFT")
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.setLayout(5,QFormLayout.FieldRole,button_layout)

        self.setLayout(layout)

# 计算cwt参数弹窗
class CWTComputePop(QDialog):
    def __init__(self,params,case='quality',parent = None):
        super().__init__(parent)
        self.setWindowTitle("小波变换")
        self.setMinimumWidth(300)
        self.setMinimumHeight(200)
        self.params = params
        self.case = case
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        self.target_freq_input = QDoubleSpinBox()
        self.target_freq_input.setRange(0.1, 1000)
        self.target_freq_input.setValue(self.params['target_freq'])
        self.target_freq_input.setSuffix(" Hz")

        self.fps_input = QSpinBox()
        self.fps_input.setRange(100,99999)
        self.fps_input.setValue(self.params['EM_fps'])

        self.cwt_size_input = QSpinBox()
        self.cwt_size_input.setRange(0, 65536)
        self.cwt_size_input.setValue(self.params['cwt_total_scales'])

        self.wavelet = QComboBox()
        self.wavelet.addItems(['cmor3-3','cmor1.5-1.0','morl(实)','cgau8'])
        self.wavelet.setCurrentText(self.params['cwt_type'])

        layout.addRow(QLabel("目标频率"),self.target_freq_input)
        layout.addRow(QLabel("小波类型"),self.wavelet)
        layout.addRow(QLabel("采样帧率"),self.fps_input)
        layout.addRow(QLabel("计算尺度"),self.cwt_size_input)

        if self.case == 'signal':
            self.cwt_scale_range = QDoubleSpinBox()
            self.cwt_scale_range.setRange(0.1, 1000)
            self.cwt_scale_range.setValue(self.params['cwt_scale_range'])
            self.cwt_scale_range.setSuffix(" Hz")
            layout.addRow(QLabel("处理跨度"), self.cwt_scale_range)

        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("执行CWT")
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.setLayout(5,QFormLayout.FieldRole,button_layout)

        self.setLayout(layout)

# EM时频变换数据导出
class MassDataSavingPop(QDialog):
    def __init__(self, parent=None, datatypes=None):
        super().__init__(parent)
        self.directory = None
        self.setWindowTitle("数据导出")
        self.setMinimumWidth(360)  # 加宽以适应新控件
        self.setMinimumHeight(260)
        # self.datatypes = datatypes if datatypes else []
        self.datatypes = ['tif','avi','gif','png']

        # 创建字体对象
        # font = QFont("Segoe UI", 10)  # 使用更现代化的字体
        # self.setFont(font)

        self.init_ui()
        self.setStyleSheet(self.style_sheet())  # 设置样式表

    def style_sheet(self):
        """返回美化界面的样式表"""
        return """
            QLineEdit, QComboBox {
                background-color: white;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 5px;
                min-height: 25px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 3px;
                padding: 4px 6px;
                font-weight: 450;
                min-width: 50px;
            }
            QPushButton#cancel {
                background-color: #6c757d;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton#cancel:hover {
                background-color: #5a6268;
            }
        """

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 1. 路径选择组件
        path_layout = QFormLayout()
        path_label = QLabel("保存路径:")
        self.path_edit = QLineEdit()
        path_layout.addRow(path_label,self.path_edit)
        self.path_edit.setPlaceholderText("选择或输入文件保存路径")
        browse_btn = QPushButton("浏览文件夹")
        browse_btn.clicked.connect(self.browse_directory)
        path_layout.addRow(QLabel(""),browse_btn)

        # 2. 文本输入框
        text_layout = QHBoxLayout()
        text_label = QLabel("文件名称:")
        self.text_edit = QLineEdit()
        self.text_edit.setPlaceholderText("请输入文件名（前缀）")
        text_layout.addWidget(text_label)
        text_layout.addWidget(self.text_edit)

        # 3. 动态数据类型选择器
        type_layout = QHBoxLayout()
        type_label = QLabel("数据类型:")
        self.type_combo = QComboBox()
        self.update_datatype(self.datatypes)  # 初始化下拉菜单
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_combo)

        # 将三个部分添加到主布局
        main_layout.addLayout(type_layout)
        main_layout.addWidget(QLabel('注意：使用avi/gif/png，会对数据有压缩！'))
        main_layout.addLayout(path_layout)
        main_layout.addLayout(text_layout)

        # 4. 确认/取消按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.setObjectName("cancel")
        cancel_btn.clicked.connect(self.reject)
        confirm_btn = QPushButton("确认导出")
        confirm_btn.clicked.connect(self.accept)

        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(confirm_btn)

        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

    def update_datatype(self, datatypes):
        """动态更新数据类型下拉菜单"""
        self.type_combo.clear()
        if datatypes:
            self.type_combo.addItems(datatypes)
        else:
            self.type_combo.addItem("无可用格式")
            self.type_combo.setEnabled(False)

    def browse_directory(self):
        """打开文件夹选择对话框"""
        self.directory = QFileDialog.getExistingDirectory(
            self,
            "选择保存文件夹",
            options=QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if self.directory:
            self.path_edit.setText(self.directory)

    def get_values(self):
        """获取用户输入的值"""
        return {
            'path': self.path_edit.text().strip(),
            'filename': self.text_edit.text().strip(),
            'filetype': self.type_combo.currentText()
        }


class DataViewAndSelectPop(QDialog):
    def __init__(self, parent=None, datadict=None, processed_datadict=None, add_canvas=False):
        super().__init__(parent)
        self.datadict = datadict or []
        self.processed_datadict = processed_datadict or []
        self.add_canvas = add_canvas

        self.selected_timestamp = None
        self.selected_index = -1
        self.selected_name = ""
        self.selected_table = None  # 记录选择来自哪个表格

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('数据选择')
        self.setMinimumSize(800, 500)  # 增加对话框尺寸以适应多个表格

        # 创建主布局
        main_layout = QVBoxLayout(self)

        # 创建选项卡容器
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget, 3)  # 选项卡占据大部分空间

        # 根据传入的数据创建表格
        self.tables = []

        if self.datadict != []:
            self.create_table_tab(self.datadict, "原始数据")

        if self.processed_datadict != []:
            self.create_table_tab(self.processed_datadict, "处理后数据")

        # 如果没有数据，显示提示信息
        if not self.tables:
            no_data_label = QLabel("没有可显示的数据")
            no_data_label.setAlignment(Qt.AlignCenter)
            self.tab_widget.addTab(no_data_label, "无数据")

        # 创建底部状态显示区域
        bottom_layout = QHBoxLayout()
        self.status_label = QLabel("目前选择的数据：")
        self.selected_data_label = QLabel("暂无")
        bottom_layout.addWidget(self.status_label)
        bottom_layout.addWidget(self.selected_data_label)
        bottom_layout.addStretch()

        main_layout.addLayout(bottom_layout)

        # 创建按钮框
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)

        # 根据add_canvas设置按钮状态
        if self.add_canvas:
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
            self.button_box.button(QDialogButtonBox.Ok).setText("确定")
        else:
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)
            self.button_box.button(QDialogButtonBox.Ok).setText("选择")

        main_layout.addWidget(self.button_box)

        # 连接按钮信号
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def create_table_tab(self, data_list, tab_name):
        """创建表格并添加到选项卡"""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        table = QTableWidget()
        tab_layout.addWidget(table)

        # 配置表格
        self.setup_table(table, data_list)

        # 添加到选项卡
        self.tab_widget.addTab(tab, tab_name)
        self.tables.append(table)

        return table

    def setup_table(self, table, data_list):
        """设置表格内容和按钮"""
        num_rows = len(data_list)
        if num_rows > 0:
            all_keys = list(data_list[0].keys())
            keys = [key for key in all_keys if key not in ['timestamp']]
            num_cols = len(keys) + 1  # 增加一列用于放置按钮
        else:
            keys = []
            num_cols = 0

        # 设置表格行数、列数和表头
        table.setRowCount(num_rows)
        table.setColumnCount(num_cols)
        if num_rows > 0:
            column_headers = keys + ['操作']  # 添加一个"操作"列
            table.setHorizontalHeaderLabels(column_headers)

        # 填充数据并插入按钮
        for row_idx, data_dict in enumerate(data_list):
            for col_idx, key in enumerate(keys):
                value = str(data_dict.get(key, ''))
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row_idx, col_idx, item)

                # 设置ToolTip
                if row_idx == 0:
                    item.setToolTip('当前数据（最新）')
                    item.setBackground(QColor(212, 237, 205))
                else:
                    item.setToolTip('历史数据')

            # 在最后一列创建并设置按钮
            button_text = "显示选择" if self.add_canvas else "设为当前"
            button = QPushButton(button_text)

            # 使用lambda表达式捕获当前行索引和表格
            button.clicked.connect(lambda checked, r=row_idx, t=table: self.on_row_button_clicked(r, t))
            table.setCellWidget(row_idx, num_cols - 1, button)

        # 调整列宽以自适应内容
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setSelectionBehavior(QTableWidget.SelectRows)  # 单选整行

        # 连接单元格点击事件
        table.cellClicked.connect(lambda row, col, t=table: self.on_cell_clicked(row, col, t))

    def on_row_button_clicked(self, row_index, table):
        """处理行按钮点击事件"""
        # 确定数据来自哪个表格
        table_index = self.tables.index(table)
        data_list = self.datadict if table_index == 0 and self.datadict !=[] else self.processed_datadict

        self.selected_index = row_index
        selected_data = data_list[row_index]
        self.selected_table = table_index

        # 获取名称和时间戳
        name = selected_data.get('name')
        self.selected_name = str(name) if name else f"数据{row_index + 1}"
        self.selected_timestamp = selected_data.get('timestamp')

        # 更新状态显示
        self.selected_data_label.setText(self.selected_name)

        # 如果add_canvas为True，执行显示操作
        if self.add_canvas:
            self.on_show_selected()
            # 启用确定按钮
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)

    def on_cell_clicked(self, row_index, col_index, table):
        """处理单元格点击事件"""
        # 只处理数据列的点击，忽略按钮列的点击
        if col_index < table.columnCount() - 1:
            self.on_row_button_clicked(row_index, table)

    def on_show_selected(self):
        """当add_canvas为True时，显示选择的画布（留空）"""
        # 这里可以添加显示画布的逻辑
        pass

    def get_selected_timestamp(self):
        """获取选择的数据信息"""
        return self.selected_timestamp

# 帮助dialog
class CustomHelpDialog(QDialog):
    """自定义非模态帮助对话框"""

    def __init__(self, title, content, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        self.setAttribute(Qt.WA_DeleteOnClose)  # 关闭时自动释放

        # 创建布局和控件
        layout = QVBoxLayout()

        # 标题标签
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # 富文本内容区域（支持HTML格式）
        content_browser = QTextBrowser()
        content_browser.setHtml(content)
        content_browser.setOpenExternalLinks(True)  # 允许打开外部链接
        layout.addWidget(content_browser)

        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        self.setLayout(layout)
        self.resize(250, 500)  # 设置初始大小