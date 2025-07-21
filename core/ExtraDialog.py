import logging
from cProfile import label
from symtable import Class
from typing import List

from PyQt5.QtGui import QColor, QIntValidator
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QRadioButton, QSpinBox, QLineEdit, QPushButton,
                             QLabel, QMessageBox, QFormLayout, QDoubleSpinBox, QColorDialog, QComboBox, QCheckBox)
from PyQt5.QtCore import Qt

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

    def update_ui_state(self):
        """根据选择的方法更新UI状态"""
        auto_selected = self.auto_radio.isChecked()
        self.threshold_spin.setEnabled(auto_selected)
        self.frame_input.setEnabled(not auto_selected)

    def get_bad_frames(self) -> List[int] :
        """获取用户选择的坏帧列表"""
        if self.auto_radio.isChecked():
            return self.parent().data_processor.detect_bad_frames_auto(
                self.parent().data['data_origin'],
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
            self.parent().data['data_origin'],
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
        self.peak_min_spin.setRange(0, 1e2)
        self.peak_min_spin.setValue(self.params['peak_min'])
        self.peak_min_spin.setSingleStep(0.1)

        self.peak_max_spin = QDoubleSpinBox()
        self.peak_max_spin.setRange(0, 1e8)
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
    def __init__(self,parent = None):
        super().__init__(parent)
        self.setWindowTitle("短时傅里叶变换")
        self.setMinimumWidth(300)
        self.setMinimumHeight(200)

        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        self.target_freq_input = QDoubleSpinBox()
        self.target_freq_input.setRange(0.1, 100)
        self.target_freq_input.setValue(30.0)
        self.target_freq_input.setSuffix(" Hz")

        self.fs_input = QSpinBox()
        self.fs_input.setRange(100,9999)
        self.fs_input.setValue(300)

        self.window_size_input = QSpinBox()
        self.window_size_input.setRange(16, 2048)
        self.window_size_input.setValue(128)

        self.noverlap_input = QSpinBox()
        self.noverlap_input.setRange(0, 1024)
        self.noverlap_input.setValue(120)

        self.custom_nfft_input = QSpinBox()
        self.custom_nfft_input.setRange(0, 1024)
        self.custom_nfft_input.setValue(256)

        layout.addRow(QLabel("目标频率"),self.target_freq_input)
        layout.addRow(QLabel("采样频率"),self.fs_input)
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
    def __init__(self,parent = None):
        super().__init__(parent)
        self.setWindowTitle("小波变换")
        self.setMinimumWidth(300)
        self.setMinimumHeight(200)

        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        self.target_freq_input = QDoubleSpinBox()
        self.target_freq_input.setRange(0.1, 100)
        self.target_freq_input.setValue(30.0)
        self.target_freq_input.setSuffix(" Hz")

        self.fs_input = QSpinBox()
        self.fs_input.setRange(100,9999)
        self.fs_input.setValue(360)

        self.cwt_size_input = QSpinBox()
        self.cwt_size_input.setRange(16, 2048)
        self.cwt_size_input.setValue(256)

        self.wavelet = QComboBox()
        self.wavelet.addItems(['morl','cmor3-3','cmor1.5-1.0','cgau8'])

        layout.addRow(QLabel("目标频率"),self.target_freq_input)
        layout.addRow(QLabel("小波类型"),self.wavelet)
        layout.addRow(QLabel("采样频率"),self.fs_input)
        layout.addRow(QLabel("小波尺寸"),self.cwt_size_input)



        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("执行CWT")
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.setLayout(5,QFormLayout.FieldRole,button_layout)

        self.setLayout(layout)