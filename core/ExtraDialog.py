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
from PyQt5.QtCore import Qt, QEvent, QTimer
import HelpContentHTML
import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.fenced_code import FencedCodeExtension
import re


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
        self.help_dialog = None
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

        self.scale_range_input = QSpinBox()
        self.scale_range_input.setRange(0,99999)
        self.scale_range_input.setValue(self.params['stft_scale_range'])

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
        layout.addRow(QLabel("平均范围"),self.scale_range_input)
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
        layout.setLayout(6,QFormLayout.FieldRole,button_layout)

        self.setLayout(layout)

    def event(self, event):
        if event.type() == QEvent.EnterWhatsThisMode:
            # QWhatsThis.leaveWhatsThisMode()
            QTimer.singleShot(0, self.show_custom_help)
            return True
        return super().event(event)

    def show_custom_help(self):
        """显示自定义非模态帮助对话框"""
        QWhatsThis.leaveWhatsThisMode()
        help_title = "STFT 帮助说明"
        help_content = r"""
        <!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; }
        h1 { color: #2e7d32; 
            border-bottom: 2px solid #4caf50; }
        h2 { color: #388e3c; }
        h3 { color: black; }
        .code-block { 
            background-color: #e8f5e9; 
            border: 1px solid #c8e6c9;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            overflow-x: auto;
        }
        .param-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        .param-table th {
            background-color: #4caf50;
            color: white;
            text-align: left;
            padding: 8px;
        }
        .param-table td {
            border: 1px solid #c8e6c9;
            padding: 8px;
        }
        .param-table tr:nth-child(even) {
            background-color: #f1f8e9;
        }
        .note {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
        }
        .math-block {
            font-size: large;
            background-color: #f9fbe7;
            border: 1px solid #dcedc8;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <h1>STFT（短时傅里叶变换）分析</h1>
    
    <h2>1. 功能概述</h2>
    <p>STFT（Short-Time Fourier Transform）是一种时频分析方法，用于分析信号频率随时间的变化特性。本实现针对电化学调制数据逐像素执行STFT分析，分析数据中周期性变化的频率特性。</p>
    
    <h2>2. 算法原理</h2>
    <p>STFT通过在信号上滑动窗口，对每个窗口内的信号段进行傅里叶变换，从而获得信号在时间和频率上的联合分布：</p>
    <div class="math-block">
    X(τ, ω) = ∫<sub>-∞</sub><sup>∞</sup> x(t) w(t-τ) e<sup>-jωt</sup> dt
    </div>
    <p>其中：</p>
    <ul>
        <li>x(t)：输入信号</li>
        <li>w(t)：窗函数</li>
        <li>τ：时间位置</li>
        <li>ω：角频率</li>
    </ul>
    
    <h2>3. 参数说明</h2>
    <table class="param-table">
        <tr>
            <th>参数</th>
            <th>默认值</th>
            <th>说明</th>
        </tr>
        <tr>
            <td>窗函数类型</td>
            <td>hann</td>
            <td>窗函数类型(如'hann'汉宁窗, 'hamming'汉明窗等，详情查看第七部分)</td>
        </tr>
        <tr>
            <td>目标频率</td>
            <td>30.0 Hz</td>
            <td>想要提取的目标频率</td>
        </tr>
        <tr>
            <td>平均范围</td>
            <td>0 Hz</td>
            <td>默认为0，不平均，否则以目标频率为中心，平均范围内包含的stft结果</td>
        </tr>
        <tr>
            <td>采样帧率</td>
            <td>360 (帧/秒)</td>
            <td>你提供的视频or时序数据所采样的帧率（影响拍摄时长）</td>
        </tr>
        <tr>
            <td>窗口大小</td>
            <td>128 (点数)</td>
            <td>进行短时傅里叶加窗的长度</td>
        </tr>
        <tr>
            <td>窗口重叠</td>
            <td>120 (点数)</td>
            <td>相邻窗重复的长度<br>（步长=窗口大小-窗口重叠）</td>
        </tr>
        <tr>
            <td>变换长度</td>
            <td>360 (点数)</td>
            <td>即参与变换的点数，最小取窗口大小，影响频率点数（非频率分辨率）</td>
        </tr>
    </table>
    
    <h2>4. 处理流程</h2>
    <ol>
        <li>导入原始数据（支持avi和tiff序列）</li>
        <li>预处理数据（去背景、展数据等操作）</li>
        <li>进行质量评价（得到）</li>
        <li>逐像素执行STFT：
            <ul>
                <li>提取像素时间序列</li>
                <li>应用窗函数</li>
                <li>计算STFT</li>
                <li>提取目标频率幅度</li>
            </ul>
        </li>
        <li>得到目标频率处具有一定时频分辨率的幅值序列</li>
    </ol>
    <div class="note">
        <strong>注意：</strong>使用前需先执行<i>预处理</i>和<i>质量评估</i>两个步骤
    </div>
    <h2>5. 输出结果</h2>
    <p>质量评价的结果为功率谱密度（PSD）</p>
    <p>STFT处理结果为时序幅值图像（可显示）：</p>
    <div class="code-block">
        stft_py_out[time_index, y_coord, x_coord]
    </div>
    <p>其中：</p>
    <ul>
        <li><strong>time_index</strong>：处理后时间</li>
        <li><strong>y_coord</strong>：像素Y坐标</li>
        <li><strong>x_coord</strong>：像素X坐标</li>
    </ul>
    <h2>6. STFT 时频分辨率浅析</h2>
    <h3>窗口大小（窗长）</h3>
    <p>
        <ul> 
            <li>较长的窗口 → 频率分辨率高（能区分更接近的频率成分），但时间分辨率低（无法精确定位快速变化的瞬态信号）。</li>
            <li>较短的窗口 → 时间分辨率高（能捕捉快速变化），但频率分辨率低（频率模糊）。
        </ul>
        这是由<strong>海森堡不确定性原理</strong>决定的固有权衡。<strong>gabor窗</strong>的特殊之处就在于，它满足了不确定性原理的最下限，保证了时域和频域同时最集中，是使时频图分辨率最高的窗函数。</p>
    <h3>采样率（采样帧率）</h3>
    <p>采样率决定了信号的最高可分析频率（Nyquist频率 = 采样率/2），
        但不会改变STFT的时频分辨率。
        <br>例如：若采样率翻倍，Nyquist频率提高，但窗口长度（以样本点计）不变时，
        实际时间窗口的持续时间（秒）会缩短（因为样本点间隔更小），
        从而可能间接影响时间分辨率。</p>
    <h3>变换步长（窗口重叠）</h3>
    <p>变换步长不会影响实际的时间分辨率，但会决定STFT结果的时间分辨能力。<br>
        <ul>
            <li>较小的步长（高重叠） → 时间采样更密集，但计算量更大。</li>
            <li>较大的步长（低重叠） → 时间采样更稀疏，计算量更小。</li></ul>
        </p>
    <h3>变换长度（nfft）</h3>
    <p>变换长度不会改变实际的频率分辨率，但决定了频率分辨能力（频率点数）。
        <ul>
            <li>较大的变换长度 → 频率点数更多，但计算量更大。</li>
            <li>较小的变换长度 → 频率点数更少，计算量更小。</li></ul>
        当变换长度≥窗口长度时，在计算中尾部会采取补零的操作，频谱插值更平滑，但不会增加真实频率信息。<br>不过在实际测试中，会对结果数值产生一定影响。</p>
    <h2>7. 窗函数介绍</h2>
    <table class="param-table">
        <tr>
            <th>窗函数</th>
            <th>名称</th>
            <th>说明</th>
        </tr>
        <tr>
            <td>hann</td>
            <td>汉宁窗</td>
            <td>默认窗，主瓣较宽，快滚降，频谱泄漏适中</td>
        </tr>
        <tr>
            <td>hamming</td>
            <td>汉明窗</td>
            <td>主瓣适中，旁瓣较低，慢滚降，频谱泄漏适中</td>
        </tr>
        <tr>
            <td>gaussian</td>
            <td>gabor窗</td>
            <td>时间频率分辨率达到理论极限（不确定性原理），旁瓣较低，低泄漏，滚降一般</td>
        </tr>

        <tr>
            <td>boxcar</td>
            <td>矩形窗</td>
            <td>所有点权重相等，主瓣最窄，频谱泄漏严重，滚降最慢</td>
        </tr>
        <tr>
            <td>blackman</td>
            <td>Blackman窗</td>
            <td>主瓣宽，旁瓣低，频谱泄漏很小，滚降快</td>
        <tr>
            <td>blackmanharris</td>
            <td>BH窗</td>
            <td>主瓣超宽，旁瓣极低，频谱泄漏很小，滚降超快</td>
        </tr>
        
    </table>
    <p><i>附：程序使用<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html#scipy.signal.stft">
        scipy.signal.stft</a>方法进行运算</i></p>
</body>
</html>
        """

        # 创建并显示自定义对话框
        self.help_dialog = CustomHelpDialog(help_title, 'custom',help_content)
        self.help_dialog.setWindowModality(Qt.NonModal)
        if self.help_dialog.exec_():
            return True
        return None
        # self.help_dialog.show()  # 非阻塞显示
        # self.help_dialog.activateWindow()
        # self.help_dialog.raise_()

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
        if self.case == 'quality':
            self.cwt_size_input.setValue(256)
        else:
            self.cwt_size_input.setValue(1)

        self.wavelet = QComboBox()
        self.wavelet.addItems(['cmor3-3','cmor1.5-1.0','cgau8','mexh','morl'])
        self.wavelet.setCurrentText(self.params['cwt_type'])

        layout.addRow(QLabel("目标频率"),self.target_freq_input)
        layout.addRow(QLabel("小波类型"),self.wavelet)
        layout.addRow(QLabel("采样帧率"),self.fps_input)
        layout.addRow(QLabel("计算尺度"),self.cwt_size_input)

        self.cwt_scale_range = QDoubleSpinBox()
        self.cwt_scale_range.setRange(0, 10000)
        self.cwt_scale_range.setValue(self.params['cwt_scale_range'])
        self.cwt_scale_range.setSuffix(" Hz")
        layout.addRow(QLabel("处理跨度"), self.cwt_scale_range)

        if self.case == 'signal':
            self.apply_btn = QPushButton("执行CWT")
        else:
            self.apply_btn = QPushButton("执行质量评价")

        button_layout = QHBoxLayout()
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.setLayout(5,QFormLayout.FieldRole,button_layout)

        self.setLayout(layout)

# 单通道信号参数弹窗
class SCSComputePop(QDialog):
    def __init__(self,params,parent = None):
        super().__init__(parent)
        self.setWindowTitle("单通道模式")
        self.setMinimumWidth(300)
        self.setMinimumHeight(120)
        self.params = params
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        self.thr_known_check = QCheckBox()
        self.thr_known_check.setChecked(self.params['thr_known'])

        self.thr_input = QDoubleSpinBox()
        self.thr_input.setRange(0.1, 1000)
        self.thr_input.setValue(self.params['scs_thr'])

        self.zoom_input = QSpinBox()
        self.zoom_input.setRange(0,100)
        self.zoom_input.setValue(self.params['scs_zoom'])


        layout.addRow(QLabel("阈值是否已知"), self.thr_known_check)
        layout.addRow(QLabel("阈值设置"),self.thr_input)
        layout.addRow(QLabel("插值倍数"),self.zoom_input)
        layout.setSpacing(10)


        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("执行")
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.setLayout(3,QFormLayout.FieldRole,button_layout)

        self.setLayout(layout)

        self.thr_known_check.toggled.connect(self.update_thr_state)
        self.update_thr_state()

    def update_thr_state(self):
        thr_known = self.thr_known_check.isChecked()
        self.thr_input.setEnabled(thr_known)

# 视频与彩色图像导出
class DataExportDialog(QDialog):
    def __init__(self, parent=None, datatypes=None,export_type='EM',canvas_info=None,is_temporal = None):
        super().__init__(parent)
        self.directory = None
        if export_type == 'EM':
            self.setWindowTitle("数据导出(EM模式)")
        else:
            self.setWindowTitle("数据导出(画布模式)")
        self.setMinimumWidth(300)  # 加宽以适应新控件
        self.setMinimumHeight(450)
        # self.datatypes = datatypes if datatypes else []
        self.is_temporal = is_temporal
        self.datatypes = ['tif','avi','gif','png']
        self.canvas_info = canvas_info
        self.export_type = export_type

        self.init_ui()
        # self.setStyleSheet(self.style_sheet())  # 设置样式表

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

        # 0. 画布选择（画布导出模式）
        form_layout = QFormLayout()
        if self.export_type == 'canvas':
            self.canvas_selector = QComboBox()
            if self.canvas_info:
                for item in self.canvas_info:
                    self.canvas_selector.addItem(item)
            form_layout.addRow(QLabel("画布选择"),self.canvas_selector)
            self.canvas_selector.currentIndexChanged.connect(self.datatypes_change)
        else:
            pass

        # 1. 路径选择组件

        path_label = QLabel("保存路径:")
        self.path_edit = QLineEdit()
        form_layout.addRow(path_label,self.path_edit)
        self.path_edit.setPlaceholderText("选择或输入文件保存路径")
        browse_btn = QPushButton("浏览文件夹")
        browse_btn.clicked.connect(self.browse_directory)
        form_layout.addRow(QLabel(""),browse_btn)

        # 2. 文本输入框
        text_label = QLabel("文件名称:")
        self.text_edit = QLineEdit()
        self.text_edit.setPlaceholderText("请输入文件名（前缀）")
        form_layout.addRow(text_label,self.text_edit)

        # 3. 动态数据类型选择器
        type_label = QLabel("数据类型:")
        self.type_combo = QComboBox()
        self.update_datatype(self.datatypes)  # 初始化下拉菜单
        form_layout.addRow(type_label, self.type_combo)
        self.type_combo.currentIndexChanged.connect(self._update_type)

        # 4. 其他参数
        self.duration_label = QLabel("时长")
        self.duration_input = QSpinBox()
        self.duration_input.setRange(0,10000)
        self.duration_input.setValue(60)
        form_layout.addRow(self.duration_label, self.duration_input)
        self.duration_label.setVisible(False)
        self.duration_input.setVisible(False)

        main_layout.addLayout(form_layout)
        main_layout.addWidget(QLabel('注意：使用avi/gif/png，会对原始数据有压缩！'))

        # 5. 确认/取消按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.setObjectName("cancel")
        cancel_btn.clicked.connect(self.reject)
        confirm_btn = QPushButton("确认导出")
        confirm_btn.clicked.connect(self.accept)

        btn_layout.addWidget(confirm_btn)
        btn_layout.addWidget(cancel_btn)

        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

    def datatypes_change(self):
        if self.is_temporal[self.canvas_selector.currentIndex()]:
            self.datatypes = ['tif','avi','gif','png']
            self.update_datatype(self.datatypes)
        if not self.is_temporal[self.canvas_selector.currentIndex()]:
            self.datatypes = ['tif', 'png']
            self.update_datatype(self.datatypes)

    def update_datatype(self, datatypes):
        """动态更新数据类型下拉菜单"""
        self.type_combo.clear()
        if datatypes:
            self.type_combo.addItems(datatypes)
        else:
            self.type_combo.addItem("无可用格式")
            self.type_combo.setEnabled(False)

    def _update_type(self):
        """选择到处类型后发生什么"""
        self.current_type = self.type_combo.currentText()
        if self.current_type == 'gif':
            self.duration_label.setVisible(True)
            self.duration_input.setVisible(True)
        else:
            self.duration_label.setVisible(False)
            self.duration_input.setVisible(False)
            return

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
            'canvas': self.canvas_selector.currentIndex() if self.export_type == 'canvas' else None,
            'path': self.path_edit.text().strip(),
            'filename': self.text_edit.text().strip(),
            'filetype': self.type_combo.currentText(),
            'duration': self.duration_input.value() if self.current_type == 'gif' else None,
        }

# 数据选择查看视窗
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
    ALL_TOPICS = ["general","canvas", "stft", "cwt", "lifetime","whole","single"]
    def __init__(self, title, topics=None, content = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        self.setAttribute(Qt.WA_DeleteOnClose)  # 关闭时自动释放
        self.content = content
        # 创建布局和控件
        layout = QVBoxLayout()

        self.tab_widget = QTabWidget()

        # 添加帮助主题
        self.add_help_tabs(topics)

        layout.addWidget(self.tab_widget)

        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        self.setLayout(layout)
        self.resize(500, 700)  # 设置初始大小

    def add_help_tabs(self, topics):
        """添加指定的帮助主题标签页"""
        # 如果没有指定主题或列表为空，则显示所有主题
        if not topics:
            topics = self.ALL_TOPICS
        elif topics == "custom":
            self.add_tab(topics,self.content)

        # 添加每个主题的标签页
        for topic in topics:
            if topic in self.ALL_TOPICS:
                self.add_tab(topic)

    def add_tab(self, topic_key, html_text = None):
        """添加单个帮助主题标签页"""
        if topic_key == 'custom':
            text = html_text
            title = '方法帮助'
        elif topic_key not in HelpContentHTML.HELP_CONTENT:
            return
        else:
            topic = HelpContentHTML.HELP_CONTENT[topic_key]
            text = topic["html"]
            title = topic["title"]
        # 创建文本浏览器
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(text)

        # 添加到标签页
        self.tab_widget.addTab(browser, title)

    # def show_topic(self, topic_key):
    #     """显示特定主题并使其成为当前标签页"""
    #     if topic_key not in HelpContentHTML.HELP_CONTENT:
    #         return
    #
    #     # 查找标签页索引
    #     for i in range(self.tab_widget.count()):
    #         if self.tab_widget.tabText(i) == HelpContentHTML.HELP_CONTENT[topic_key]["title"]:
    #             self.tab_widget.setCurrentIndex(i)
    #             return
    #
    #     # 如果没找到，添加新标签页
    #     self.add_tab(topic_key)
    #     self.tab_widget.setCurrentIndex(self.tab_widget.count() - 1)

    # def _render_markdown(self, md_content):
    #     """将Markdown转换为HTML"""
    #     # 添加扩展支持代码高亮和围栏代码块
    #     extensions = [
    #         CodeHiliteExtension(noclasses=True),
    #         FencedCodeExtension()
    #     ]
    #     return markdown.markdown(md_content, extensions=extensions)

# 画布及roi查看和选择
class ROIInfoDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.roi_type = None
        self.setWindowTitle("图像与ROI信息（双击选择）")
        self.setMinimumSize(600, 400)

        self.parent_window = parent
        self.init_ui()
        self.load_data()

    def init_ui(self):
        layout = QVBoxLayout()

        # 创建表格
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["画布ID", "图像名称", "图像尺寸", "ROI类型", "ROI详情"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.doubleClicked.connect(self.handle_row_double_click)

        layout.addWidget(self.table)

        # 添加按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def load_data(self):
        """加载所有画布的信息到表格"""
        canvas_info = self.parent_window.get_all_canvas_info()
        self.table.setRowCount(0)

        for info in canvas_info:
            # 添加画布基本信息行
            row_position = self.table.rowCount()
            self.table.insertRow(row_position)
            self.table.setItem(row_position, 0, QTableWidgetItem(str(info['canvas_id'])))
            self.table.setItem(row_position, 1, QTableWidgetItem(info['image_name']))
            self.table.setItem(row_position, 2, QTableWidgetItem(f"{info['image_size'][1]}×{info['image_size'][0]}"))
            self.table.setItem(row_position, 3, QTableWidgetItem("这是画布"))
            self.table.setItem(row_position, 4, QTableWidgetItem(""))

            # 添加ROI信息行
            for roi in info['ROIs']:
                row_position = self.table.rowCount()
                self.table.insertRow(row_position)
                self.table.setItem(row_position, 0, QTableWidgetItem(str(info['canvas_id'])))
                self.table.setItem(row_position, 1, QTableWidgetItem(info['image_name']))
                self.table.setItem(row_position, 2, QTableWidgetItem(""))

                # ROI类型
                self.table.setItem(row_position, 3, QTableWidgetItem(roi['type']))

                # ROI详情
                if roi['type'] == 'v_rect':
                    x, y = roi['position']
                    w, h = roi['size']
                    details = f"位置: ({x}, {y}), 尺寸: {w}×{h}"
                elif roi['type'] == 'v_line':
                    x1, y1 = roi['start']
                    x2, y2 = roi['end']
                    details = f"起点: ({x1}, {y1}), 终点: ({x2}, {y2}), 宽度: {roi['width']}"
                elif roi['type'] == 'anchor':
                    x, y = roi['position']
                    details = f"位置: ({x}, {y})"
                elif roi['type'] == 'pixel_roi':
                    n = roi['counts']
                    details = f"选中{n}个像素"
                else:
                    details = "没有ROI"

                self.table.setItem(row_position, 4, QTableWidgetItem(details))

        # 调整列宽
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def handle_row_double_click(self, index):
        """双击行时选择对应的画布"""
        row = index.row()
        canvas_id_item = self.table.item(row, 0)
        self.roi_type = self.table.item(row, 3).text()
        if canvas_id_item:
            canvas_id = int(canvas_id_item.text())
            self.parent_window.set_cursor_id(canvas_id)
            self.accept()

# 伪色彩管理弹窗
class ColorMapDialog(QDialog):
    def __init__(self, parent = None, colormap_list = None, canvas_info=None,params = None):
        super().__init__(parent)
        if canvas_info is None:
            canvas_info = []
        self.params = params
        self.setWindowTitle("伪色彩管理器")
        self.setMinimumWidth(300)
        self.setMinimumHeight(300)
        self.colormap_list = colormap_list
        self.canvas_info = canvas_info
        self.canvas_index = -1
        self.parent_window = parent
        self.imagemin = params['min_value'] if params['min_value'] is not None else 0
        self.imagemax = params['max_value'] if params['min_value'] is not None else 255
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.colormap_control_layout = QFormLayout()

        # 区域选择器
        self.canvas_selector = QComboBox()
        self.canvas_selector.addItems(["所有区域"])
        if self.canvas_info:
            for item in self.canvas_info:
                self.canvas_selector.addItem(item)
        self.canvas_selector.currentIndexChanged.connect(self._handle_canvas_change)

        # 伪彩色开关
        self.colormap_toggle = QCheckBox()
        self.colormap_toggle.stateChanged.connect(self._handle_colormap_toggle)

        # 伪彩色方案选择
        self.colormap_selector = QComboBox()
        self.colormap_selector.addItems(self.colormap_list)

        # 边界设置
        self.boundary_set = QComboBox()
        self.boundary_set.addItems(['自动设置','手动设置'])
        self.boundary_set.currentIndexChanged.connect(self._handle_boundary_set)

        self.up_boundary_set = QDoubleSpinBox()
        self.up_boundary_set.setRange(0,999999)
        self.up_boundary_set.setDecimals(1)
        self.low_boundary_set = QDoubleSpinBox()
        self.low_boundary_set.setRange(0,999999)
        self.low_boundary_set.setDecimals(1)

        # 添加到布局
        self.colormap_control_layout.addRow(QLabel("应用区域:"),self.canvas_selector)
        self.colormap_control_layout.addRow(QLabel("伪彩显示:"),self.colormap_toggle)
        self.colormap_control_layout.addRow(QLabel("配色方案:"),self.colormap_selector)
        self.colormap_control_layout.addRow(QLabel("边界设置:"),self.boundary_set)
        self.colormap_control_layout.addRow(QLabel("上界设置:"),self.up_boundary_set)
        self.colormap_control_layout.addRow(QLabel('下界设置:'),self.low_boundary_set)

        self.colormap_control_layout.setSpacing(10)

        layout.addLayout(self.colormap_control_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

        self._handle_colormap_toggle()
        self._handle_boundary_set()

    def _handle_colormap_toggle(self):
        if self.colormap_toggle.isChecked():
            self.colormap_selector.setEnabled(True)
            # self.canvas_selector.setEnabled(True)
            self.boundary_set.setEnabled(True)
        else:
            self.colormap_selector.setEnabled(False)
            # self.canvas_selector.setEnabled(False)
            self.boundary_set.setEnabled(False)

    def _handle_canvas_change(self):
        if self.canvas_selector.currentIndex() >= 1:
            canvas_id = self.canvas_selector.currentIndex()-1
            canvas_params = self.parent_window.display_canvas[canvas_id].args_dict
            self.imagemin = self.parent_window.display_canvas[canvas_id].data.imagemin
            self.imagemax = self.parent_window.display_canvas[canvas_id].data.imagemax
            self.colormap_toggle.setChecked(canvas_params['use_colormap'])
            self.colormap_selector.setCurrentText(canvas_params['colormap'])
            self.up_boundary_set.setValue(self.imagemax)
            self.low_boundary_set.setValue(self.imagemin)
            self.canvas_index = self.canvas_selector.currentIndex() - 1
            if not canvas_params['auto_boundary_set']:
                self.boundary_set.setCurrentIndex(1)
            else:
                self.boundary_set.setCurrentIndex(0)
        else:
            self.canvas_index = -1
            self.colormap_toggle.setChecked(self.params['use_colormap'])
            self.colormap_selector.setCurrentText(self.params['colormap'])
            self.imagemin = self.params['min_value'] if self.params['min_value'] is not None else 0
            self.imagemax = self.params['max_value'] if self.params['min_value'] is not None else 255
            self.up_boundary_set.setValue(self.imagemax)
            self.low_boundary_set.setValue(self.imagemin)
            if not self.params['auto_boundary_set']:
                self.boundary_set.setCurrentIndex(1)
            else:
                self.boundary_set.setCurrentIndex(0)

    def _handle_boundary_set(self):
        if self.boundary_set.currentIndex() == 0:
            self.auto_boundary_set = True
            self.up_boundary_set.setEnabled(False)
            self.low_boundary_set.setEnabled(False)
        else:
            self.auto_boundary_set = False
            self.up_boundary_set.setEnabled(True)
            self.low_boundary_set.setEnabled(True)
            self.up_boundary_set.setValue(self.imagemax)
            self.low_boundary_set.setValue(self.imagemin)

    def get_value(self):
        return {'colormap':self.colormap_selector.currentText() if self.colormap_toggle.isChecked() else None,
            'use_colormap':self.colormap_toggle.isChecked(),
            'auto_boundary_set':self.auto_boundary_set,
            'min_value':self.low_boundary_set.value() if not self.auto_boundary_set else None,
            'max_value':self.up_boundary_set.value() if not self.auto_boundary_set else None,}
