import logging
import time
from cProfile import label
from symtable import Class
from typing import List

import numpy as np
from PyQt5.QtGui import QColor, QIntValidator, QFont
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QRadioButton, QSpinBox, QLineEdit, QPushButton,
                             QLabel, QMessageBox, QFormLayout, QDoubleSpinBox, QColorDialog, QComboBox, QCheckBox,
                             QFileDialog, QWhatsThis, QTextBrowser, QTableWidget, QDialogButtonBox, QTableWidgetItem,
                             QHeaderView, QAbstractItemView, QTabWidget, QWidget, QListWidget, QListWidgetItem,
                             QSizePolicy, QTreeWidget, QTreeWidgetItem)
from PyQt5.QtCore import Qt, QEvent, QTimer, QModelIndex, pyqtSignal
import HelpContentHTML
from DataManager import Data,ProcessedData
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
        self.help_dialog = CustomHelpDialog(help_title, help_content, self)
        self.help_dialog.show()  # 非阻塞显示

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
        aim_data = self.parent().data.data_origin

        # 修复数据
        fixed_data = self.parent().data_processor.fix_bad_frames(
            aim_data,
            self.bad_frames,
            n_frames
        )

        # 重新处理显示数据
        data_amend = self.parent().data_processor.amend_data(fixed_data)

        self.parent().data.update_data(**data_amend)

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
        self.datatypes = ['tif','avi','gif','png','plt']
        self.canvas_info = canvas_info
        self.export_type = export_type
        self.current_type = 'tif'

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
        self.duration_label = QLabel("视频时长：")
        self.duration_input = QSpinBox()
        self.duration_input.setRange(0,10000)
        self.duration_input.setValue(60)
        form_layout.addRow(self.duration_label, self.duration_input)
        self.duration_label.setVisible(False)
        self.duration_input.setVisible(False)
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("导出图的title，可留空")
        self.title_label = QLabel("导图标题：")
        form_layout.addRow(self.title_label , self.title_input)
        self.title_input.setVisible(False)
        self.title_label.setVisible(False)
        self.colorbar_label_label = QLabel("彩棒标签：")
        self.colorbar_label_input = QLineEdit()
        self.colorbar_label_input.setPlaceholderText("导出图的右侧的标签，可留空")
        form_layout.addRow(self.colorbar_label_label , self.colorbar_label_input)
        self.colorbar_label_label.setVisible(False)
        self.colorbar_label_input.setVisible(False)

        self.info_label = QLabel('注意：使用avi/gif/png，会对原始数据有压缩！')
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.info_label)

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
            self.datatypes = ['tif','avi','gif','png','plt']
            self.update_datatype(self.datatypes)
        if not self.is_temporal[self.canvas_selector.currentIndex()]:
            self.datatypes = ['tif', 'png', 'plt']
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
        if self.current_type == 'gif' or self.current_type == 'avi':
            self.duration_label.setVisible(True)
            self.duration_input.setVisible(True)
            self.title_input.setVisible(False)
            self.title_label.setVisible(False)
            self.colorbar_label_label.setVisible(False)
            self.colorbar_label_input.setVisible(False)
            self.info_label.setText('注意：使用avi/gif/png，会对原始数据有压缩！')
        elif self.current_type == 'plt':
            self.duration_label.setVisible(False)
            self.duration_input.setVisible(False)
            self.title_input.setVisible(True)
            self.title_label.setVisible(True)
            self.colorbar_label_label.setVisible(True)
            self.colorbar_label_input.setVisible(True)
            self.info_label.setText("本方法是用于导出带colorbar结果的tif图")
        else:
            self.duration_label.setVisible(False)
            self.duration_input.setVisible(False)
            self.title_input.setVisible(False)
            self.title_label.setVisible(False)
            self.colorbar_label_label.setVisible(False)
            self.colorbar_label_input.setVisible(False)
            self.info_label.setText('注意：使用avi/gif/png，会对原始数据有压缩！')
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
            # 'canvas': self.canvas_selector.currentIndex() if self.export_type == 'canvas' else None,
            # 'path': self.path_edit.text().strip(),
            # 'filename': self.text_edit.text().strip(),
            # 'filetype': self.type_combo.currentText(),
            'duration': self.duration_input.value() if self.current_type in ['gif','avi'] else None,
            'title': self.title_input.text() if self.current_type in ['plt'] else None,
            'colorbar_label': self.colorbar_label_input.text() if self.current_type in ['plt'] else None,
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
        self.button_box = QDialogButtonBox(QDialogButtonBox.Cancel, Qt.Horizontal, self)

        # 根据add_canvas设置按钮状态(现在不要这个按钮了)
        # if self.add_canvas:
        #     self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
        #     self.button_box.button(QDialogButtonBox.Ok).setText("确定")
        # else:
        #     self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)
        #     self.button_box.button(QDialogButtonBox.Ok).setText("选择")

        main_layout.addWidget(self.button_box)

        # 连接按钮信号
        # self.button_box.accepted.connect(self.accept)
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
                item.setToolTip(value)
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row_idx, col_idx, item)

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
        if table_index == 0 and self.datadict != []:
            data_list = self.datadict
            self.selected_table = 'data'
        else:
            data_list = self.processed_datadict
            self.selected_table = 'processed_data'

        self.selected_index = row_index
        selected_data = data_list[row_index]

        # 获取名称和时间戳
        name = selected_data.get('name')
        self.selected_name = str(name) if name else f"数据{row_index + 1}"
        self.selected_timestamp = selected_data.get('timestamp')

        # 更新状态显示
        self.selected_data_label.setText(self.selected_name)

        self.accept()

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
        """获取选择的数据信息,(timestamp,selected_type(data or processed_data))"""
        return self.selected_timestamp, self.selected_table

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
        self.canvas_id = None
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
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["画布ID", "图像名称", "图像尺寸", "ROI类型", "ROI详情",'操作'])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.doubleClicked.connect(self.handle_click)

        layout.addWidget(self.table)

        # 添加按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        # button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def load_data(self):
        """加载所有画布的信息到表格"""
        self.canvas_info = self.parent_window.get_all_canvas_info()
        self.table.setRowCount(0)

        for info in self.canvas_info:
            # 添加画布基本信息行
            row_position = self.table.rowCount()
            self.table.insertRow(row_position)
            # 创建并设置所有单元格
            items_data = [
                str(info['canvas_id']),
                info['image_name'],
                f"{info['image_size'][1]}×{info['image_size'][0]}",
                "这是画布本身",
                ""
            ]

            for col, text in enumerate(items_data):
                item = QTableWidgetItem(text)
                item.setToolTip(text)  # 始终设置ToolTip
                self.table.setItem(row_position, col, item)

            # 添加ROI信息行
            for roi in info['ROIs']:
                row_position = self.table.rowCount()
                self.table.insertRow(row_position)

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

                roi_items_data = [
                    str(info['canvas_id']),
                    info['image_name'],
                    "",
                    roi['type'],
                    details
                ]
                for col, text in enumerate(roi_items_data):
                    item = QTableWidgetItem(text)
                    item.setToolTip(text)  # 始终设置ToolTip
                    self.table.setItem(row_position, col, item)
                # self.table.setItem(row_position, 4, QTableWidgetItem(details))
                button_text = "ROI设置"
                button = QPushButton(button_text)

                # 使用lambda表达式捕获当前行索引和表格
                button.clicked.connect(lambda checked, r=row_position: self.handle_click(r))
                self.table.setCellWidget(row_position, 5, button)

        # 调整列宽
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

    def handle_click(self, index: QModelIndex|int):
        """双击行时选择对应的画布"""
        if isinstance(index, QModelIndex):
            row = index.row()
        else:
            row = index
        canvas_id_item = self.table.item(row, 0)
        roi_type_item = self.table.item(row, 3)
        self.roi_type = self.table.item(row, 3).text()
        if not canvas_id_item:
            return None
        self.canvas_id = int(canvas_id_item.text())
        # 查找对应的画布信息
        canvas_info = None
        for info in self.canvas_info:
            if info['canvas_id'] == self.canvas_id:
                canvas_info = info
                break

        # 判断是画布行还是ROI行
        if roi_type_item and roi_type_item.text() != "这是画布本身":
            # 这是ROI行
            roi_type = roi_type_item.text()

            # 查找对应的ROI信息
            roi_info = None
            for roi in canvas_info['ROIs']:
                if roi['type'] == roi_type:
                    # 根据类型和详细信息匹配ROI
                    roi_info = roi
                    break

            if roi_info:
                # 返回ROI和画布信息
                self.selected_roi_info = {
                    'type': roi_type,
                    'canvas_info': {
                        'canvas_id': canvas_info['canvas_id'],
                        'image_name': canvas_info['image_name'],
                        'image_size': canvas_info['image_size']
                    },
                    'roi_info': roi_info
                }
        # self.parent_window.set_cursor_id(canvas_id) # 忘记这个有没有用了

        self.accept()

# ROI编辑处理对话框
class ROIProcessedDialog(QDialog):
    def __init__(self, draw_type, canvas_id, roi, roi_info,data_type = None, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("ROI设置对话框")
        self.setMinimumSize(250, 400)
        self.draw_type = draw_type
        self.canvas_id = canvas_id
        self.roi_info = roi_info
        self.data_type = data_type

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("所选ROI信息："))
        self.infolist = QListWidget()
        self.infolist.setAlternatingRowColors(True)  # 交替行颜色
        self.infolist.setSelectionMode(QListWidget.NoSelection)  # 不可选择
        self.infolist.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout.addWidget(self.infolist)

        form_layout = QFormLayout()
        self.crop_check = QCheckBox()
        self.inverse_check = QCheckBox()
        self.reset_value = QDoubleSpinBox()
        self.reset_value.setValue(1)
        self.zoom_check = QCheckBox()
        self.zoom_check.setText("插值放大|倍数")
        self.zoom_check.setEnabled(False)
        self.zoom_factor = QDoubleSpinBox()
        self.zoom_factor.setValue(1)
        self.zoom_factor.setEnabled(False)
        self.fast_check = QCheckBox()
        self.fast_check.setEnabled(False)
        self.crop_check.toggled.connect(lambda checked: self.zoom_check.setEnabled(checked))
        self.zoom_check.toggled.connect(lambda checked: self.zoom_factor.setEnabled(checked))
        form_layout.addRow(QLabel("截取数据:"),self.crop_check)
        form_layout.addRow(QLabel("注释："), QLabel('选择后选区数据\n会被裁剪提取出来'))
        form_layout.addRow(QLabel("选区反转:"),self.inverse_check)
        form_layout.addRow(QLabel("注释："), QLabel('默认操作都是针对选区的，\n该选项会反转选区'))
        form_layout.addRow(QLabel("选区赋值:"),self.reset_value)
        form_layout.addRow(QLabel("注释："), QLabel('为原值的倍数，\n填0相当于去掉原值'))
        form_layout.addRow(QLabel("选区放大："),QLabel("是否进行插值放大以及放大的倍数"))
        form_layout.addRow(self.zoom_check, self.zoom_factor)
        form_layout.addRow(QLabel("便捷操作:"),self.fast_check)
        layout.addLayout(form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        self.update_list()
        self.setLayout(layout)


    def update_list(self):
        fast_text = ''
        if self.draw_type == 'v_rect':
            type_text = "ROI类型：矢量矩形选区"
            # x, y = self.roi_info['ROIs']['position']
            w, h = self.roi_info['roi_info']['size']
            text1 = f"该ROI尺寸为：{w}×{h} pixels"
            text2 = f"该ROI所属数据：{self.roi_info['canvas_info']['image_name']}"
            crop_text = f"是否支持截取数据：是（注意是否有便捷操作）"
            inverse_text = f"是否支持选区反转：否"
            reset_text = f"是否支持选区赋值：是"
            zoom_text = f"是否支持选区插值放大：是（需截取数据）"
            self.inverse_check.setEnabled(False)
            if self.data_type == 'Accumulated_time_amplitude_map':
                fast_text = f'支持的便捷操作: 单通道计算快速实现（母函数截取数据）'
                self.fast_check.setEnabled(True)
                self.fast_check.setChecked(True)
        elif self.draw_type == 'v_line':
            type_text = "ROI类型：矢量直线选区"
            x1, y1 = self.roi_info['roi_info']['start']
            x2, y2 = self.roi_info['roi_info']['end']
            text1 = f"该ROI起点: ({x1}, {y1}), 终点: ({x2}, {y2}), 宽度: {self.roi_info['roi_info']['width']}"
            text2 = f"该ROI所属数据：{self.roi_info['canvas_info']['image_name']}"
            crop_text = f"是否支持截取数据：否"
            inverse_text = f"是否支持选区反转：否"
            reset_text = f"是否支持选区赋值：否"
            zoom_text = f"是否支持选区插值放大：否"
            fast_text = f'矢量直线目前不支持高级设置，请直接点ok'
            self.reset_value.setEnabled(False)
            self.crop_check.setEnabled(False)
            self.inverse_check.setEnabled(False)
            self.zoom_check.setEnabled(False)
            self.zoom_factor.setEnabled(False)
        else : #self.draw_type == 'pixel_roi'
            type_text = "ROI类型：像素绘制选区"
            # x, y = self.roi_info['ROIs']['position']
            n = self.roi_info['roi_info']['counts']
            text1 = f"该ROI共覆盖：{n}个 pixels"
            text2 = f"该ROI所属数据：{self.roi_info['canvas_info']['image_name']}"
            crop_text = f"是否支持截取数据：是"
            inverse_text = f"是否支持选区反转：是"
            reset_text = f"是否支持选区赋值：是"
            zoom_text = f"是否支持选区插值放大：是"
        self.infolist.addItem(QListWidgetItem(type_text))
        self.infolist.addItem(QListWidgetItem(text1))
        self.infolist.addItem(QListWidgetItem(text2))
        self.infolist.addItem(QListWidgetItem(crop_text))
        self.infolist.addItem(QListWidgetItem(inverse_text))
        self.infolist.addItem(QListWidgetItem(reset_text))
        self.infolist.addItem(QListWidgetItem(zoom_text))
        if fast_text:
            self.infolist.addItem(QListWidgetItem(fast_text))
        self.infolist.adjustSize()

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
        self.up_boundary_set.setRange(-999999,999999)
        self.up_boundary_set.setDecimals(3)
        self.low_boundary_set = QDoubleSpinBox()
        self.low_boundary_set.setRange(-999999,999999)
        self.low_boundary_set.setDecimals(3)

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
            canvas = self.parent_window.display_canvas[canvas_id]
            self.imagemin = self.parent_window.display_canvas[canvas_id].min_value
            self.imagemax = self.parent_window.display_canvas[canvas_id].max_value
            self.colormap_toggle.setChecked(canvas.use_colormap)
            self.colormap_selector.setCurrentText(canvas.colormap)
            self.up_boundary_set.setValue(self.imagemax)
            self.low_boundary_set.setValue(self.imagemin)
            self.canvas_index = self.canvas_selector.currentIndex() - 1
            if not canvas.auto_boundary_set:
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

# 选择数据绘制plot
class DataPlotSelectDialog(QDialog):
    sig_plot_request = pyqtSignal(np.ndarray, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("数据流管理与导出")
        self.resize(900, 500)
        self.setModal(False)  # 设为非模态，方便一边看数据一边操作主界面

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 顶部说明
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("数据层级结构：原始数据 (Data) -> 处理数据 (ProcessedData) -> ..."))

        refresh_btn = QPushButton("刷新列表")
        refresh_btn.clicked.connect(self.refresh_data)
        header_layout.addWidget(refresh_btn)

        layout.addLayout(header_layout)

        # 核心控件：QTreeWidget
        self.tree = QTreeWidget()
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels(["名称 / Key", "类型", "尺寸 & 大小", "数值范围", "创建时 / 值", "操作"])

        # 调整列宽
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(4, QHeaderView.ResizeToContents)

        self.tree.setAlternatingRowColors(True)   # 开启交替行颜色（可选，看起来更像表格）
        self.tree.setAnimated(True)              # 开启展开收起的动画
        self.tree.setIndentation(20)            # 设置缩进宽度

        layout.addWidget(self.tree)

    def refresh_data(self):
        self.tree.clear()

        # 1. 获取所有数据
        data_history = Data.get_history_list()
        processed_data_history = ProcessedData.get_history_list()

        # 2. 建立节点映射表 { timestamp_float: QTreeWidgetItem }
        # 用于通过 timestamp 快速（或遍历）找到父节点的 TreeItem
        self.node_map = {}

        # --- 第一步：加载所有原始 Data (作为根节点) ---
        # 倒序显示，让最新的在最上面（符合直觉），但在构建Map时要注意顺序
        # 为了逻辑顺畅，我们先建立好所有的Data节点
        for data_obj in reversed(data_history):
            root_item = QTreeWidgetItem(self.tree)
            self._setup_data_item(root_item, data_obj)

            # 添加 Parameters
            if data_obj.parameters:
                param_node = QTreeWidgetItem(root_item)
                param_node.setText(0, "⚙️ Parameters")
                self._fill_dict_items(param_node, data_obj.parameters)

            # 记录到 Map 中，供后续 ProcessedData 查找父节点
            self.node_map[data_obj.timestamp] = root_item

        # --- 第二步：加载 ProcessedData (支持多层嵌套) ---

        # 关键点：必须按【创建时间正序】排序。
        # 这样保证在处理 "子ProcessedData" 时，它的 "父ProcessedData" 已经被创建并加入到 self.node_map 中了。
        # 假设 ProcessedData 也有 .timestamp 属性代表其创建时间
        sorted_processed = sorted(processed_data_history, key=lambda x: getattr(x, 'timestamp', 0))

        orphan_processed = []  # 记录找不到爹的孤儿数据

        for proc_obj in sorted_processed:
            # 1. 寻找父节点
            parent_ts = proc_obj.timestamp_inherited
            parent_item = None

            # 由于浮点数精度问题，不能直接 dict.get(float)，需要模糊匹配
            # 优化：如果数据量极大，建议将 timestamp 格式化字符串作为 key
            # 这里采用遍历匹配 (对于UI显示的数据量级通常没问题)
            for ts, item in self.node_map.items():
                if abs(ts - parent_ts) < 1e-6:
                    parent_item = item
                    break

            if parent_item:
                # 2. 找到了父节点（可能是 Data，也可能是之前添加的 ProcessedData）
                proc_item = QTreeWidgetItem(parent_item)
                self._setup_processed_item(proc_item, proc_obj)

                # 添加 Out Processed Results
                if proc_obj.out_processed:
                    out_node = QTreeWidgetItem(proc_item)
                    out_node.setText(0, "⚙️ Other Results")
                    self._fill_dict_items(out_node, proc_obj.out_processed)

                # 3. 重要：将当前 ProcessedData 也加入 Map
                # 这样后续的数据如果是基于它的，就可以把它当做父节点
                if hasattr(proc_obj, 'timestamp'):
                    self.node_map[proc_obj.timestamp] = proc_item
            else:
                # 没找到父节点，暂时放入孤儿列表
                orphan_processed.append(proc_obj)

        # --- 第三步：处理真正的孤儿数据 (原始数据已被删除或丢失) ---
        if orphan_processed:
            orphan_root = QTreeWidgetItem(self.tree)
            orphan_root.setText(0, "历史处理记录 (无关联源数据)")
            # 设置颜色提示
            # orphan_root.setForeground(0, QBrush(Qt.GlobalColor.gray))
            orphan_root.setExpanded(True)

            for proc_obj in orphan_processed:
                # 注意：这里孤儿内部如果也有嵌套关系，上面的逻辑因为找不到第一级父节点，
                # 后续子节点也会掉入 orphan_processed。
                # 在孤儿区简单平铺显示，或者也可以再做一次递归，视需求而定。
                # 这里做简单平铺处理：
                proc_item = QTreeWidgetItem(orphan_root)
                self._setup_processed_item(proc_item, proc_obj)

                if proc_obj.out_processed:
                    out_node = QTreeWidgetItem(proc_item)
                    out_node.setText(0, "⚙️ Out Processed Results")
                    self._fill_dict_items(out_node, proc_obj.out_processed)

        self.tree.expandToDepth(1)

    def _setup_data_item(self, item: QTreeWidgetItem, data_obj: Data):
        """配置 Data 类型的行显示"""
        item.setText(0, f"📦 {data_obj.name}")
        item.setText(1, f"原始 ({data_obj.format_import})")
        item.setText(2, self._shape_to_str(data_obj.datashape)+'\n'+self._format_array_size(data_obj.data_origin))
        item.setText(3, f"{data_obj.datamin:.2f} ~ {data_obj.datamax:.2f}")
        # 将时间戳格式化
        time_str = time.strftime('%y/%m/%d %H:%M:%S', time.localtime(data_obj.timestamp))
        item.setText(4, time_str)

        # 检查是否线性数据并添加按钮
        self._check_and_add_plot_button(item, data_obj.data_origin, data_obj.name, data_obj)

    def _setup_processed_item(self, item: QTreeWidgetItem, proc_obj: ProcessedData):
        """配置 ProcessedData 类型的行显示"""
        item.setText(0, f"🔎 {re.sub(r'[^@]+@', '...@', proc_obj.name)}") # 类似输出: ...@...@r_stft
        item.setText(1, f"🏷️ {proc_obj.type_processed}")
        if proc_obj.data_processed is not None:
            item.setText(2, self._shape_to_str(proc_obj.datashape)+'\n'+self._format_array_size(proc_obj.data_processed))
            item.setText(3, f"{proc_obj.datamin:.2f} ~ {proc_obj.datamax:.2f}")
        else:
            item.setText(2, "None")
        time_str = time.strftime('%y/%m/%d %H:%M:%S', time.localtime(proc_obj.timestamp))
        item.setText(4, time_str)

        # 检查是否线性数据并添加按钮
        if proc_obj.data_processed is not None:
            self._check_and_add_plot_button(item, proc_obj.data_processed, proc_obj.name, proc_obj)

    def _fill_dict_items(self, parent_item: QTreeWidgetItem, data_dict: dict):
        """递归填充字典数据"""
        for k, v in data_dict.items():
            child = QTreeWidgetItem(parent_item)
            child.setText(0, str(k))

            # 如果值是 numpy 数组，显示其摘要
            if isinstance(v, np.ndarray):
                child.setText(1, "ndarray")
                child.setText(2, self._shape_to_str(v.shape))
                child.setText(3, f'{v.min():.2f} ~ {v.max():.2f}')
                child.setText(4, "Array Data")
                # 如果是一维数组，也允许导出
                self._check_and_add_plot_button(child, v, str(k), None)
            elif isinstance(v, dict):
                child.setText(1, "dict")
                self._fill_dict_items(child, v)  # 递归
            elif isinstance(v, list):
                child.setText(1, "list")
                child.setText(2, self._shape_to_str(len(v)))
                child.setText(3, f'{min(v):.2f} ~ {max(v):.2f}')
                child.setText(4, "List Data")
            elif isinstance(v, float):
                child.setText(1, "float")
                child.setText(4,f'{v:.4f}')
            else:
                child.setText(1, type(v).__name__)
                child.setText(4, str(v))

    def _check_and_add_plot_button(self, item: QTreeWidgetItem, data_array: np.ndarray, name: str, original_obj):
        """
        判断数据是否为线性（1D），如果是，在最后一列添加按钮
        """
        if not isinstance(data_array, np.ndarray):
            return

        is_linear = False
        # 判断逻辑：一维数组，或者二维数组中有一维是1 (例如 (1000, 1))
        if data_array.ndim == 1:
            is_linear = True
        elif data_array.ndim == 2:
            if data_array.shape[0] == 1 or data_array.shape[1] == 1:
                is_linear = True

        if is_linear:
            btn = QPushButton("导出绘图")
            # 使用 lambda 捕获数据
            # 注意：lambda 中的变量绑定问题，需要默认参数
            btn.clicked.connect(lambda _, d=data_array, n=name, o=original_obj: self.emit_plot_signal(d, n, o))
            btn.setStyleSheet("padding: 0px;")

            # 因为 QTreeWidget 是 ItemView，需要用 setItemWidget 将 Widget 放入单元格
            self.tree.setItemWidget(item, 5, btn)

    def emit_plot_signal(self, data, name, obj):
        """发射信号"""
        print(f"Requesting plot for: {name}, Shape: {data.shape}")
        # 如果是 (N, 1) 转为 (N,)
        if data.ndim == 2:
            if data.shape[1] == 1:
                data = data.flatten()
            elif data.shape[0] == 1:
                data = data.flatten()

        self.sig_plot_request.emit(data, name)

    # def _format_size(self, data_obj):
    #     """格式化数据大小显示"""
    #     if hasattr(data_obj, 'data_processed') and data_obj.data_processed is not None:
    #         return self.format_array_size(data_obj.data_processed)
    #     elif hasattr(data_obj, 'data_origin') and data_obj.data_origin is not None:
    #         return self.format_array_size(data_obj.data_origin)
    #     return "N/A"

    @staticmethod
    def _format_array_size(array):
        """格式化numpy数组大小"""
        if array is None:
            return " 0 bytes"
        size_bytes = array.nbytes
        for unit in ['bytes', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f" {size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f" {size_bytes:.1f} TB"

    @staticmethod
    def _shape_to_str(shape):
        """将形状转换为 t×h×w 格式的字符串 """
        if hasattr(shape, 'shape'):
            # 如果传入的是numpy数组对象
            shape = shape.shape

        # 确保shape是可迭代的
        if not hasattr(shape, '__iter__'):
            shape = (shape,)

        # 将每个维度转换为字符串并用乘号连接
        return '×'.join(str(dim) for dim in shape)
