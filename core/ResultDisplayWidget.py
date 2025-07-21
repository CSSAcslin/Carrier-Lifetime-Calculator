import logging

import numpy as np
import matplotlib as plt
import matplotlib.font_manager as fm
import pandas as pd
from PyQt5.QtCore import pyqtSignal
from fontTools.misc.cython import returns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QScrollArea,
                             QFileDialog, QSlider, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QTabWidget
                             )
from numpy.ma.core import shape
from pandas.core.interchange.dataframe_protocol import DataFrame


class ResultDisplayWidget(QTabWidget):
    """结果显示部件"""
    tab_type_changed = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(self.close_tab)
        self.setMovable(True)
        self.currentChanged.connect(self._current_index)

        self.font_list = fm.findSystemFonts(fontpaths=r"C:\Windows\Fonts" , fontext='ttf')
        self.chinese_fonts = [f for f in self.font_list if
                         any(c in f.lower() for c in ['simhei', 'simsun', 'microsoft yahei', 'fang'])]
        self.plot_settings = {
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
        self._init_counters()
        # 存储每个选项卡的数据
        self.tab_data = {}
        self.current_data = None
        self.current_dataframe = None

        # 设置Matplotlib默认字体
        if self.chinese_fonts:
            plt.rcParams['font.sans-serif'] = ['microsoft yahei']  # Windows常用
            # 或者使用其他字体如: 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi'
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题



    def _init_counters(self):
        """计数器初始化"""
        self.tab_counters = {
            'heatmap': 1,  # 热图
            'curve': 1,  # 寿命曲线
            'roi': 1,  # ROI曲线
            'diff': 1,  # 扩散系数
            'var': 1,  # 方差演化
            'quality': 1, # 信号评估
            'series': 1 # EM信号
        }

    def _current_index(self, index):
        """存储当前选中的选项卡索引"""
        self.current_index = index
        if index >= 0:
            tab = self.widget(index)
            tab_id = id(tab)

            if tab_id in self.tab_data:
                # # 更新当前原始数据（无用，待删）
                # self.current_data = self.tab_data.get(tab_id).get('raw_data')
                # 更新当前DataFrame
                self.current_dataframe = self.tab_data.get(tab_id).get('dataframe')

            # 发出标签页类型信号
            if tab_id in self.tab_data:
                tab_type = self.tab_data[tab_id]['type']
                self.tab_type_changed.emit(tab_type)
                return
            elif self.current_mode:
                self.tab_type_changed.emit(self.current_mode)
                return


    def store_tab_data(self, tab, tab_type, **kwargs):
        """存储选项卡数据"""
        self.tab_data[id(tab)] = {
            'type': tab_type,
            'raw_data': kwargs,
            'dataframe': self.current_dataframe.copy() if self.current_dataframe is not None else None
        }


    def close_tab(self, index):
        """关闭指定选项卡"""
        tab = self.widget(index)
        tab_id = id(tab)

        # 删除关联数据
        if tab_id in self.tab_data:
            del self.tab_data[tab_id]

        tab.deleteLater()
        self.removeTab(index)

        # 更新当前索引
        if self.count() > 0:
            self.current_index = self.currentIndex()

    def create_tab(self, tab_type, title_prefix, reuse_current=False):
        """创建新的结果选项卡"""
        if reuse_current and self.count() > 0:
            # 重用当前标签
            index = self.current_index
            title = self.tabText(index)

            # 获取现有部件
            tab = self.widget(index)
            canvas = tab.findChild(FigureCanvas)
            figure = canvas.figure
            figure.clear()

            # 返回现有资源
            return figure, canvas, index, title, tab
        else:
            # 创建新标签页
            # 标签名运算
            count = self.tab_counters[tab_type]
            self.tab_counters[tab_type] += 1
            title = f"{title_prefix}{count}"

            # 创建新的绘图画布
            figure = Figure()
            canvas = FigureCanvas(figure)

            # 创建新的标签页
            tab = QWidget()
            layout = QVBoxLayout(tab)
            layout.addWidget(canvas)

            # 添加到选项卡
            index = self.addTab(tab, title)
            self.setCurrentIndex(index)

            return figure, canvas, index, title, tab

    def update_plot_settings(self, new_settings,update = True):
        """更新绘图设置"""
        self.plot_settings.update(new_settings)
        if update:
            self.update_plot()

    def update_plot(self):
        """根据当前设置重新绘图"""
        if self.count() == 0:
            # 没有选项卡时就不需要更新
            return

        tab = self.widget(self.current_index)
        tab_id = id(tab)
        tab_info = self.tab_data.get(tab_id)

        if not tab_info:
            return

        tab_type = tab_info['type']
        raw_data = tab_info.get('raw_data', {})

        # 根据标签页类型调用对应的绘图方法
        if tab_type == 'heatmap':
            self.display_distribution_map(
                raw_data['lifetime_map'],
                reuse_current=True
            )
        elif tab_type == 'curve':
            self.display_lifetime_curve(
                raw_data['phy_signal'],
                raw_data['lifetime'],
                raw_data['r_squared'],
                raw_data['fit_curve'],
                raw_data['time_points'],
                raw_data['boundary'],
                raw_data['model_type'],
                reuse_current=True
            )
        elif tab_type == 'roi':
            self.display_roi_series(
                raw_data['positions'],
                raw_data['intensities'],
                raw_data.get('fig_title', ""),
                reuse_current=True
            )
        elif tab_type == 'diff':
            self.display_diffusion_coefficient(
                raw_data['frame_data_dict'],
                reuse_current=True
            )
        elif tab_type == 'var':
            self.plot_variance_evolution(
                reuse_current=True)

        elif tab_type == 'series':
            self.plot_time_series(
                raw_data['time'],
                raw_data['series'],
                reuse_current=True)


    def display_distribution_map(self, lifetime_map, reuse_current=False):
        """显示寿命热图"""
        self.current_mode = "heatmap"
        figure, canvas, index, title, tab = self.create_tab(self.current_mode, '热', reuse_current)

        cmap = self.plot_settings['heatmap_cmap']
        levels = self.plot_settings['contour_levels']

        ax = figure.add_subplot(111)
        # 显示热图
        im = ax.imshow(lifetime_map, cmap=cmap)
        figure.colorbar(im, ax=ax, label='lifetime')
        ax.set_title("载流子寿命分布图")
        ax.axis('off')
        figure.tight_layout()
        canvas.draw()

        # 保存当前数据
        self.current_dataframe = pd.DataFrame(lifetime_map)
        self.store_tab_data(tab, self.current_mode, lifetime_map=lifetime_map)

    def display_lifetime_curve(self, phy_signal, lifetime, r_squared, fit_curve,time_points,boundary, model_type, reuse_current=False):
        """显示区域分析结果"""
        # 使用原来的结果显示区域
        self.current_mode = "curve"
        figure, canvas, index, title, tab = self.create_tab(self.current_mode, '寿', reuse_current)

        line_style = self.plot_settings['line_style']
        line_width = self.plot_settings['line_width']
        marker_style = self.plot_settings['marker_style']
        marker_size = self.plot_settings['marker_size']
        color = self.plot_settings['color']
        show_grid = self.plot_settings['show_grid']
        set_axis = self.plot_settings['set_axis']

        ax = figure.add_subplot(111)
        if set_axis:
            max_bound = boundary['max']
            min_bound = 0
            ax.set_ylim(min_bound, max_bound)

        # 绘制原始曲线
        # time_points = time_points - time_points[0]  # 从0开始
        ax.plot(time_points, phy_signal,
                markeredgecolor=color, # 点边缘色
                markeredgewidth=line_width,
                label='原始数据',
                 marker=marker_style,       # 方形点
                 markersize=marker_size,    # 点大小
                 linestyle='')

        if not self.plot_settings['_from_start_cal']:
            # 这是从最大值算的绘制拟合曲线
            max_idx = np.argmax(phy_signal)
            fit_time = time_points[max_idx:]
            ax.plot(fit_time, fit_curve, 'r', linestyle=line_style, label='拟合曲线')
            # 标记最大值
            ax.axvline(time_points[max_idx], color='g', linestyle=':', label='峰值位置')
        elif self.plot_settings['_from_start_cal'] and np.shape(time_points)==np.shape(fit_curve):
            # 这是从头算的拟合曲线绘制
            fit_time = time_points
            ax.plot(fit_time, fit_curve, 'r', linestyle=line_style, label='拟合曲线')

        # 标记r^2和τ
        if model_type == 'single':
            ax.text(0.05, 0.95, f' τ={lifetime:.2f}\n'
                                +r'$R^2$='
                                +f'{r_squared:.3f}',
                     transform=ax.transAxes,
                     ha='left', va='top',
                     bbox=dict(facecolor='white', alpha=0.8))
        elif model_type == 'double':
            ax.text(0.05, 0.95, f' τ1={lifetime[0]:.2f}\n'
                    + f' τ2={lifetime[1]:.2f}\n'
                    + r'$R^2$='
                    + f'{r_squared:.3f}',
                    transform=ax.transAxes,
                    ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.8))

        ax.set_xlabel('时间/ps')
        ax.set_ylabel('信号强度')
        ax.set_title('载流子寿命曲线')
        ax.legend()
        ax.grid(show_grid)

        canvas.draw()
        self.current_dataframe = pd.DataFrame({
                                'time': pd.Series(time_points),
                                'signal': pd.Series(phy_signal),
                                'fit_time': pd.Series(fit_time),
                                'fit_curve':pd.Series(fit_curve)
                            })
        self.store_tab_data(tab, self.current_mode,
                           phy_signal=phy_signal,
                           lifetime=lifetime,
                           r_squared=r_squared,
                           fit_curve=fit_curve,
                           time_points=time_points,
                           boundary=boundary,
                           model_type=model_type)

    def display_roi_series(self, positions, intensities, fig_title="", reuse_current=False):
        """绘制向量ROI信号强度曲线"""
        self.current_mode = "roi"
        figure, canvas, index, title, tab = self.create_tab(self.current_mode, 'ROI', reuse_current)
        ax = figure.add_subplot(111)
        # line_style = self.plot_settings['line_style']
        # line_width = self.plot_settings['line_width']
        marker_style = self.plot_settings['marker_style']
        marker_size = self.plot_settings['marker_size']
        color = self.plot_settings['color']

        # 绘制曲线
        # ax.plot(positions, intensities,
        #         markeredgecolor=color,
        #         marker=marker_style,       # 方形点
        #         markersize=marker_size,    # 点大小
        #         linewidth=line_width,
        #         label='信号强度')
        ax.scatter(positions, intensities,
                c=color,
                marker=marker_style,       # 方形点
                label='采样点')

        # 设置图表属性
        ax.set_title(fig_title)
        ax.set_xlabel("位置 (像素)")
        ax.set_ylabel("对比度")
        ax.legend()

        canvas.draw()

        self.current_dataframe = pd.DataFrame({
                                        'time': pd.Series(positions),
                                        'signal': pd.Series(intensities),
                                    })
        self.store_tab_data(tab, self.current_mode, fig_title=fig_title , positions=positions, intensities=intensities)

    def display_diffusion_coefficient(self, frame_data_dict, reuse_current=False):
        """绘制多帧信号及高斯拟合"""
        # if frame_data_dict is None:
        #     logging.warning('缺数据或数据有问题无法绘图')
        #     return
        self.current_mode = "diff"
        figure, canvas, index, title, tab = self.create_tab(self.current_mode, '扩', reuse_current)
        ax = figure.add_subplot(111)
        self.dif_result = frame_data_dict
        marker_style = self.plot_settings['marker_style']
        marker_size = self.plot_settings['marker_size']
        color = self.plot_settings['color']

        for i, data in enumerate(self.dif_result['signal']):
            positions = data[0]
            intensities = data[1]

            # 绘制原始数据
            ax.scatter(positions, intensities,s=1)

        for i, series in enumerate(self.dif_result['fitting']):
            positions = series[0]
            fitting_curve = series[1]
            ax.plot(positions, fitting_curve, '--',
                    label=f'{self.dif_result["time_series"][i]:.0f}ps')

        # 设置图表属性
        ax.set_title("多帧ROI信号强度及高斯拟合")
        ax.set_xlabel("位置 (μm)")
        ax.set_ylabel("对比度")
        ax.grid(True)
        ax.legend()
        canvas.draw()

        # 以下是整合数据
        try:
            layer1,layer2 = [],[]
            times = self.dif_result['time_series']
            layer1.extend(['时间点：'])
            layer2.extend(['位置(μm)'])
            for i in range(times.shape[0]):
                # times0 = np.full(len(times),'时间点：')
                # times2 = np.full(len(times),'μs')
                layer1.extend([f'{times[i]:.2f}','μs'])
                layer2.extend(['原始数值','拟合曲线'])
            max_len = max(data.shape[1] for data in self.dif_result['signal'])
            outcome = []
            position = np.pad(self.dif_result['signal'][0,0], (0, max_len - len(self.dif_result['signal'][0,0])),
                              mode='constant', constant_values=np.nan)
            outcome.extend([position])
            for i,data in enumerate(self.dif_result['signal']):
                signal = np.pad(data[1], (0, max_len - len(data[1])),
                                  mode='constant', constant_values=np.nan)
                fitting = np.pad(self.dif_result['fitting'][i, 1], (0, max_len - len(self.dif_result['fitting'][i, 1])),
                                  mode='constant', constant_values=np.nan)
                outcome.extend([signal,fitting])
            columns = pd.MultiIndex.from_arrays([layer1,layer2])
            self.current_dataframe = pd.DataFrame(np.array(outcome).T, columns = columns)
            self.store_tab_data(tab, self.current_mode, frame_data_dict=frame_data_dict)
        except Exception as e:
            logging.error(f'数据打包出现问题：{e}')

    def plot_variance_evolution(self, reuse_current=False):
        """绘制方差随时间变化图并计算扩散系数"""
        if not hasattr(self,"dif_result"):
            logging.warning("请按照顺序点击按钮")
            return
        self.current_mode = "var"
        figure, canvas, index, title, tab = self.create_tab(self.current_mode, '方', reuse_current)

        ax = figure.add_subplot(111)
        show_grid = self.plot_settings['show_grid']

        times = self.dif_result["sigma"][0]
        variances = self.dif_result["sigma"][1]
        sigma_trim = self.dif_result["sigma"][:, self.dif_result["sigma"][1, :] != 0]
        # 绘制数据点
        ax.scatter(times, variances, c='r', s=50, label='方差数据')

        # 线性拟合
        slope, intercept = np.polyfit(times, variances, 1)
        fit_line = slope * times + intercept
        ax.plot(times, fit_line, 'b--',
                label=f'线性拟合 (D={slope / 2:.2e})')

        # 设置图表属性
        ax.set_title("高斯方差随时间演化")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("方差 (μm²)")
        ax.grid(show_grid)
        ax.legend()
        canvas.draw()

        self.current_dataframe = pd.DataFrame({
                                        'time': self.dif_result['time_series'],
                                        'sigma': sigma_trim[1],
        })
        self.store_tab_data(tab, self.current_mode, dif_result=self.dif_result)

    def quality_avg(self, f, t, Zxx, target_freq):
        """绘制平均信号STFT结果（信号质量评估）"""
        # 提取目标频率附近的区域
        freq_range = [target_freq - 1, target_freq + 1]

        self.current_mode = "quality"
        figure, canvas, index, title, tab = self.create_tab(self.current_mode, 'EM')

        ax = figure.add_subplot(111)

        spec = ax.pcolormesh(t, f, 10 * np.log10(np.abs(Zxx)),
                             shading='gouraud', cmap='viridis')
        ax.set_ylabel('频率 [Hz]')
        ax.set_xlabel('时间 [秒]')
        figure.colorbar(spec, label='强度 [dB]')
        # 标记目标频率
        ax.axhline(y=target_freq, color='r', linestyle='--', alpha=0.7)
        ax.set_title(f'信号质量评估 (目标频率: {target_freq} Hz)')
        canvas.draw()

        # self.current_dataframe = pd.DataFrame({"Zxx":Zxx}) 目前有问题
        self.current_dataframe = None
        self.store_tab_data(tab, self.current_mode, freq = f,
            time = t,
            Zxx = Zxx,
            target_freq = target_freq)

    def plot_time_series(self,time, series,reuse_current=False):
        """信号处理结果"""
        self.current_mode = "series"
        time = time[1:-1]
        series = series[1:-1]
        figure, canvas, index, title, tab = self.create_tab(self.current_mode, 'signal',reuse_current)
        show_grid = self.plot_settings['show_grid']
        line_style = self.plot_settings['line_style']
        line_width = self.plot_settings['line_width']

        ax = figure.add_subplot(111)
        ax.plot(time, series, 'b-', linewidth=line_width)
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(r"$\Delta$S")
        ax.grid(show_grid)
        canvas.draw()

        self.current_dataframe = pd.DataFrame({
            'time': time,
            'series': series,
        })
        self.store_tab_data(tab, self.current_mode, time = time, series = series)
