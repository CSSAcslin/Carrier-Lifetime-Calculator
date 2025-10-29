import logging
import os
import time
import copy
import numpy as np
import tifffile as tiff
import sif_parser
import cv2
from collections import deque
from typing import ClassVar, Optional, List, Dict, Any, Union
from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot
from dataclasses import dataclass, field, fields
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image



class DataManager(QObject):
    save_request_back = pyqtSignal(dict)
    read_request_back = pyqtSignal(dict)
    remove_request_back = pyqtSignal(dict)
    amend_request_back = pyqtSignal(dict)
    data_progress_signal = pyqtSignal(int, int)
    def __init__(self, parent=None):
        super(DataManager, self).__init__(parent)
        self.color_map_manager = ColorMapManager()
        logging.info("图像数据管理线程已启动")

    @staticmethod
    def to_uint8(data):
        """归一化和数字类型调整"""
        data_o = data.image_backup
        min_value = data.imagemin
        max_value = data.imagemax
        if data.datatype == np.uint8 and max_value == 255:
            data.image_data = data_o.copy()
            return True

        # 计算数组的最小值和最大值
        data.image_data = ((data_o - min_value)/(max_value- min_value)*255).astype(np.uint8)
        return True

    def to_colormap(self,data,params):
        """伪色彩实现（其实仅在生成视图时才会更新）"""
        logging.info("样式应用中，预览会同步更新")
        self.color_map_manager = ColorMapManager()
        colormode = params['colormap']
        if params['auto_boundary_set']:
            min_value = data.imagemin
            max_value = data.imagemax
        else:
            min_value = params['min_value']
            max_value = params['max_value']
        if colormode is None:
            self.to_uint8(data)
            data.colormode = colormode
            return False
        if data.is_temporary:
            T,H,W = data.imageshape
            self.data_progress_signal.emit(0,T)
            new_data = np.zeros((T,H,W, 4), dtype=np.uint8)
            for i,image in enumerate(data.image_backup):
                new_data[i] = self.color_map_manager.apply_colormap(
                                                                    image,
                                                                    colormode,
                                                                    min_value,
                                                                    max_value
                                                                )
                self.data_progress_signal.emit(i, T)
            data.image_data = new_data
            self.data_progress_signal.emit(T, T)
        else:
            H, W = data.imageshape
            new_data = np.zeros((H, W, 4), dtype=np.uint8)
            new_data = self.color_map_manager.apply_colormap(
                data.image_backup,
                colormode,
                min_value,
                max_value
            )
            data.image_data = new_data
        data.colormode = colormode
        return True

    @pyqtSlot(np.ndarray, str, str, str, bool)
    def export_data(self, result, output_dir, prefix, format_type='tif',is_temporal=True,duration = 100):
        """
        时频变换后目标频率下的结果导出
        支持多种格式: tif, avi, png, gif

        参数:
            result: 输入数据数组
            output_dir: 输出目录路径
            prefix: 文件前缀
            format_type: 导出格式 ('tif', 'avi', 'png', 'gif')
        """
        format_type = format_type.lower()

        # 根据格式类型调用不同的导出函数
        if format_type == 'tif':
            return self.export_as_tif(result, output_dir, prefix,is_temporal)
        elif format_type == 'avi':
            return self.export_as_avi(result, output_dir, prefix)
        elif format_type == 'png':
            return self.export_as_png(result, output_dir, prefix,is_temporal)
        elif format_type == 'gif':
            return self.export_as_gif(result, output_dir, prefix, duration)
        else:
            logging.error(f"不支持的格式类型: {format_type}")
            raise ValueError(f"不支持格式: {format_type}。请使用 'tif', 'avi', 'png' 或 'gif'")

    def _normalize_data(self, data):
        """统一归一化处理，支持彩色/灰度数据"""
        if data.dtype == np.uint8:
            return data.copy()

        # 计算全局最小最大值（避免逐帧计算不一致）
        data_min = data.min()
        data_max = data.max()

        # 处理全零数据
        if data_max - data_min < 1e-6:
            return np.zeros_like(data, dtype=np.uint8)

        normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        return normalized

    def export_as_tif(self, result, output_dir, prefix,is_temporal=True):
        """支持彩色TIFF导出"""
        created_files = []
        if is_temporal:
            num_frames = result.shape[0]
            num_digits = len(str(num_frames))
            self.data_progress_signal.emit(0, num_frames)

            for frame_idx in range(num_frames):
                frame_name = f"{prefix}-{frame_idx:0{num_digits}d}.tif"
                output_path = os.path.join(output_dir, frame_name)

                frame = result[frame_idx]
                photometric = 'minisblack' if frame.ndim == 2 else 'rgb'
                tiff.imwrite(output_path, frame, photometric=photometric)

                created_files.append(output_path)
                self.data_progress_signal.emit(frame_idx + 1, num_frames)
        else:
            num_frames = 1
            frame_name = f"{prefix}.tif"
            output_path = os.path.join(output_dir, frame_name)
            photometric = 'minisblack' if result.ndim == 2 else 'rgb'
            tiff.imwrite(output_path, result, photometric=photometric)
            created_files.append(output_path)
            self.data_progress_signal.emit(num_frames+1, num_frames)

        logging.info(f'导出TIFF完成: {output_dir}, 共{num_frames}帧')
        return created_files

    def export_as_avi(self, result, output_dir, prefix):
        """支持彩色视频导出"""
        num_frames = result.shape[0]
        self.data_progress_signal.emit(0, num_frames)
        os.makedirs(output_dir, exist_ok=True)

        # 归一化处理
        normalized = self._normalize_data(result)

        # 确定视频参数
        height, width = normalized.shape[1:3]
        is_color = normalized.ndim == 4 and normalized.shape[3] in (3, 4)

        # 处理彩色数据 (RGB→BGR转换)
        if is_color:
            # 去除Alpha通道（如果需要）
            if normalized.shape[3] == 4:
                normalized = normalized[..., :3]
            # RGB转BGR
            normalized = normalized[..., ::-1]

        # 创建视频
        output_path = os.path.join(output_dir, f"{prefix}.avi")
        fps = max(10, min(30, num_frames // 10))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=is_color)

        for frame_idx in range(num_frames):
            frame = normalized[frame_idx]
            # 灰度视频需要单通道格式
            if not is_color and frame.ndim == 3:
                frame = frame.squeeze()
            out.write(frame)
            self.data_progress_signal.emit(frame_idx + 1, num_frames)

        out.release()
        logging.info(f'导出AVI完成: {output_path}, 共{num_frames}帧')
        return [output_path]

    def export_as_png(self, result, output_dir, prefix,is_temporal=True):
        """支持彩色PNG导出"""
        created_files = []
        # 归一化处理
        normalized = self._normalize_data(result)

        if is_temporal:
            num_frames = result.shape[0]
            num_digits = len(str(num_frames))
            self.data_progress_signal.emit(0, num_frames)
            for frame_idx in range(num_frames):
                frame_name = f"{prefix}-{frame_idx:0{num_digits}d}.png"
                output_path = os.path.join(output_dir, frame_name)

                frame = normalized[frame_idx]
                # 自动检测图像模式
                if frame.ndim == 2:
                    img = Image.fromarray(frame, 'L')
                elif frame.shape[2] == 4:
                    img = Image.fromarray(frame, 'RGBA')
                else:
                    img = Image.fromarray(frame, 'RGB')

                img.save(output_path)
                created_files.append(output_path)
                self.data_progress_signal.emit(frame_idx + 1, num_frames)

        else:
            num_frames = 1
            frame_name = f"{prefix}.png"
            output_path = os.path.join(output_dir, frame_name)
            if normalized.ndim == 2:
                img = Image.fromarray(normalized, 'L')
            elif normalized.shape[2] == 4:
                img = Image.fromarray(normalized, 'RGBA')
            img.save(output_path)
            created_files.append(output_path)
            self.data_progress_signal.emit(num_frames+1, num_frames)

        logging.info(f'导出PNG完成: {output_dir}, 共{num_frames}帧')
        return created_files

    def export_as_gif(self, result, output_dir, prefix,duration = 60):
        """优化彩色GIF导出"""
        num_frames = result.shape[0]
        self.data_progress_signal.emit(0, num_frames)

        # 归一化处理
        normalized = self._normalize_data(result)
        images = []
        palette_img = None

        for frame_idx in range(num_frames):
            frame = normalized[frame_idx]

            # 处理彩色帧
            if normalized.ndim == 4:
                # 去除Alpha通道
                if frame.shape[2] == 4:
                    frame = frame[..., :3]
                img = Image.fromarray(frame, 'RGB')

                # 使用全局调色板
                if palette_img is None:
                    palette_img = img.convert('P', palette=Image.ADAPTIVE, colors=256)
                    images.append(palette_img)
                else:
                    images.append(img.quantize(palette=palette_img))
            # 处理灰度帧
            else:
                images.append(Image.fromarray(frame, 'L'))

        output_path = os.path.join(output_dir, f"{prefix}.gif")
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            optimize=True
        )
        self.data_progress_signal.emit(num_frames, num_frames)
        logging.info(f'导出GIF完成: {output_path}, 共{num_frames}帧')
        return [output_path]

@dataclass
class Data:
    """
    数据导入类型\n
    data_origin 原始导入数据（经过初步处理）\n
    time_point 时间点（已匹配时间尺度）\n
    format_import 导入格式 \n
    image_import 原生成像数据\n
    parameters 其他参数\n
    name 数据命名\n
    timestamp 时间戳（用于识别匹配数据流）\n
    ROI_applied 是否应用ROI蒙版 \n
    history 历史保存（3组）\n
    serial_number 序号\n
    内含参数还有：self.datashape；
        self.timelength；
        self.framesize；
        self.dtype；
        self.datamax；
        self.datamin；
    """
    data_origin : np.ndarray
    time_point : np.ndarray
    format_import : str
    image_import : np.ndarray
    parameters : dict = None
    name : str = None
    timestamp : float = field(init=False, default_factory=time.time)
    ROI_applied : bool = field(init=False, default=False)
    history : ClassVar[deque] = deque(maxlen=3)
    serial_number: int = field(init=False)
    _counter : int = field(init=False, repr=False, default=0)
    _amend_counter : int = field(init=False, default=0)

    def __post_init__(self):
        Data._counter += 1
        self.serial_number = Data._counter # 生成序号
        self._recalculate()

        if self.name is None:
            self.name = f"{self.format_import}_{self.serial_number}"
        else:
            self.name = f"{self.name}_{self.serial_number}"
        Data.history.append(copy.deepcopy(self)) # 实例存储

    def _recalculate(self):
        self.datashape = self.data_origin.shape
        self.timelength = self.datashape[0] if self.data_origin.ndim == 3 else 1 # 默认不存在单像素点数据
        self.framesize = (self.datashape[1], self.datashape[2]) if self.data_origin.ndim == 3 else (self.datashape[0], self.datashape[1])
        self.datatype = self.data_origin.dtype
        self.datamax = self.data_origin.max()
        self.datamin = self.data_origin.min()

    def get_data_mean(self):
        return self.data_origin.mean()

    def get_data_std(self):
        return self.data_origin.std()

    def get_data_median(self):
        return np.median(self.data_origin)

    def update(self, **kwargs):
        Data._amend_counter += 1
        if 'data_origin' in kwargs:
            return self._create_new_instance(**kwargs)

        # 更新其他字段
        for key, value in kwargs.items():
            setattr(self, key, value)

        # 更新历史记录
        self._update_history()
        return self

    def apply_ROI(self, mask: np.ndarray):
        """设置 ROI 蒙版"""
        # 验证蒙版形状
        if mask is None:
            raise ValueError("无效蒙版")
        if mask.shape != self.datashape:
            raise ValueError(f"蒙版形状 {mask.shape} 与图像形状 {self.datashape} 不匹配")

        self.ROI_mask = mask

            # 根据蒙版类型应用不同的处理
        if self.ROI_mask.dtype == bool:
            # 布尔蒙版：将非 ROI 区域置零
            if self.ndim == 3:
                self.ROI_mask = copy.deepcopy(self.data_processed)
                for every_data in self.ROI_mask:
                    every_data[~mask] = self.datamin
            elif self.ndim == 2:
                self.ROI_mask = copy.deepcopy(self.data_processed)
                self.ROI_mask[~mask] = self.datamin
            else:
                raise ValueError("该数据无法应用ROI蒙版")
        else:
            # 数值蒙版：应用乘法操作
            if self.ndim >= 2:
                self.data_processed_ROI = self.data_processed * self.ROI_mask
            else:
                raise ValueError("该数据无法应用ROI蒙版")

        self.ROI_applied = True

    def _create_new_instance(self, **kwargs) -> 'Data':
        """创建新实例（当data_origin变更时）"""
        # 获取当前所有字段值
        current_values = {f.name: getattr(self, f.name) for f in fields(self) if f.init}

        # 应用更新
        current_values.update(kwargs)

        # 创建新实例（会分配新序列号）
        new_instance = Data(
            data_origin=current_values['data_origin'],
            time_point=current_values['time_point'],
            format_import=current_values['format_import'],
            image_import=current_values['image_import'],
            parameters=current_values.get('parameters'),
            name= f"{current_values.get('name')}@"
        )
        return new_instance

    def __setitem__(self, key, value):
        """字典式赋值支持"""
        valid_keys = [f.name for f in fields(self)]
        if key not in valid_keys:
            raise KeyError(f"Invalid field: {key}. Valid fields: {valid_keys}")

        setattr(self, key, value)

        # 特殊字段处理
        # 特殊处理：如果更新data_origin，创建新实例
        if key == 'data_origin':
            return self._create_new_instance(data_origin=value)

        setattr(self, key, value)

        # 更新历史记录中的当前实例
        self._update_history()

    def __repr__(self):
        return (
            f"Data<{self.name}: '{self.timestamp}'>' "
            f"({self.format_import}) | "
            f"Shape: {self.datashape} | "
            f"Range: [{self.datamin:.2f}, {self.datamax:.2f}] | "
            f"Time: {self.time_point[0] if self.time_point.size > 0 else 'N/A'}>"
        )

    @classmethod
    def find_history(cls, timestamp: float) -> Optional['ProcessedData']:
        """根据时间戳查找历史记录中的特定数据"""
        # 使用生成器表达式高效查找
        try:
            return next(
                (data for data in cls.history if abs(data.timestamp - timestamp) < 1e-6),
                None
            )
        except Exception as e:
            print(f"查找历史记录时出错: {e}")
            return None

    @classmethod
    def get_history_by_serial(cls, serial_number: int) -> Optional['Data']:
        """根据序列号获取历史记录并调整位置"""
        for i, record in enumerate(cls.history):
            if record.serial_number == serial_number:
                # 移除并重新添加以调整位置
                cls.history.remove(record)
                cls.history.append(record)
                return copy.deepcopy(record)
        return None

    # @classmethod
    # def get_history_by_timestamp(cls, timestamp: float) -> Optional['Data']:
    #     """根据时间戳获取历史记录并调整位置"""
    #     for i, record in enumerate(cls.history):
    #         if abs(record.timestamp - timestamp) < 1e-6:  # 浮点数精度处理
    #             # 移除并重新添加以调整位置
    #             cls.history.remove(record)
    #             cls.history.append(record)
    #             return copy.deepcopy(record)
    #     return None

    @classmethod
    def get_history_list(cls) -> list:
        """获取当前历史记录列表（按从旧到新排序）"""
        return list(cls.history)

    @classmethod
    def get_history_serial_numbers(cls) -> list:
        """获取历史记录的序列号列表（按从旧到新排序）"""
        return [record.serial_number for record in cls.history]

    @classmethod
    def get_history_timestamps(cls) -> List[float]:
        """获取历史记录的时间戳列表（按从旧到新排序）"""
        return [record.timestamp for record in cls.history]

    @classmethod
    def get_history_summary(cls) -> str:
        """获取历史记录的摘要信息"""
        summary = []
        for record in cls.history:
            summary.append(
                f"#{record.serial_number}: {record.name} "
                f"({time.strftime('%H:%M:%S', time.localtime(record.timestamp))})"
            )
        return " | ".join(summary)

    def _update_history(self):
        """更新历史记录中的当前实例"""
        # 查找历史记录中的当前实例
        for i, record in enumerate(Data.history):
            if record.serial_number == self.serial_number:
                # 更新历史记录中的实例
                Data.history[i] = copy.deepcopy(self)
                break
        return None

    @classmethod
    def clear_history(cls):
        """清空所有历史记录"""
        cls.history.clear()

@dataclass
class ProcessedData:
    """
    经过处理的数据\n
    timestamp_inherited 处理前数据的时间戳: float\n
    name 命名（需要更新）: str\n
    type_processed 处理类型（最后） : str\n
    time_point 时间点: np.ndarray\n
    data_processed 处理出来的数据（此处存放尤指具有时空尺度的核心数据）: np.ndarray = None\n
    out_processed 其他处理出来的数据（比如拟合得到的参数，二维序列等等）: dict = None\n
    timestamp 新数据时间戳\n
    ROI_applied 是否应用ROI蒙版 \n
    history 历史，无限保留，考虑和绘图挂钩: ClassVar[Dict[str, 'ProcessedData']] = {}\n
    """
    timestamp_inherited : float
    name : str
    type_processed : str
    time_point : np.ndarray = None
    data_processed: np.ndarray = None
    out_processed: dict = None
    timestamp : float = field(init=False, default_factory=time.time)
    ROI_applied : bool = False
    history: ClassVar[deque] = deque(maxlen=30)

    def __post_init__(self):
        if self.data_processed is not None:
            self.datashape = self.data_processed.shape if self.data_processed is not None else None
            self.timelength = self.datashape[0] if self.data_processed.ndim == 3 else 1  # 默认不存在单像素点数据
            if self.data_processed.ndim == 3 :
                self.framesize = (self.datashape[1], self.datashape[2])
            elif self.data_processed.ndim == 2 :
                self.framesize = (self.datashape[0], self.datashape[1])
            elif self.data_processed.ndim == 1 :
                self.framesize = (self.datashape[0])
            self.datamin = self.data_processed.min()
            self.datamax = self.data_processed.max()
            self.datatype = self.data_processed.dtype
            self.datamean = self.data_processed.mean()

        # 添加到历史记录
        ProcessedData.history.append(copy.deepcopy(self))

    def apply_ROI(self, mask: np.ndarray):
        """设置 ROI 蒙版"""
        # 验证蒙版形状
        if mask is None:
            raise ValueError("无效蒙版")
        if mask.shape != self.datashape:
            raise ValueError(f"蒙版形状 {mask.shape} 与图像形状 {self.datashape} 不匹配")

        self.ROI_mask = mask

            # 根据蒙版类型应用不同的处理
        if self.ROI_mask.dtype == bool:
            # 布尔蒙版：将非 ROI 区域置零
            if self.ndim == 3:
                self.ROI_mask = copy.deepcopy(self.data_processed)
                for every_data in self.ROI_mask:
                    every_data[~mask] = self.datamin
            elif self.ndim == 2:
                self.ROI_mask = copy.deepcopy(self.data_processed)
                self.ROI_mask[~mask] = self.datamin
            else:
                raise ValueError("该数据无法应用ROI蒙版")
        else:
            # 数值蒙版：应用乘法操作
            if self.ndim >= 2:
                self.data_processed = self.data_processed * self.ROI_mask
            else:
                raise ValueError("该数据无法应用ROI蒙版")

        self.ROI_applied = True

    @classmethod
    def find_history(cls, timestamp: float) -> Optional['ProcessedData']:
        """根据时间戳查找历史记录中的特定数据"""
        # 使用生成器表达式高效查找
        try:
            return next(
                (data for data in cls.history if abs(data.timestamp - timestamp) < 1e-6),
                None
            )
        except Exception as e:
            print(f"查找历史记录时出错: {e}")
            return None
    # @classmethod
    # def remove_from_history(cls, name: str):
    #     """从历史记录中删除指定名称的处理数据"""
    #     if name in cls.history:
    #         del cls.history[name]
    #
    @classmethod
    def clear_history(cls):
        """清空所有历史记录"""
        cls.history.clear()
    #
    # @classmethod
    # def get_by_name(cls, name: str) -> Optional['ProcessedData']:
    #     """通过名称获取处理数据"""
    #     return cls.history.get(name)
    #
    # @classmethod
    # def get_by_original_timestamp(cls, timestamp: float) -> List['ProcessedData']:
    #     """通过原始数据时间戳获取所有相关处理数据"""
    #     return [data for data in cls.history.values()
    #             if abs(data.timestamp_inherited - timestamp) < 1e-6]
    #
    # @classmethod
    # def get_by_processing_type(cls, processing_type: str) -> List['ProcessedData']:
    #     """通过处理类型获取所有相关处理数据"""
    #     return [data for data in cls.history.values()
    #             if data.type_processed == processing_type]
    #
    # @classmethod
    # def get_history_names(cls) -> List[str]:
    #     """获取所有历史记录的名称列表"""
    #     return list(cls.history.keys())

    def __repr__(self):
        return (
            f"ProcessedData<{self.name} | "
            f"Type: {self.type_processed} | "
            f"Shape: {self.datashape} | "
            f"Range: [{self.datamin:.2f}, {self.datamax:.2f}]>"
        )

@dataclass
class ImagingData:
    """
    图像显示类型\n
    timestamp_inherited 原始数据来源\n
    """
    timestamp_inherited: float
    image_backup: np.ndarray = None # 原始数据
    image_data: np.ndarray = None # 归一，放宽，整数化后的数据
    image_ROI: np.ndarray = None
    image_type: str = None
    colormode: str = None # 色彩模式，目前在sub类中实现
    canvas_num: int = field(default=0)
    is_temporary: bool = field(init=False ,default=False)
    timestamp : float = field(init=False, default_factory=time.time)

    def __post_init__(self):
        self.image_data = self.to_uint8(self.image_backup)
        self.imageshape = self.image_data.shape
        self.ndim = self.image_data.ndim
        self.totalframes = self.imageshape[0] if self.ndim == 3 else 1
        self.framesize = (self.imageshape[1], self.imageshape[2]) if self.ndim == 3 else (self.imageshape[0], self.imageshape[1])
        # 不考虑数据点只有一个的情况
        self.is_temporary = True if self.ndim == 3 else False
        self.imagemin = self.image_backup.min()
        self.imagemax = self.image_backup.max()
        self.datatype = self.image_backup.dtype
        self.ROI_mask = None
        self.ROI_applied = False

    @classmethod
    def create_image(cls, data_obj: Union['Data', 'ProcessedData'],*arg:str) -> 'ImagingData':
        """初始化ImagingData"""

        instance = cls.__new__(cls)

        # 设置图像数据
        if isinstance(data_obj, Data):
            # instance.image_data = data_obj.data_origin.copy()
            instance.image_backup = data_obj.data_origin.copy()
            instance.timestamp_inherited = data_obj.timestamp
            instance.source_type = "Data"
            instance.source_name = data_obj.name
            instance.source_format = data_obj.format_import
        elif isinstance(data_obj, ProcessedData):
            if arg:
                # instance.image_data = data_obj.out_processed[arg].copy()
                instance.image_backup =  data_obj.out_processed[arg].copy()
            else:
                # instance.image_data = data_obj.data_processed.copy()
                instance.image_backup = data_obj.data_processed.copy()
            instance.timestamp_inherited = data_obj.timestamp
            instance.source_type = "ProcessedData"
            instance.source_name = data_obj.name
            instance.source_format = data_obj.type_processed


        # 调用后初始化
        instance.__post_init__()
        return instance

    def apply_ROI(self, mask: np.ndarray):
        """设置 ROI 蒙版"""
        # 验证蒙版形状
        if mask is None:
            raise ValueError("无效蒙版")
        if mask.shape != self.imageshape:
            raise ValueError(f"蒙版形状 {mask.shape} 与图像形状 {self.imageshape} 不匹配")

        self.ROI_mask = mask

            # 根据蒙版类型应用不同的处理
        if self.ROI_mask.dtype == bool:
            # 布尔蒙版：将非 ROI 区域置零
            if self.ndim == 3:
                self.ROI_mask = copy.deepcopy(self.image_data)
                for every_data in self.ROI_mask:
                    every_data[~mask] = self.imagemin
            if self.ndim == 2:
                self.ROI_mask = copy.deepcopy(self.image_data)
                self.ROI_mask[~mask] = self.imagemin
            else:
                raise ValueError("该数据无法应用ROI蒙版")
        else:
            # 数值蒙版：应用乘法操作
            if self.ndim >= 2:
                self.image_ROI = self.image_data * self.ROI_mask
            else:
                raise ValueError("该数据无法应用ROI蒙版")

        self.ROI_applied = True

    def to_uint8(self,data = None):
        """归一化和数字类型调整"""
        if data is None:
            data = self.image_backup
        if data.dtype == np.uint8 and np.max(data) == 255:
            return data.copy()

        # 计算数组的最小值和最大值
        min_val = np.min(data)
        max_val = np.max(data)

        # if self.source_format == "ROI_stft" or self.source_format == "ROI_cwt":
        #     return ((data - np.min(data))/(np.max(data)- np.min(data))*255).astype(np.uint8)
        result = ((data - min_val)/(max_val- min_val)*255).astype(np.uint8)
        # 通用线性变换公式
        # 使用64位浮点保证精度，避免中间步骤溢出
        # scaled = (data.astype(np.float64) - min_val) * (255.0 / (max_val - min_val))
        #
        # # 四舍五入并确保在[0,255]范围内
        # result = np.clip(np.round(scaled), 0, 255).astype(np.uint8)
        return result

    # def to_colormap(self,colormode,min_value=None,max_value=None):
    #     """伪色彩实现（其实仅在生成视图时才会更新）"""
    #     logging.info("请稍等片刻，更换样式需重载数据")
    #     color_map_manager = ColorMapManager()
    #     if colormode is None:
    #         return None
    #     if self.is_temporary:
    #         T,H,W = self.imageshape
    #         self._signals.data_progress_signal(0,T)
    #         new_data = np.zeros((T,H,W, 4), dtype=np.uint8)
    #         for i,image in enumerate(self.image_backup):
    #             new_data[i] = color_map_manager.apply_colormap(
    #                                                                 image,
    #                                                                 colormode,
    #                                                                 min_value,
    #                                                                 max_value
    #                                                             )
    #             self._signals.data_progress_signal(i, T)
    #         self.image_data = new_data
    #         self._signals.data_progress_signal(T, T)
    #         return None
    #     else:
    #         H, W = self.imageshape
    #         new_data = np.zeros((H, W, 4), dtype=np.uint8)
    #         new_data = color_map_manager.apply_colormap(
    #             self.image_backup,
    #             colormode,
    #             min_value,
    #             max_value
    #         )
    #         self.image_data = new_data
    #         return None

    # def export_image(self,type):
    #     pass

    # @classmethod
    # def from_array(cls, image_array: np.ndarray, **metadata) -> 'ImageData':
    #     """从 NumPy 数组创建 ImageData"""
    #     instance = cls.__new__(cls)
    #     instance.image_data = image_array.copy()
    #     instance.source_type = "RawArray"
    #
    #     # 设置可选元数据
    #     instance.source_name = metadata.get('name')
    #     instance.source_serial = metadata.get('serial')
    #     instance.source_format = metadata.get('format', "Unknown")
    #
    #     # 初始化其他字段
    #     instance.ROI_mask = None
    #     instance.ROI_applied = False
    #
    #     # 调用后初始化
    #     instance.__post_init__()
    #     return instance

    def __repr__(self):
        return (
            f"ImageData<Source: {self.source_type}:{self.source_name} "
            f"| Shape: {self.imageshape} | "
            f"Range: [{self.imagemin:.2f}, {self.imagemax:.2f}]>"
        )

class ColorMapManager:
    """伪彩色映射管理器"""

    def __init__(self):
        self.colormaps = {
            "Jet": self.jet_colormap,
            "Hot": self.hot_colormap,
            # "Cool": self.cool_colormap,
            # "Spring": self.spring_colormap,
            # "Summer": self.summer_colormap,
            # "Autumn": self.autumn_colormap,
            # "Winter": self.winter_colormap,
            # "Bone": self.bone_colormap,
            # "Copper": self.copper_colormap,
            # "Greys": self.greys_colormap,
            # "Viridis": self.viridis_colormap,
            # "Plasma": self.plasma_colormap,
            # "Inferno": self.inferno_colormap,
            # "Magma": self.magma_colormap,
            # "Cividis": self.cividis_colormap,
            # "Rainbow": self.rainbow_colormap,
            # "Turbo": self.turbo_colormap
        }

        # 创建Matplotlib兼容的colormap
        self.matplotlib_cmaps = {
            "Jet": cm.jet,
            "Hot": cm.hot,
            "Cool": cm.cool,
            "Spring": cm.spring,
            "Summer": cm.summer,
            "Autumn": cm.autumn,
            "Winter": cm.winter,
            "Bone": cm.bone,
            "Copper": cm.copper,
            "Greys": cm.gray,
            "Viridis": cm.viridis,
            "Plasma": cm.plasma,
            "Inferno": cm.inferno,
            "Magma": cm.magma,
            "Cividis": cm.cividis,
            "Rainbow": self.create_rainbow_cmap(),
            "Turbo": cm.turbo,
            'CMRmap':cm.CMRmap,
            'gnuplot2':cm.gnuplot2,
        }

    def get_colormap_names(self):
        """获取所有可用的colormap名称"""
        return list(self.matplotlib_cmaps.keys())  # 暂时用Matplotlib

    def apply_colormap(self, image_data, colormap_name, min_val=None, max_val=None):
        """应用伪彩色映射到图像数据"""
        if colormap_name not in self.matplotlib_cmaps:
            colormap_name = "Jet"  # 默认使用Jet

        cmap = self.matplotlib_cmaps[colormap_name]

        # 归一化数据
        if min_val is None:
            min_val = np.min(image_data)
        if max_val is None:
            max_val = np.max(image_data)

        # 避免除以零
        if min_val == max_val:
            normalized = np.zeros_like(image_data)
        else:
            normalized = (image_data - min_val) / (max_val - min_val)
            normalized = np.clip(normalized, 0, 1)

        # 应用colormap
        colored = (cmap(normalized) * 255).astype(np.uint8)
        return colored

    def create_rainbow_cmap(self):
        """创建自定义彩虹colormap"""
        cdict = {
            'red': [(0.0, 1.0, 1.0),
                    (0.15, 0.0, 0.0),
                    (0.3, 0.0, 0.0),
                    (0.45, 0.0, 0.0),
                    (0.6, 1.0, 1.0),
                    (0.75, 1.0, 1.0),
                    (1.0, 1.0, 1.0)],
            'green': [(0.0, 0.0, 0.0),
                      (0.15, 0.0, 0.0),
                      (0.3, 1.0, 1.0),
                      (0.45, 1.0, 1.0),
                      (0.6, 1.0, 1.0),
                      (0.75, 0.0, 0.0),
                      (1.0, 0.0, 0.0)],
            'blue': [(0.0, 0.0, 0.0),
                     (0.15, 1.0, 1.0),
                     (0.3, 1.0, 1.0),
                     (0.45, 0.0, 0.0),
                     (0.6, 0.0, 0.0),
                     (0.75, 0.0, 0.0),
                     (1.0, 1.0, 1.0)]
        }
        return LinearSegmentedColormap('Rainbow', cdict)

    # 以下是各种colormap的实现（保留作为参考）
    def jet_colormap(self, value):
        """Jet colormap实现"""
        if value < 0.125:
            r = 0
            g = 0
            b = 0.5 + 4 * value
        elif value < 0.375:
            r = 0
            g = 4 * (value - 0.125)
            b = 1
        elif value < 0.625:
            r = 4 * (value - 0.375)
            g = 1
            b = 1 - 4 * (value - 0.375)
        elif value < 0.875:
            r = 1
            g = 1 - 4 * (value - 0.625)
            b = 0
        else:
            r = max(1 - 4 * (value - 0.875), 0)
            g = 0
            b = 0
        return (int(r * 255), int(g * 255), int(b * 255))

    def hot_colormap(self, value):
        """Hot colormap实现"""
        r = min(3 * value, 1.0)
        g = min(3 * value - 1, 1.0) if value > 1 / 3 else 0
        b = min(3 * value - 2, 1.0) if value > 2 / 3 else 0
        return (int(r * 255), int(g * 255), int(b * 255))

    # 其他colormap实现类似，这里省略以节省空间...
    # 实际使用中我们使用matplotlib的实现



