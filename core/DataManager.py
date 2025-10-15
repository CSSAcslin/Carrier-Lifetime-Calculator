import time
import copy
import numpy as np
from collections import deque
from typing import ClassVar, Optional, List, Dict, Any, Union
from PyQt5.QtCore import QObject, pyqtSignal
from dataclasses import dataclass, field, fields


@dataclass
class Data:
    """
    数据导入类型\n
    data_origin 原始导入数据（经过初步处理）\n
    time_point 时间点（已匹配时间尺度）\n
    format_import 导入格式 \n
    image_import 原生成像数据\n
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

        # Data.history.append( # 类级别存储
        #     {self.serial_number: {
        #         'data_origin' : self.data_origin,
        #         'time_point' : self.time_point,
        #         'format_import' : self.format_import,
        #         'image_import': self.image_import,
        #         'parameters' : self.parameters,
        #         'name' : self.name,
        #         'timestamp' : self.timestamp}}
        # )
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
    image_backup: np.ndarray = None
    image_data: np.ndarray = None
    image_ROI: np.ndarray = None
    image_type: str = None
    colormode: str = None
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
        self.imagemin = self.image_data.min()
        self.imagemax = self.image_data.max()
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

    def to_uint8(self,data):
        """归一化和数字类型调整"""
        # 如果已经是uint8类型且值在0-255范围内，直接返回
        # if data.dtype == np.uint8 and data.min() >= 0 and 1 <= data.max() <= 255:
        #     return data

        # 计算数组的最小值和最大值
        min_val = np.min(data)
        max_val = np.max(data)

        if self.source_format == "ROI_stft" or self.source_format == "ROI_cwt":
            return ((data - np.min(data))/(np.max(data)- np.min(data))*255).astype(np.uint8)

        # 处理常数数组的特殊情况
        # if min_val == max_val:
        #     # 根据常数值映射到0/128/255
        #     if min_val <= 0:
        #         return np.zeros_like(data, dtype=np.uint8)
        #     elif min_val >= 255:
        #         return np.full_like(data, 255, dtype=np.uint8)
        #     else:
        #         return np.full_like(data, round(min_val), dtype=np.uint8)
        #
        # # 如果已经是整数类型且在0-255范围内，直接转换
        # if np.issubdtype(data.dtype, np.integer) and min_val >= 0 and max_val <= 255:
        #     return data.astype(np.uint8)

        # 通用线性变换公式
        # 使用64位浮点保证精度，避免中间步骤溢出
        scaled = (data.astype(np.float64) - min_val) * (255.0 / (max_val - min_val))

        # 四舍五入并确保在[0,255]范围内
        result = np.clip(np.round(scaled), 0, 255).astype(np.uint8)
        return result

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

class DataManager(QObject):
    save_request_back = pyqtSignal(dict)
    read_request_back = pyqtSignal(dict)
    remove_request_back = pyqtSignal(dict)
    amend_request_back = pyqtSignal(dict)
    def __init__(self, parent=None):
        super(DataManager, self).__init__(parent)
        # 找数据用next
        # latest_aabb = next(
        #     (data for data in reversed(Data.history) if data.data_type == "aabb"),
        #     None
        # )