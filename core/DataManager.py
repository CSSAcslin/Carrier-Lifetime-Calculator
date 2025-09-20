import time
import copy
import numpy as np
from collections import deque
from typing import ClassVar, Optional, List, Dict
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
    _history 历史保存（3组）\n
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
    _history : ClassVar[deque] = deque(maxlen=3)
    serial_number: int = field(init=False)
    _counter : int = field(init=False, repr=False, default=0)
    _amend_counter : int = field(init=False, default=0)

    def __post_init__(self):
        Data._counter += 1
        self.serial_number = Data._counter # 生成序号
        self._recalculate()

        if self.name is None:
            self.name = f"{self.format_import}_{self.serial_number}"

        # Data._history.append( # 类级别存储
        #     {self.serial_number: {
        #         'data_origin' : self.data_origin,
        #         'time_point' : self.time_point,
        #         'format_import' : self.format_import,
        #         'image_import': self.image_import,
        #         'parameters' : self.parameters,
        #         'name' : self.name,
        #         'timestamp' : self.timestamp}}
        # )
        Data._history.append(copy.deepcopy(self)) # 实例存储

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
    def get_history_by_serial(cls, serial_number: int) -> Optional['Data']:
        """根据序列号获取历史记录并调整位置"""
        for i, record in enumerate(cls._history):
            if record.serial_number == serial_number:
                # 移除并重新添加以调整位置
                cls._history.remove(record)
                cls._history.append(record)
                return copy.deepcopy(record)
        return None

    @classmethod
    def get_history_by_timestamp(cls, timestamp: float) -> Optional['Data']:
        """根据时间戳获取历史记录并调整位置"""
        for i, record in enumerate(cls._history):
            if abs(record.timestamp - timestamp) < 1e-6:  # 浮点数精度处理
                # 移除并重新添加以调整位置
                cls._history.remove(record)
                cls._history.append(record)
                return copy.deepcopy(record)
        return None

    @classmethod
    def get_history_list(cls) -> list:
        """获取当前历史记录列表（按从旧到新排序）"""
        return list(cls._history)

    @classmethod
    def get_history_serial_numbers(cls) -> list:
        """获取历史记录的序列号列表（按从旧到新排序）"""
        return [record.serial_number for record in cls._history]

    @classmethod
    def get_history_timestamps(cls) -> List[float]:
        """获取历史记录的时间戳列表（按从旧到新排序）"""
        return [record.timestamp for record in cls._history]

    @classmethod
    def get_history_summary(cls) -> str:
        """获取历史记录的摘要信息"""
        summary = []
        for record in cls._history:
            summary.append(
                f"#{record.serial_number}: {record.name} "
                f"({time.strftime('%H:%M:%S', time.localtime(record.timestamp))})"
            )
        return " | ".join(summary)

    def _update_history(self):
        """更新历史记录中的当前实例"""
        # 查找历史记录中的当前实例
        for i, record in enumerate(Data._history):
            if record.serial_number == self.serial_number:
                # 更新历史记录中的实例
                Data._history[i] = copy.deepcopy(self)
                break
        return None

    @classmethod
    def clear_history(cls):
        """清空所有历史记录"""
        cls._history.clear()

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

    _history 历史，无限保留，考虑和绘图挂钩: ClassVar[Dict[str, 'ProcessedData']] = {}\n
    """
    timestamp_inherited : float
    name : str
    type_processed : str
    time_point : np.ndarray = None
    data_processed: np.ndarray = None
    out_processed: dict = None
    timestamp : float = field(init=False, default_factory=time.time)

    _history: ClassVar[Dict[str, 'ProcessedData']] = {}

    def __post_init__(self):
        if self.data_processed is not None:
            self.datashape = self.data_processed.shape
            self.timelength = self.datashape[0] if self.data_processed.ndim == 3 else 1  # 默认不存在单像素点数据
            self.framesize = (self.datashape[1], self.datashape[2]) if self.data_processed.ndim == 3 else (
                self.datashape[0], self.datashape[1])
            self.datamin = self.data_processed.min()
            self.datamax = self.data_processed.max()
            self.datatype = self.data_processed.dtype
            self.datamean = self.data_processed.mean()

        # 添加到历史记录
        ProcessedData._history[self.name] = self

    @classmethod
    def remove_from_history(cls, name: str):
        """从历史记录中删除指定名称的处理数据"""
        if name in cls._history:
            del cls._history[name]

    @classmethod
    def clear_history(cls):
        """清空所有历史记录"""
        cls._history.clear()

    @classmethod
    def get_by_name(cls, name: str) -> Optional['ProcessedData']:
        """通过名称获取处理数据"""
        return cls._history.get(name)

    @classmethod
    def get_by_original_timestamp(cls, timestamp: float) -> List['ProcessedData']:
        """通过原始数据时间戳获取所有相关处理数据"""
        return [data for data in cls._history.values()
                if abs(data.timestamp_inherited - timestamp) < 1e-6]

    @classmethod
    def get_by_processing_type(cls, processing_type: str) -> List['ProcessedData']:
        """通过处理类型获取所有相关处理数据"""
        return [data for data in cls._history.values()
                if data.type_processed == processing_type]

    @classmethod
    def get_history_names(cls) -> List[str]:
        """获取所有历史记录的名称列表"""
        return list(cls._history.keys())

    def __repr__(self):
        return (
            f"ProcessedData<{self.name} | "
            f"Type: {self.type_processed} | "
            f"Shape: {self.data_shape} | "
            f"Range: [{self.data_min:.2f}, {self.data_max:.2f}]>"
        )

@dataclass
class ImagingData:
    """
    图像显示类型\n
    timestamp_inherited 原始数据来源\n
    """
    timestamp_inherited: float
    image_data: np.ndarray
    image_type: str = None
    colormode: str = None
    region_code: int = field(default=1)
    _is_temporary: bool = field(init=False ,default=False)

    def __post_init__(self):
        self.image_shape = self.image_data.shape
        self.total_frames = self.image_shape[0] if self.image_data.ndim == 3 else 1
        self.frame_size = (self.image_shape[1], self.image_shape[2]) if self.image_data.ndim == 3 else (self.image_shape[0], self.image_shape[1])
        # 不考虑数据点只有一个的情况
        self._is_temporary = True if self.image_data.ndim == 3 else False



# test = Data(np.arange(1),time_point=np.arange(1),format_import="test",image_import=np.ones((1,1)))
# test2 = Data(np.arange(2),time_point=np.arange(2),format_import="test",image_import=np.ones((2,2)))
# test3 = Data(np.arange(3),time_point=np.arange(3),format_import="test",image_import=np.ones((3,3)))
# print(test3.history)
# test4 = Data(np.arange(4),time_point=np.arange(4),format_import="test",image_import=np.ones((4,4)))
#
# print(test)
# print(test2)
# print(test3)
# print(test3.history)
# print(test4)
# print(test4.history)

# if __name__ == "__main__":
#     # 创建测试数据
#     data1 = Data(
#         data_origin=np.random.rand(100, 100),
#         time_point=np.array([0.5, 1.0, 1.5]),
#         format_import="CSV",
#         image_import=np.zeros((100, 100))
#     )
#
#     time.sleep(0.1)  # 确保时间戳不同
#
#     data2 = Data(
#         data_origin=np.random.rand(50, 50),
#         time_point=np.array([2.0, 2.5]),
#         format_import="JSON",
#         image_import=np.ones((50, 50))
#     )
#
#     time.sleep(0.1)
#
#     data3 = Data(
#         data_origin=np.random.rand(80, 80),
#         time_point=np.array([3.0, 3.5, 4.0]),
#         format_import="XML",
#         image_import=np.full((80, 80), 0.5)
#     )
#
#     print("初始历史记录:")
#     print("序列号:", Data.get_history_serial_numbers())
#     print("时间戳:", Data.get_history_timestamps())
#     print("摘要:", Data.get_history_summary())
#
#     # 按序列号获取历史记录
#     retrieved = Data.get_history_by_serial(data1.serial_number)
#     print("\n按序列号获取历史记录:", retrieved)
#     print("调整后的历史记录:")
#     print("序列号:", Data.get_history_serial_numbers())
#     print("时间戳:", Data.get_history_timestamps())
#
#     # 按时间戳获取历史记录
#     retrieved = Data.get_history_by_timestamp(data2.timestamp)
#     print("\n按时间戳获取历史记录:", retrieved)
#     print("调整后的历史记录:")
#     print("序列号:", Data.get_history_serial_numbers())
#     print("时间戳:", Data.get_history_timestamps())
#     print("摘要:", Data.get_history_summary())



class DataManager(QObject):
    save_request_back = pyqtSignal(dict)
    read_request_back = pyqtSignal(dict)
    remove_request_back = pyqtSignal(dict)
    amend_request_back = pyqtSignal(dict)
    def __init__(self, parent=None):
        super(DataManager, self).__init__(parent)