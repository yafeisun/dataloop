from enum import IntEnum

__app_name__ = 'ipm_checker'
__version__ = '0.7.2'


class StateCode(IntEnum):  # pyright: ignore[reportRedeclaration]
    success = 0  # 成功
    error_input_param = 2  # 输入参数错误
    # 11-40 warning 提示

    # 41-70 警告类型

    # 71-100 错误类型
    # 车辆参数类型 71-90
    error_missing_car_yaml = 71,  # type: ignore # 缺少车辆参数文件
    error_missing_view = 72,  # pyright: ignore[reportAssignmentType] # 缺少视图
    error_missing_frame = 73,  # pyright: ignore[reportAssignmentType] # 缺少帧
    error_shape_error = 75,  # pyright: ignore[reportAssignmentType] # shape not matching
    # 其他错误类型 91-99
    error_unknown = 91,  # pyright: ignore[reportAssignmentType] # 未知
    # 13-10 提示
    # other 自定义错误类型


class RunState:
    code: StateCode = StateCode.success
    msg = 'Ok'


class IPMException(Exception):
    code = 0


class CarYamlMissingError(IPMException):
    code = StateCode.error_missing_car_yaml


class ViewMissingError(IPMException):
    code = StateCode.error_missing_view


class FrameAlignError(IPMException):
    code = StateCode.error_missing_frame


class ParamShapeError(IPMException):
    code = StateCode.error_shape_error