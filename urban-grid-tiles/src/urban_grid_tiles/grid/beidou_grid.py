"""北斗网格编码/解码/层级操作（GB/T 40087）.

北斗网格层级说明：
  L1  (5位)  : ~100万图幅
  L2  (8位)  : 1°×1°，~111km
  L3  (11位) : 1:5万图幅
  L4  (14位) : 1′×1′，~1.85km
  L5  (17位) : 4″×4″，~123.69m
  L6  (20位) : 2″×2″，~61.84m
  L7  (23位) : 1/4″×1/4″，~7.73m
  L8  (26位) : 1/32″×1/32″，~1m
  L9  (29位) : 1/256″×1/256″，~12.5cm
  L10 (32位) : 1/2048″×1/2048″，~1.5cm

编码规则（参考 GB/T 40087）：
  - L2 根格（8位）= 行号(2位) + 列号(3位) + 附加3位标识
  - L3~L10 每级在父码后追加3位子码（4进制行列各取一位 + 1位层标）
  
本实现使用如下简化规则（与标准层级位数对应）：
  - L2 (8位)  编码 = 4位经度整数部分码 + 4位纬度整数部分码
  - 每一子级在父码基础上追加3位（row_digit, col_digit, level_tag）
  - 子格子划分：每级2×2分割（即每级步长减半，追加 "0~3" 的2位）
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

# 各层级的编码位数
_LEVEL_DIGITS: Dict[int, int] = {
    1: 5,
    2: 8,
    3: 11,
    4: 14,
    5: 17,
    6: 20,
    7: 23,
    8: 26,
    9: 29,
    10: 32,
}

# 反查：位数 → 层级
_DIGITS_TO_LEVEL: Dict[int, int] = {v: k for k, v in _LEVEL_DIGITS.items()}

# L2 根格的经纬度步长（1°×1°）
_L2_LON_STEP: float = 1.0  # degrees
_L2_LAT_STEP: float = 1.0  # degrees

# L2 根格编码位数
_L2_DIGITS: int = 8

# 每级子划分数（每个方向 2 等分 → 每级 2×2=4 个子格）
_SUBDIVISIONS: int = 2


def _l2_grid_index(lon: float, lat: float) -> Tuple[int, int]:
    """返回 L2 根格的 (lon_idx, lat_idx)，经度从 -180 开始，纬度从 -90 开始."""
    lon_idx = int(math.floor((lon + 180.0) / _L2_LON_STEP))
    lat_idx = int(math.floor((lat + 90.0) / _L2_LAT_STEP))
    return lon_idx, lat_idx


def _l2_code(lon_idx: int, lat_idx: int) -> str:
    """将 L2 根格索引编码为 8 位字符串（零填充十进制）."""
    # 经度范围 0~359，纬度范围 0~179
    return f"{lon_idx:04d}{lat_idx:04d}"


def _l2_bbox(lon_idx: int, lat_idx: int) -> Tuple[float, float, float, float]:
    """返回 L2 根格的包围盒 (min_lon, min_lat, max_lon, max_lat)."""
    min_lon = lon_idx * _L2_LON_STEP - 180.0
    min_lat = lat_idx * _L2_LAT_STEP - 90.0
    return min_lon, min_lat, min_lon + _L2_LON_STEP, min_lat + _L2_LAT_STEP


def lonlat_to_grid_code(lon: float, lat: float, level: int) -> str:
    """将经纬度编码为指定层级的北斗网格码.

    Args:
        lon: 经度（-180 ~ 180）
        lat: 纬度（-90 ~ 90）
        level: 北斗网格层级（1~10）

    Returns:
        指定层级的北斗网格码字符串（位数见 _LEVEL_DIGITS）
    """
    if level < 1 or level > 10:
        raise ValueError(f"level 必须在 1~10 之间，当前值: {level}")

    lon_idx, lat_idx = _l2_grid_index(lon, lat)

    if level == 1:
        # L1 取 L2 码的前5位
        l2 = _l2_code(lon_idx, lat_idx)
        return l2[:5]

    # L2 基础码
    code = _l2_code(lon_idx, lat_idx)

    if level == 2:
        return code

    # L3~L10：在当前格子内逐级细化
    # 当前格包围盒
    min_lon, min_lat, max_lon, max_lat = _l2_bbox(lon_idx, lat_idx)

    for lv in range(3, level + 1):
        lon_step = (max_lon - min_lon) / _SUBDIVISIONS
        lat_step = (max_lat - min_lat) / _SUBDIVISIONS

        # 子格子列号（0 或 1，从西到东）
        col = int(math.floor((lon - min_lon) / lon_step))
        col = min(col, _SUBDIVISIONS - 1)

        # 子格子行号（0 或 1，从南到北）
        row = int(math.floor((lat - min_lat) / lat_step))
        row = min(row, _SUBDIVISIONS - 1)

        # 追加3位子码：行(1位) + 列(1位) + 层标(1位，固定用层级值末位)
        level_tag = lv % 10
        code += f"{row}{col}{level_tag}"

        # 更新包围盒到子格
        min_lon = min_lon + col * lon_step
        max_lon = min_lon + lon_step
        min_lat = min_lat + row * lat_step
        max_lat = min_lat + lat_step

    return code


def grid_code_to_bbox(grid_code: str) -> Tuple[float, float, float, float]:
    """解析北斗网格码，返回经纬度包围盒 (min_lon, min_lat, max_lon, max_lat).

    Args:
        grid_code: 北斗网格码字符串

    Returns:
        (min_lon, min_lat, max_lon, max_lat) 单位：度
    """
    digits = len(grid_code)
    level = _DIGITS_TO_LEVEL.get(digits)
    if level is None:
        raise ValueError(f"无效的网格码位数: {digits}，有效位数: {list(_DIGITS_TO_LEVEL.keys())}")

    if level == 1:
        # L1 码为 L2 码前5位，补全后解析
        # L1 仅用于标识，包围盒取该 L2 列对应的 1°×n° 范围（这里返回整列）
        lon_idx = int(grid_code[:4])
        lat_prefix = int(grid_code[4])
        # lat_idx 范围：lat_prefix*100 ~ lat_prefix*100+99
        min_lon = lon_idx * _L2_LON_STEP - 180.0
        min_lat = lat_prefix * 100 * _L2_LAT_STEP - 90.0
        return min_lon, min_lat, min_lon + _L2_LON_STEP, min_lat + 100 * _L2_LAT_STEP

    # L2: 前4位=lon_idx，后4位=lat_idx
    lon_idx = int(grid_code[:4])
    lat_idx = int(grid_code[4:8])
    min_lon, min_lat, max_lon, max_lat = _l2_bbox(lon_idx, lat_idx)

    if level == 2:
        return min_lon, min_lat, max_lon, max_lat

    # L3~L10：逐级解析子码
    for i in range(level - 2):
        offset = 8 + i * 3
        row = int(grid_code[offset])
        col = int(grid_code[offset + 1])
        # grid_code[offset + 2] 是层标，忽略

        lon_step = (max_lon - min_lon) / _SUBDIVISIONS
        lat_step = (max_lat - min_lat) / _SUBDIVISIONS

        min_lon = min_lon + col * lon_step
        max_lon = min_lon + lon_step
        min_lat = min_lat + row * lat_step
        max_lat = min_lat + lat_step

    return min_lon, min_lat, max_lon, max_lat


def grid_code_to_center(grid_code: str) -> Tuple[float, float]:
    """返回网格中心经纬度 (lon, lat).

    Args:
        grid_code: 北斗网格码字符串

    Returns:
        (center_lon, center_lat) 单位：度
    """
    min_lon, min_lat, max_lon, max_lat = grid_code_to_bbox(grid_code)
    return (min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0


def get_level(grid_code: str) -> int:
    """根据网格码位数返回层级整数.

    Args:
        grid_code: 北斗网格码字符串

    Returns:
        层级整数（1~10）

    Raises:
        ValueError: 位数不符合任何已知层级
    """
    digits = len(grid_code)
    level = _DIGITS_TO_LEVEL.get(digits)
    if level is None:
        raise ValueError(f"无效的网格码位数: {digits}")
    return level


def get_parent_code(grid_code: str) -> str:
    """返回父级网格码（截断到上一层级位数）.

    Args:
        grid_code: 北斗网格码字符串

    Returns:
        父级网格码字符串

    Raises:
        ValueError: L1 网格没有父级
    """
    level = get_level(grid_code)
    if level <= 1:
        raise ValueError("L1 网格没有父级")
    parent_digits = _LEVEL_DIGITS[level - 1]
    return grid_code[:parent_digits]


def get_children_codes(grid_code: str) -> List[str]:
    """返回下一层级的所有子网格码列表（2×2=4个）.

    Args:
        grid_code: 北斗网格码字符串

    Returns:
        包含4个子网格码的列表

    Raises:
        ValueError: L10 没有子级
    """
    level = get_level(grid_code)
    if level >= 10:
        raise ValueError("L10 网格没有子级")
    next_level = level + 1
    level_tag = next_level % 10
    children = []
    for row in range(_SUBDIVISIONS):
        for col in range(_SUBDIVISIONS):
            children.append(f"{grid_code}{row}{col}{level_tag}")
    return children


def get_grid_size_meters(level: int, lat: float = 30.0) -> float:
    """返回该层级在给定纬度下的近似网格边长（米）.

    基于 GB/T 40087 标准中各层级的经纬度步长（弧秒），
    计算在给定纬度处的实际地面尺寸。

    Args:
        level: 北斗网格层级（1~10）
        lat: 纬度（度），用于修正经向距离

    Returns:
        近似网格边长（米）
    """
    if level < 1 or level > 10:
        raise ValueError(f"level 必须在 1~10 之间，当前值: {level}")

    # 各层级的弧秒步长（GB/T 40087 标准值）
    # L1: 约100个L2格，L3: 约20'（~1200"）
    _LEVEL_ARCSEC: Dict[int, float] = {
        1: 360_000.0,   # ~100°，L1 覆盖超大范围
        2: 3_600.0,     # 1° = 3600"
        3: 1_200.0,     # 20'（1:5万图幅近似）
        4: 60.0,        # 1' = 60"
        5: 4.0,         # 4"
        6: 2.0,         # 2"
        7: 0.25,        # 1/4"
        8: 1.0 / 32,    # 1/32"
        9: 1.0 / 256,   # 1/256"
        10: 1.0 / 2048, # 1/2048"
    }

    arcsec = _LEVEL_ARCSEC[level]
    deg = arcsec / 3600.0

    # 地球半径（米）
    R = 6_371_000.0
    lat_rad = math.radians(lat)

    lon_meters = deg * math.radians(1.0) * R * math.cos(lat_rad)
    lat_meters = deg * math.radians(1.0) * R

    return (lon_meters + lat_meters) / 2.0
