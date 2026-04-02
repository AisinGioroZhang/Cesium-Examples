"""射线视域分析（LOS，基于高度场，numpy 向量化）.

从雷达位置向周围发射射线，判断每条射线在步进过程中是否被地形/建筑遮挡，
返回可见与被遮挡的北斗网格码集合。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Set, Tuple

import numpy as np

from urban_grid_tiles.grid.beidou_grid import (
    get_grid_size_meters,
    lonlat_to_grid_code,
)


def compute_los(
    radar_lon: float,
    radar_lat: float,
    radar_alt: float,
    height_field: Any,
    max_range_m: float,
    level: int,
    azimuth_step_deg: float = 1.0,
    elevation_angles_deg: Tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 45.0),
) -> Dict[str, Set[str]]:
    """基于高度场做射线步进，计算雷达可探测区域.

    对每个方位角 + 俯仰角组合发射射线，沿射线按网格步长步进，
    判断是否被地形/建筑遮挡，累积可见与遮挡的网格码集合。

    Args:
        radar_lon: 雷达经度（度）
        radar_lat: 雷达纬度（度）
        radar_alt: 雷达高度（米，绝对高程）
        height_field: HeightField 实例
        max_range_m: 最大探测距离（米）
        level: 使用的北斗网格层级
        azimuth_step_deg: 方位角步长（度），越小精度越高
        elevation_angles_deg: 俯仰角列表（度，负值向下，正值向上）

    Returns:
        {"visible": set[str], "occluded": set[str]}
    """
    # 地球半径（米），用于经纬度 ↔ 距离转换
    R = 6_371_000.0

    # 网格步长（米），用于射线步进
    grid_size = get_grid_size_meters(level, radar_lat)
    step_m = grid_size * 0.8  # 步长略小于格子大小，避免跨越漏判

    # 度/米 换算
    deg_per_m_lat = 1.0 / (math.pi / 180.0 * R)
    deg_per_m_lon = 1.0 / (math.pi / 180.0 * R * math.cos(math.radians(radar_lat)))

    visible: Set[str] = set()
    occluded: Set[str] = set()

    azimuths = np.arange(0.0, 360.0, azimuth_step_deg)

    for el_deg in elevation_angles_deg:
        el_rad = math.radians(el_deg)
        cos_el = math.cos(el_rad)
        sin_el = math.sin(el_rad)

        # 水平速度分量（每步）
        step_h = step_m * cos_el  # 水平分量（米/步）
        step_v = step_m * sin_el  # 垂直分量（米/步）

        # 向量化方位角
        az_rad = np.radians(azimuths)
        dx_m = step_h * np.sin(az_rad)  # 东向（米/步）
        dy_m = step_h * np.cos(az_rad)  # 北向（米/步）

        # 每条射线独立步进
        n_rays = len(azimuths)
        cur_lons = np.full(n_rays, radar_lon)
        cur_lats = np.full(n_rays, radar_lat)
        cur_alts = np.full(n_rays, radar_alt)
        blocked = np.zeros(n_rays, dtype=bool)

        n_steps = int(max_range_m / step_m) + 1
        for _ in range(n_steps):
            # 步进
            cur_lons = cur_lons + dx_m * deg_per_m_lon
            cur_lats = cur_lats + dy_m * deg_per_m_lat
            cur_alts = cur_alts + step_v

            # 计算当前位置到雷达的水平距离
            dist_m = np.sqrt(
                ((cur_lons - radar_lon) / deg_per_m_lon) ** 2
                + ((cur_lats - radar_lat) / deg_per_m_lat) ** 2
            )
            in_range = dist_m <= max_range_m

            # 获取当前位置地面高度（逐点查询，避免大量内存分配）
            active = in_range & ~blocked
            if not np.any(active):
                break

            active_idx = np.where(active)[0]
            for idx in active_idx:
                lon = float(cur_lons[idx])
                lat = float(cur_lats[idx])
                alt = float(cur_alts[idx])

                surface_h = height_field.get_height(lon, lat)
                if surface_h is None:
                    surface_h = 0.0

                code = lonlat_to_grid_code(lon, lat, level)

                if alt <= surface_h:
                    # 被遮挡
                    blocked[idx] = True
                    occluded.add(code)
                else:
                    visible.add(code)

    # 被遮挡的网格从可见集合中移除
    visible -= occluded
    return {"visible": visible, "occluded": occluded}
