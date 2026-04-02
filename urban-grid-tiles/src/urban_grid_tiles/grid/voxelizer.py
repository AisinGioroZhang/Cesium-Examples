"""三维体素化：水平北斗网格 × 垂直分层."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from urban_grid_tiles.grid.beidou_grid import (
    get_grid_size_meters,
    grid_code_to_center,
    lonlat_to_grid_code,
)


class Voxelizer:
    """将区域离散化为三维体素（水平北斗网格 × 垂直分层）.

    Args:
        height_field: HeightField 实例，用于确定体素占用状态
        vertical_min: 最低绝对高程（米）
        vertical_max: 最高绝对高程（米）
        vertical_step: 垂直层步长（米）
    """

    # 占用状态编码
    OCCUPANCY_AIR = "air"
    OCCUPANCY_TERRAIN = "terrain"
    OCCUPANCY_BUILDING = "building"

    def __init__(
        self,
        height_field: Any,  # HeightField，避免循环导入用 Any
        vertical_min: float = 0.0,
        vertical_max: float = 500.0,
        vertical_step: float = 10.0,
    ) -> None:
        self.height_field = height_field
        self.vertical_min = vertical_min
        self.vertical_max = vertical_max
        self.vertical_step = vertical_step

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def voxelize(
        self,
        bbox: Tuple[float, float, float, float],
        level: int,
    ) -> List[Dict[str, Any]]:
        """对指定包围盒区域按北斗层级进行体素化.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat) 单位：度
            level: 北斗网格层级（水平分辨率）

        Returns:
            体素字典列表，每个体素包含：
            - grid_code: str
            - layer: int
            - center_lon: float
            - center_lat: float
            - center_alt: float
            - size_h: float
            - size_v: float
            - occupancy: str
        """
        grid_codes = self._collect_grid_codes(bbox, level)
        voxels: List[Dict[str, Any]] = []

        n_layers = int(math.ceil((self.vertical_max - self.vertical_min) / self.vertical_step))
        size_h = get_grid_size_meters(level)
        size_v = self.vertical_step

        for code in grid_codes:
            center_lon, center_lat = grid_code_to_center(code)
            surface_h = self.height_field.get_height(center_lon, center_lat)

            for layer_idx in range(n_layers):
                alt_bottom = self.vertical_min + layer_idx * self.vertical_step
                alt_center = alt_bottom + self.vertical_step / 2.0

                if surface_h is not None and alt_bottom < surface_h:
                    # 高度场明确指示是地形或建筑
                    occupancy = self._classify_occupancy(
                        center_lon, center_lat, alt_bottom
                    )
                else:
                    occupancy = self.OCCUPANCY_AIR

                voxels.append(
                    {
                        "grid_code": code,
                        "layer": layer_idx,
                        "center_lon": center_lon,
                        "center_lat": center_lat,
                        "center_alt": alt_center,
                        "size_h": size_h,
                        "size_v": size_v,
                        "occupancy": occupancy,
                    }
                )
        return voxels

    def voxelize_to_numpy(
        self,
        bbox: Tuple[float, float, float, float],
        level: int,
    ) -> np.ndarray:
        """体素化并返回 numpy structured array（便于向量化处理）.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat) 单位：度
            level: 北斗网格层级

        Returns:
            numpy structured array，字段同 voxelize() 返回的字典
        """
        voxels = self.voxelize(bbox, level)
        if not voxels:
            dtype = np.dtype(
                [
                    ("grid_code", "U32"),
                    ("layer", np.int32),
                    ("center_lon", np.float64),
                    ("center_lat", np.float64),
                    ("center_alt", np.float64),
                    ("size_h", np.float64),
                    ("size_v", np.float64),
                    ("occupancy", "U10"),
                ]
            )
            return np.empty(0, dtype=dtype)

        grid_codes = [v["grid_code"] for v in voxels]
        layers = np.array([v["layer"] for v in voxels], dtype=np.int32)
        center_lons = np.array([v["center_lon"] for v in voxels], dtype=np.float64)
        center_lats = np.array([v["center_lat"] for v in voxels], dtype=np.float64)
        center_alts = np.array([v["center_alt"] for v in voxels], dtype=np.float64)
        size_hs = np.array([v["size_h"] for v in voxels], dtype=np.float64)
        size_vs = np.array([v["size_v"] for v in voxels], dtype=np.float64)
        occupancies = [v["occupancy"] for v in voxels]

        n = len(voxels)
        dtype = np.dtype(
            [
                ("grid_code", "U32"),
                ("layer", np.int32),
                ("center_lon", np.float64),
                ("center_lat", np.float64),
                ("center_alt", np.float64),
                ("size_h", np.float64),
                ("size_v", np.float64),
                ("occupancy", "U10"),
            ]
        )
        arr = np.empty(n, dtype=dtype)
        arr["grid_code"] = grid_codes
        arr["layer"] = layers
        arr["center_lon"] = center_lons
        arr["center_lat"] = center_lats
        arr["center_alt"] = center_alts
        arr["size_h"] = size_hs
        arr["size_v"] = size_vs
        arr["occupancy"] = occupancies
        return arr

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _collect_grid_codes(
        self,
        bbox: Tuple[float, float, float, float],
        level: int,
    ) -> List[str]:
        """枚举 bbox 内所有北斗网格码（光栅扫描）.

        使用网格步长进行采样，保证每个格子至少有一个采样点落在 bbox 内。
        """
        from urban_grid_tiles.grid.beidou_grid import _L2_LON_STEP, _L2_LAT_STEP, _SUBDIVISIONS

        min_lon, min_lat, max_lon, max_lat = bbox

        # 计算该层级的经纬度步长
        lon_step = _L2_LON_STEP
        lat_step = _L2_LAT_STEP
        for _ in range(level - 2):
            lon_step /= _SUBDIVISIONS
            lat_step /= _SUBDIVISIONS

        seen: set = set()
        codes: List[str] = []

        lat = min_lat + lat_step / 2.0
        while lat < max_lat:
            lon = min_lon + lon_step / 2.0
            while lon < max_lon:
                code = lonlat_to_grid_code(lon, lat, level)
                if code not in seen:
                    seen.add(code)
                    codes.append(code)
                lon += lon_step
            lat += lat_step

        return codes

    def _classify_occupancy(
        self, lon: float, lat: float, alt: float
    ) -> str:
        """根据高度场判断体素占用类型.

        Args:
            lon: 经度
            lat: 纬度
            alt: 体素底部高度（米）

        Returns:
            "terrain" / "building" / "air"
        """
        hf = self.height_field
        # 如果高度场支持分层查询（terrain_height、building_height），则区分
        if hasattr(hf, "get_terrain_height") and hasattr(hf, "get_building_height"):
            terrain_h = hf.get_terrain_height(lon, lat) or 0.0
            building_h = hf.get_building_height(lon, lat) or 0.0
            if building_h > terrain_h and alt < building_h:
                return self.OCCUPANCY_BUILDING
            if alt < terrain_h:
                return self.OCCUPANCY_TERRAIN
            return self.OCCUPANCY_AIR
        # 简单高度场
        surface_h = hf.get_height(lon, lat) or 0.0
        if alt < surface_h:
            return self.OCCUPANCY_TERRAIN
        return self.OCCUPANCY_AIR
