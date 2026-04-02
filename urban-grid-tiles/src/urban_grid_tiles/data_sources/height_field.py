"""统一高度场构建：合并地形 GeoTIFF + 建筑矢量."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

from urban_grid_tiles.grid.beidou_grid import (
    _L2_LAT_STEP,
    _L2_LON_STEP,
    _SUBDIVISIONS,
    lonlat_to_grid_code,
    grid_code_to_center,
    grid_code_to_bbox,
)


class HeightField:
    """统一高度场：以 numpy 数组存储，与北斗网格对齐.

    Attributes:
        _data: 二维 float32 数组，shape (n_rows, n_cols)
        _bbox: (min_lon, min_lat, max_lon, max_lat)
        _lon_step: 每列经度步长（度）
        _lat_step: 每行纬度步长（度）
        _terrain_data: 地形高度数组（可选）
        _building_data: 建筑高度数组（可选）
    """

    def __init__(
        self,
        data: np.ndarray,
        bbox: Tuple[float, float, float, float],
        lon_step: float,
        lat_step: float,
        terrain_data: np.ndarray | None = None,
        building_data: np.ndarray | None = None,
    ) -> None:
        self._data = data.astype(np.float32)
        self._bbox = bbox
        self._lon_step = lon_step
        self._lat_step = lat_step
        self._terrain_data = terrain_data
        self._building_data = building_data

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_tif(
        cls,
        tif_path: str | Path,
        level: int,
        bbox: Tuple[float, float, float, float],
    ) -> "HeightField":
        """从 GeoTIFF 读取地形高程，重采样到指定北斗层级分辨率.

        Args:
            tif_path: GeoTIFF 路径
            level: 目标北斗层级
            bbox: 目标包围盒 (min_lon, min_lat, max_lon, max_lat)

        Returns:
            HeightField 实例
        """
        from urban_grid_tiles.data_sources.tif_loader import TifLoader

        loader = TifLoader(tif_path)
        loader.load()

        lon_step, lat_step = _grid_steps(level)
        min_lon, min_lat, max_lon, max_lat = bbox

        n_cols = max(1, int(round((max_lon - min_lon) / lon_step)))
        n_rows = max(1, int(round((max_lat - min_lat) / lat_step)))

        # 向量化采样
        lons = np.linspace(min_lon + lon_step / 2, max_lon - lon_step / 2, n_cols)
        lats = np.linspace(min_lat + lat_step / 2, max_lat - lat_step / 2, n_rows)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        heights = np.vectorize(loader.get_height)(lon_grid, lat_grid)
        heights = np.where(heights is None, 0.0, heights).astype(np.float32)

        return cls(heights, bbox, lon_step, lat_step, terrain_data=heights.copy())

    @classmethod
    def from_vector(
        cls,
        vector_path: str | Path,
        level: int,
        bbox: Tuple[float, float, float, float],
        height_field_attr: str = "height",
    ) -> "HeightField":
        """从 shp/geojson 矢量栅格化建筑高度.

        Args:
            vector_path: 矢量文件路径
            level: 目标北斗层级
            bbox: 目标包围盒
            height_field_attr: 建筑高度属性字段名

        Returns:
            HeightField 实例
        """
        from urban_grid_tiles.data_sources.vector_loader import VectorLoader

        loader = VectorLoader(vector_path, height_attr=height_field_attr)
        loader.load()

        lon_step, lat_step = _grid_steps(level)
        arr, actual_bbox = loader.rasterize(bbox, lon_step, lat_step)

        return cls(arr, actual_bbox, lon_step, lat_step, building_data=arr.copy())

    @staticmethod
    def merge(
        terrain_hf: "HeightField",
        building_hf: "HeightField",
    ) -> "HeightField":
        """合并地形高度场和建筑高度场（取最大值）.

        两个高度场需具有相同的 bbox、分辨率。若分辨率不同，以地形高度场为准，
        对建筑高度场进行最近邻重采样。

        Args:
            terrain_hf: 地形高度场
            building_hf: 建筑高度场

        Returns:
            合并后的 HeightField 实例
        """
        # 若尺寸不一致，简单取地形 shape
        terrain = terrain_hf._data
        building = building_hf._data

        if terrain.shape != building.shape:
            # 最近邻重采样 building 到 terrain shape
            from scipy.ndimage import zoom

            zoom_r = terrain.shape[0] / building.shape[0]
            zoom_c = terrain.shape[1] / building.shape[1]
            building = zoom(building, (zoom_r, zoom_c), order=0).astype(np.float32)

        merged = np.maximum(terrain, building)

        return HeightField(
            merged,
            terrain_hf._bbox,
            terrain_hf._lon_step,
            terrain_hf._lat_step,
            terrain_data=terrain_hf._terrain_data,
            building_data=building_hf._building_data,
        )

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def get_height(self, lon: float, lat: float) -> float | None:
        """返回该点的合并高度（米）.

        Args:
            lon: 经度（度）
            lat: 纬度（度）

        Returns:
            高度值（米），超出范围返回 None
        """
        row, col = self._lonlat_to_rc(lon, lat)
        if row is None or col is None:
            return None
        return float(self._data[row, col])

    def get_terrain_height(self, lon: float, lat: float) -> float | None:
        """返回该点地形高度（米），若无地形数据返回 None."""
        if self._terrain_data is None:
            return None
        row, col = self._lonlat_to_rc(lon, lat)
        if row is None or col is None:
            return None
        return float(self._terrain_data[row, col])

    def get_building_height(self, lon: float, lat: float) -> float | None:
        """返回该点建筑高度（米），若无建筑数据返回 None."""
        if self._building_data is None:
            return None
        row, col = self._lonlat_to_rc(lon, lat)
        if row is None or col is None:
            return None
        return float(self._building_data[row, col])

    def get_height_array(self, grid_codes: List[str]) -> np.ndarray:
        """批量返回多个网格中心的高度（numpy 向量化）.

        Args:
            grid_codes: 北斗网格码列表

        Returns:
            shape (N,) 的 float32 数组，无数据处填 0.0
        """
        centers = np.array([grid_code_to_center(c) for c in grid_codes])  # (N, 2)
        lons = centers[:, 0]
        lats = centers[:, 1]

        min_lon, min_lat, max_lon, max_lat = self._bbox
        n_rows, n_cols = self._data.shape

        cols = ((lons - min_lon) / self._lon_step).astype(int)
        rows = ((lats - min_lat) / self._lat_step).astype(int)

        valid = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)

        result = np.zeros(len(grid_codes), dtype=np.float32)
        result[valid] = self._data[rows[valid], cols[valid]]
        return result

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _lonlat_to_rc(
        self, lon: float, lat: float
    ) -> Tuple[int | None, int | None]:
        """将经纬度转换为数组行列索引."""
        min_lon, min_lat, max_lon, max_lat = self._bbox
        if not (min_lon <= lon < max_lon and min_lat <= lat < max_lat):
            return None, None
        n_rows, n_cols = self._data.shape
        col = int((lon - min_lon) / self._lon_step)
        row = int((lat - min_lat) / self._lat_step)
        col = min(col, n_cols - 1)
        row = min(row, n_rows - 1)
        return row, col


def _grid_steps(level: int) -> Tuple[float, float]:
    """返回给定北斗层级的 (lon_step, lat_step) 单位：度."""
    lon_step = _L2_LON_STEP
    lat_step = _L2_LAT_STEP
    for _ in range(level - 2):
        lon_step /= _SUBDIVISIONS
        lat_step /= _SUBDIVISIONS
    return lon_step, lat_step
