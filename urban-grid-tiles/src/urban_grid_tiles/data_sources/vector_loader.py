"""shp/geojson 建筑矢量数据读取."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from urban_grid_tiles.data_sources.base import DataSource


class VectorLoader(DataSource):
    """从 Shapefile 或 GeoJSON 文件读取建筑矢量数据并栅格化为高度场.

    Args:
        vector_path: 矢量文件路径（.shp / .geojson / .json）
        height_attr: 建筑高度属性字段名（默认 "height"）
        default_height: 当属性字段缺失时使用的默认高度（米）
    """

    def __init__(
        self,
        vector_path: str | Path,
        height_attr: str = "height",
        default_height: float = 10.0,
    ) -> None:
        self.vector_path = Path(vector_path)
        self.height_attr = height_attr
        self.default_height = default_height
        self._gdf = None  # geopandas.GeoDataFrame
        self._bbox: Tuple[float, float, float, float] | None = None

    # ------------------------------------------------------------------
    # DataSource 接口
    # ------------------------------------------------------------------

    def load(self) -> None:
        """读取矢量文件到内存（geopandas GeoDataFrame）."""
        import geopandas as gpd

        self._gdf = gpd.read_file(self.vector_path)

        # 确保为 WGS84
        if self._gdf.crs is not None and self._gdf.crs.to_epsg() != 4326:
            self._gdf = self._gdf.to_crs(epsg=4326)

        bounds = self._gdf.total_bounds  # (minx, miny, maxx, maxy)
        self._bbox = (
            float(bounds[0]),
            float(bounds[1]),
            float(bounds[2]),
            float(bounds[3]),
        )

    def get_height(self, lon: float, lat: float) -> float | None:
        """返回给定经纬度处最高建筑物的高度（米）.

        使用 shapely 空间查询，返回覆盖该点的所有建筑物中最高值。

        Args:
            lon: 经度（度）
            lat: 纬度（度）

        Returns:
            建筑高度（米），若该点无建筑则返回 None
        """
        if self._gdf is None:
            raise RuntimeError("请先调用 load() 加载数据")

        from shapely.geometry import Point

        point = Point(lon, lat)
        hits = self._gdf[self._gdf.geometry.contains(point)]

        if hits.empty:
            return None

        if self.height_attr in hits.columns:
            heights = hits[self.height_attr].fillna(self.default_height).astype(float)
        else:
            heights = [self.default_height] * len(hits)

        return float(max(heights))

    def rasterize(
        self,
        bbox: Tuple[float, float, float, float],
        lon_step: float,
        lat_step: float,
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """将建筑矢量栅格化为二维高度数组.

        Args:
            bbox: 目标范围 (min_lon, min_lat, max_lon, max_lat)
            lon_step: 栅格经度步长（度）
            lat_step: 栅格纬度步长（度）

        Returns:
            (height_array, actual_bbox)
            - height_array: shape (n_rows, n_cols) 的 float32 数组
            - actual_bbox: 实际栅格范围（对齐到步长的网格）
        """
        if self._gdf is None:
            raise RuntimeError("请先调用 load() 加载数据")

        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.features import rasterize as rio_rasterize

        min_lon, min_lat, max_lon, max_lat = bbox
        n_cols = max(1, int(round((max_lon - min_lon) / lon_step)))
        n_rows = max(1, int(round((max_lat - min_lat) / lat_step)))

        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, n_cols, n_rows)

        # 准备 (geometry, value) 对
        shapes = []
        for _, row in self._gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            h = float(row[self.height_attr]) if self.height_attr in self._gdf.columns else self.default_height
            if h is None or np.isnan(h):
                h = self.default_height
            shapes.append((geom.__geo_interface__, h))

        if shapes:
            arr = rio_rasterize(
                shapes,
                out_shape=(n_rows, n_cols),
                transform=transform,
                fill=0.0,
                dtype=np.float32,
                merge_alg=rasterio.enums.MergeAlg.replace,
            )
        else:
            arr = np.zeros((n_rows, n_cols), dtype=np.float32)

        return arr, bbox

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """返回数据覆盖包围盒 (min_lon, min_lat, max_lon, max_lat)."""
        if self._bbox is None:
            raise RuntimeError("请先调用 load() 加载数据")
        return self._bbox

    @property
    def geodataframe(self):
        """返回原始 GeoDataFrame（只读）."""
        if self._gdf is None:
            raise RuntimeError("请先调用 load() 加载数据")
        return self._gdf
