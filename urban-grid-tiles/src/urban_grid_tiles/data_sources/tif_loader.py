"""GeoTIFF 地形数据读取."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from urban_grid_tiles.data_sources.base import DataSource


class TifLoader(DataSource):
    """从 GeoTIFF 文件读取地形高程数据.

    Args:
        tif_path: GeoTIFF 文件路径
    """

    def __init__(self, tif_path: str | Path) -> None:
        self.tif_path = Path(tif_path)
        self._data: np.ndarray | None = None
        self._transform = None  # affine.Affine
        self._nodata: float | None = None
        self._bbox: Tuple[float, float, float, float] | None = None

    # ------------------------------------------------------------------
    # DataSource 接口
    # ------------------------------------------------------------------

    def load(self) -> None:
        """打开并读取 GeoTIFF 高程数据到内存."""
        import rasterio
        from rasterio.crs import CRS
        from rasterio.warp import reproject, Resampling, calculate_default_transform

        with rasterio.open(self.tif_path) as src:
            target_crs = CRS.from_epsg(4326)
            if src.crs == target_crs:
                self._data = src.read(1).astype(np.float32)
                self._transform = src.transform
                self._nodata = src.nodata
                bounds = src.bounds
            else:
                # 重投影到 WGS84
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds
                )
                data = np.empty((height, width), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )
                self._data = data
                self._transform = transform
                self._nodata = src.nodata
                from rasterio.transform import array_bounds
                bounds = array_bounds(height, width, transform)

            self._bbox = (bounds.left, bounds.bottom, bounds.right, bounds.top)

    def get_height(self, lon: float, lat: float) -> float | None:
        """返回给定经纬度处的插值高程（米）.

        Args:
            lon: 经度（度）
            lat: 纬度（度）

        Returns:
            高程值，若在范围外或为 nodata 则返回 None
        """
        if self._data is None:
            raise RuntimeError("请先调用 load() 加载数据")

        from affine import Affine

        transform: Affine = self._transform  # type: ignore[assignment]
        # 将经纬度转换为像素坐标
        col_f = (lon - transform.c) / transform.a
        row_f = (lat - transform.f) / transform.e

        row = int(round(row_f))
        col = int(round(col_f))

        rows, cols = self._data.shape
        if not (0 <= row < rows and 0 <= col < cols):
            return None

        value = float(self._data[row, col])
        if self._nodata is not None and value == self._nodata:
            return None
        return value

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """返回数据覆盖包围盒 (min_lon, min_lat, max_lon, max_lat)."""
        if self._bbox is None:
            raise RuntimeError("请先调用 load() 加载数据")
        return self._bbox

    @property
    def data(self) -> np.ndarray:
        """返回原始高程数组（只读）."""
        if self._data is None:
            raise RuntimeError("请先调用 load() 加载数据")
        return self._data
