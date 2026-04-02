"""抽象基类 DataSource，定义多源数据读取接口."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple


class DataSource(ABC):
    """所有数据源的抽象基类.

    子类需实现 ``load()`` 方法，将底层数据加载到内存中供后续处理。
    """

    @abstractmethod
    def load(self) -> None:
        """加载数据源到内存."""

    @abstractmethod
    def get_height(self, lon: float, lat: float) -> float | None:
        """返回给定经纬度处的高度值（米）.

        Args:
            lon: 经度（度）
            lat: 纬度（度）

        Returns:
            高度值（米），若该点无数据则返回 None
        """

    @property
    @abstractmethod
    def bbox(self) -> Tuple[float, float, float, float]:
        """返回数据源覆盖的经纬度包围盒 (min_lon, min_lat, max_lon, max_lat)."""
