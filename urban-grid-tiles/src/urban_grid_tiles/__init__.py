"""urban_grid_tiles — 多源城市数据 → 北斗网格体素 3D Tiles 生成工具链."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("urban_grid_tiles")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
