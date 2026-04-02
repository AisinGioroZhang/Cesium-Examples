"""高度场构建测试（使用合成数据，不依赖真实文件）."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from urban_grid_tiles.data_sources.height_field import HeightField, _grid_steps
from urban_grid_tiles.grid.beidou_grid import lonlat_to_grid_code


# ------------------------------------------------------------------
# 辅助：构造合成高度场
# ------------------------------------------------------------------

BBOX = (116.0, 39.7, 116.7, 40.1)  # 北京五环近似范围
LEVEL = 5


def make_height_field(value: float = 50.0, level: int = LEVEL) -> HeightField:
    """构造一个填充固定高度值的高度场."""
    lon_step, lat_step = _grid_steps(level)
    min_lon, min_lat, max_lon, max_lat = BBOX
    n_cols = max(1, int(round((max_lon - min_lon) / lon_step)))
    n_rows = max(1, int(round((max_lat - min_lat) / lat_step)))
    data = np.full((n_rows, n_cols), value, dtype=np.float32)
    return HeightField(data, BBOX, lon_step, lat_step)


def make_terrain_building_fields() -> tuple:
    """构造地形（10m）与建筑（30m）高度场."""
    lon_step, lat_step = _grid_steps(LEVEL)
    min_lon, min_lat, max_lon, max_lat = BBOX
    n_cols = max(1, int(round((max_lon - min_lon) / lon_step)))
    n_rows = max(1, int(round((max_lat - min_lat) / lat_step)))

    terrain_data = np.full((n_rows, n_cols), 10.0, dtype=np.float32)
    building_data = np.zeros((n_rows, n_cols), dtype=np.float32)
    # 让右下角有建筑
    building_data[: n_rows // 2, : n_cols // 2] = 30.0

    terrain_hf = HeightField(
        terrain_data, BBOX, lon_step, lat_step, terrain_data=terrain_data.copy()
    )
    building_hf = HeightField(
        building_data, BBOX, lon_step, lat_step, building_data=building_data.copy()
    )
    return terrain_hf, building_hf


# ------------------------------------------------------------------
# 测试
# ------------------------------------------------------------------


class TestHeightFieldGetHeight:
    def test_returns_correct_value(self):
        hf = make_height_field(42.0)
        # 取 bbox 中心
        lon = (BBOX[0] + BBOX[2]) / 2
        lat = (BBOX[1] + BBOX[3]) / 2
        h = hf.get_height(lon, lat)
        assert h == pytest.approx(42.0, abs=0.1)

    def test_out_of_bbox_returns_none(self):
        hf = make_height_field(42.0)
        h = hf.get_height(200.0, 90.0)  # 明确超出范围
        assert h is None

    def test_boundary_points(self):
        hf = make_height_field(5.0)
        # 最左下角应能查到
        h = hf.get_height(BBOX[0] + 1e-6, BBOX[1] + 1e-6)
        assert h == pytest.approx(5.0, abs=0.1)


class TestHeightFieldMerge:
    def test_merge_takes_max(self):
        terrain_hf, building_hf = make_terrain_building_fields()
        merged = HeightField.merge(terrain_hf, building_hf)
        n_rows, n_cols = merged._data.shape

        # 有建筑的区域（右下角）应为 30，无建筑区域应为 10
        mid_r = n_rows // 4
        mid_c = n_cols // 4
        h_with_building = float(merged._data[mid_r, mid_c])
        assert h_with_building == pytest.approx(30.0, abs=0.1), (
            f"有建筑区域应为 30.0，实际 {h_with_building}"
        )

        h_terrain_only = float(merged._data[n_rows - 1, n_cols - 1])
        assert h_terrain_only == pytest.approx(10.0, abs=0.1), (
            f"纯地形区域应为 10.0，实际 {h_terrain_only}"
        )

    def test_merge_shape_matches_terrain(self):
        terrain_hf, building_hf = make_terrain_building_fields()
        merged = HeightField.merge(terrain_hf, building_hf)
        assert merged._data.shape == terrain_hf._data.shape

    def test_merge_different_shapes(self):
        """两个高度场形状不同时，合并不应抛出异常."""
        lon_step, lat_step = _grid_steps(LEVEL)
        min_lon, min_lat, max_lon, max_lat = BBOX
        n_cols = max(1, int(round((max_lon - min_lon) / lon_step)))
        n_rows = max(1, int(round((max_lat - min_lat) / lat_step)))

        t_data = np.full((n_rows, n_cols), 5.0, dtype=np.float32)
        b_data = np.full((n_rows // 2, n_cols // 2), 20.0, dtype=np.float32)

        t_hf = HeightField(t_data, BBOX, lon_step, lat_step)
        b_hf = HeightField(b_data, BBOX, lon_step * 2, lat_step * 2)

        merged = HeightField.merge(t_hf, b_hf)
        assert merged._data.shape == t_data.shape


class TestHeightFieldGetHeightArray:
    def test_batch_query(self):
        hf = make_height_field(99.0)
        lon, lat = (BBOX[0] + BBOX[2]) / 2, (BBOX[1] + BBOX[3]) / 2
        code = lonlat_to_grid_code(lon, lat, LEVEL)
        heights = hf.get_height_array([code] * 5)
        assert heights.shape == (5,)
        assert all(abs(h - 99.0) < 1.0 for h in heights)

    def test_out_of_range_returns_zero(self):
        hf = make_height_field(99.0)
        # 使用一个明确超出范围的网格码（北京以外）
        code = lonlat_to_grid_code(120.0, 30.0, LEVEL)
        heights = hf.get_height_array([code])
        assert heights[0] == pytest.approx(0.0)


class TestGridSteps:
    def test_l2_steps(self):
        lon_step, lat_step = _grid_steps(2)
        assert lon_step == pytest.approx(1.0)
        assert lat_step == pytest.approx(1.0)

    def test_steps_halve_each_level(self):
        prev_lon, prev_lat = _grid_steps(2)
        for lv in range(3, 9):
            lon, lat = _grid_steps(lv)
            assert lon == pytest.approx(prev_lon / 2)
            assert lat == pytest.approx(prev_lat / 2)
            prev_lon, prev_lat = lon, lat
