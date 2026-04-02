"""北斗网格编码单元测试."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# 确保 src 在路径中
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from urban_grid_tiles.grid.beidou_grid import (
    _LEVEL_DIGITS,
    get_children_codes,
    get_grid_size_meters,
    get_level,
    get_parent_code,
    grid_code_to_bbox,
    grid_code_to_center,
    lonlat_to_grid_code,
)


class TestLevelDigits:
    """验证各层级位数符合 GB/T 40087 标准."""

    def test_level_digits_mapping(self):
        expected = {
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
        assert _LEVEL_DIGITS == expected

    def test_each_level_increases_by_3(self):
        digits = sorted(_LEVEL_DIGITS.items())
        for i in range(1, len(digits)):
            level_prev, d_prev = digits[i - 1]
            level_cur, d_cur = digits[i]
            assert d_cur - d_prev == 3, (
                f"L{level_prev}→L{level_cur} 位数差应为3，实际为 {d_cur - d_prev}"
            )


class TestLonlatToGridCode:
    """lonlat_to_grid_code() 编码行为测试."""

    def test_returns_correct_digit_count(self):
        lon, lat = 116.3, 39.9  # 北京
        for level in range(1, 11):
            code = lonlat_to_grid_code(lon, lat, level)
            assert len(code) == _LEVEL_DIGITS[level], (
                f"L{level} 码应有 {_LEVEL_DIGITS[level]} 位，实际 {len(code)} 位"
            )

    def test_parent_is_prefix_of_child(self):
        lon, lat = 116.3, 39.9
        for level in range(3, 9):
            parent_code = lonlat_to_grid_code(lon, lat, level - 1)
            child_code = lonlat_to_grid_code(lon, lat, level)
            assert child_code.startswith(parent_code), (
                f"L{level} 码 {child_code!r} 应以父码 {parent_code!r} 为前缀"
            )

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            lonlat_to_grid_code(116.0, 39.0, 0)
        with pytest.raises(ValueError):
            lonlat_to_grid_code(116.0, 39.0, 11)

    def test_different_locations_give_different_codes(self):
        """不同经纬度在细粒度层级下应产生不同的网格码."""
        code_a = lonlat_to_grid_code(116.3, 39.9, 7)
        code_b = lonlat_to_grid_code(116.31, 39.91, 7)
        # 两点相距 > 1km，在 L7（~7.7m）下必然不同
        assert code_a != code_b

    def test_boundary_longitude(self):
        """测试经度边界值不抛异常."""
        code = lonlat_to_grid_code(179.9, 0.0, 4)
        assert len(code) == _LEVEL_DIGITS[4]


class TestGridCodeToBbox:
    """grid_code_to_bbox() 解码行为测试."""

    def test_l2_bbox_size(self):
        """L2 格子应该是 1°×1° 的方块."""
        code = lonlat_to_grid_code(116.3, 39.9, 2)
        min_lon, min_lat, max_lon, max_lat = grid_code_to_bbox(code)
        assert abs((max_lon - min_lon) - 1.0) < 1e-6
        assert abs((max_lat - min_lat) - 1.0) < 1e-6

    def test_point_inside_bbox(self):
        """编码点应落在其对应网格的包围盒内."""
        lon, lat = 116.3, 39.9
        for level in range(2, 9):
            code = lonlat_to_grid_code(lon, lat, level)
            min_lon, min_lat, max_lon, max_lat = grid_code_to_bbox(code)
            assert min_lon <= lon < max_lon, f"L{level} 经度不在包围盒内"
            assert min_lat <= lat < max_lat, f"L{level} 纬度不在包围盒内"

    def test_child_bbox_inside_parent_bbox(self):
        """子格子包围盒应完全包含在父格子包围盒内."""
        lon, lat = 116.3, 39.9
        for level in range(3, 8):
            parent_code = lonlat_to_grid_code(lon, lat, level - 1)
            child_code = lonlat_to_grid_code(lon, lat, level)
            p_bbox = grid_code_to_bbox(parent_code)
            c_bbox = grid_code_to_bbox(child_code)
            assert c_bbox[0] >= p_bbox[0]
            assert c_bbox[1] >= p_bbox[1]
            assert c_bbox[2] <= p_bbox[2]
            assert c_bbox[3] <= p_bbox[3]

    def test_invalid_digit_count_raises(self):
        with pytest.raises(ValueError):
            grid_code_to_bbox("1234567")  # 7 位不合法
        with pytest.raises(ValueError):
            grid_code_to_bbox("123456789")  # 9 位不合法


class TestGridCodeToCenter:
    """grid_code_to_center() 测试."""

    def test_center_inside_bbox(self):
        lon, lat = 116.3, 39.9
        code = lonlat_to_grid_code(lon, lat, 5)
        center_lon, center_lat = grid_code_to_center(code)
        min_lon, min_lat, max_lon, max_lat = grid_code_to_bbox(code)
        assert min_lon < center_lon < max_lon
        assert min_lat < center_lat < max_lat


class TestGetLevel:
    """get_level() 测试."""

    def test_correct_level_from_digits(self):
        lon, lat = 116.3, 39.9
        for expected_level in range(1, 11):
            code = lonlat_to_grid_code(lon, lat, expected_level)
            assert get_level(code) == expected_level

    def test_invalid_code_raises(self):
        with pytest.raises(ValueError):
            get_level("1234567")  # 7位不合法


class TestGetParentCode:
    """get_parent_code() 测试."""

    def test_parent_is_correct_level(self):
        lon, lat = 116.3, 39.9
        code = lonlat_to_grid_code(lon, lat, 7)
        parent = get_parent_code(code)
        assert get_level(parent) == 6
        assert len(parent) == _LEVEL_DIGITS[6]

    def test_parent_is_prefix(self):
        lon, lat = 116.3, 39.9
        code = lonlat_to_grid_code(lon, lat, 8)
        parent = get_parent_code(code)
        assert code.startswith(parent)

    def test_l1_has_no_parent(self):
        code = lonlat_to_grid_code(116.3, 39.9, 1)
        with pytest.raises(ValueError):
            get_parent_code(code)


class TestGetChildrenCodes:
    """get_children_codes() 测试."""

    def test_returns_4_children(self):
        code = lonlat_to_grid_code(116.3, 39.9, 4)
        children = get_children_codes(code)
        assert len(children) == 4

    def test_children_are_next_level(self):
        code = lonlat_to_grid_code(116.3, 39.9, 5)
        children = get_children_codes(code)
        for child in children:
            assert get_level(child) == 6
            assert child.startswith(code)

    def test_children_bboxes_tile_parent(self):
        """4 个子格子的包围盒应无缝覆盖父格子."""
        code = lonlat_to_grid_code(116.3, 39.9, 5)
        parent_bbox = grid_code_to_bbox(code)
        children = get_children_codes(code)
        child_bboxes = [grid_code_to_bbox(c) for c in children]

        # 覆盖范围
        c_min_lon = min(b[0] for b in child_bboxes)
        c_min_lat = min(b[1] for b in child_bboxes)
        c_max_lon = max(b[2] for b in child_bboxes)
        c_max_lat = max(b[3] for b in child_bboxes)

        assert abs(c_min_lon - parent_bbox[0]) < 1e-9
        assert abs(c_min_lat - parent_bbox[1]) < 1e-9
        assert abs(c_max_lon - parent_bbox[2]) < 1e-9
        assert abs(c_max_lat - parent_bbox[3]) < 1e-9

    def test_l10_has_no_children(self):
        code = lonlat_to_grid_code(116.3, 39.9, 10)
        with pytest.raises(ValueError):
            get_children_codes(code)


class TestGetGridSizeMeters:
    """get_grid_size_meters() 测试."""

    def test_l2_approx_111km(self):
        size = get_grid_size_meters(2, lat=30.0)
        # L2=1°×1°，赤道约 111km，纬度 30° 处约 96~111km
        assert 80_000 < size < 130_000, f"L2 尺寸应约 111km，实际 {size:.0f}m"

    def test_higher_level_smaller_size(self):
        sizes = [get_grid_size_meters(lv, lat=39.9) for lv in range(2, 9)]
        for i in range(len(sizes) - 1):
            assert sizes[i] > sizes[i + 1], (
                f"L{i+2} ({sizes[i]:.1f}m) 应大于 L{i+3} ({sizes[i+1]:.1f}m)"
            )

    def test_l7_approx_7m(self):
        size = get_grid_size_meters(7, lat=39.9)
        # 标准说 ~7.73m，允许较大误差
        assert 1 < size < 50, f"L7 尺寸应约 7m，实际 {size:.2f}m"

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            get_grid_size_meters(0)
        with pytest.raises(ValueError):
            get_grid_size_meters(11)
