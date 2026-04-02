"""tileset.json 构建测试（使用合成数据）."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Set

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from urban_grid_tiles.tiles.tileset_builder import (
    _bbox_to_region,
    _geometric_error_for_level,
    build_full_tileset,
    build_sub_tileset,
)
from urban_grid_tiles.grid.beidou_grid import (
    get_children_codes,
    lonlat_to_grid_code,
    get_level,
)


BBOX = (116.0, 39.7, 116.7, 40.1)
LEVELS = [4, 5, 6]


def make_tile_index(bbox: tuple, level: int) -> Dict[str, str]:
    """构造一个覆盖 bbox 的 tile_index."""
    from urban_grid_tiles.grid.beidou_grid import (
        _L2_LON_STEP,
        _L2_LAT_STEP,
        _SUBDIVISIONS,
    )

    lon_step = _L2_LON_STEP
    lat_step = _L2_LAT_STEP
    for _ in range(level - 2):
        lon_step /= _SUBDIVISIONS
        lat_step /= _SUBDIVISIONS

    min_lon, min_lat, max_lon, max_lat = bbox
    index: Dict[str, str] = {}
    seen: Set[str] = set()
    lat = min_lat + lat_step / 2
    while lat < max_lat:
        lon = min_lon + lon_step / 2
        while lon < max_lon:
            code = lonlat_to_grid_code(lon, lat, level)
            if code not in seen:
                seen.add(code)
                index[code] = f"tiles/{code[:4]}/{code}.glb"
            lon += lon_step
        lat += lat_step
    return index


class TestBboxToRegion:
    def test_in_radians(self):
        bbox = (116.0, 39.7, 116.7, 40.1)
        region = _bbox_to_region(bbox)
        assert region[0] == pytest.approx(math.radians(116.0), abs=1e-9)
        assert region[1] == pytest.approx(math.radians(39.7), abs=1e-9)
        assert region[2] == pytest.approx(math.radians(116.7), abs=1e-9)
        assert region[3] == pytest.approx(math.radians(40.1), abs=1e-9)

    def test_default_altitude(self):
        region = _bbox_to_region((0, 0, 1, 1))
        assert region[4] == 0.0   # minHeight
        assert region[5] == 500.0  # maxHeight

    def test_custom_altitude(self):
        region = _bbox_to_region((0, 0, 1, 1), min_alt=10.0, max_alt=200.0)
        assert region[4] == 10.0
        assert region[5] == 200.0


class TestGeometricError:
    def test_decreases_with_level(self):
        errors = [_geometric_error_for_level(lv) for lv in range(2, 9)]
        for i in range(len(errors) - 1):
            assert errors[i] > errors[i + 1]

    def test_minimum_not_zero(self):
        for lv in range(1, 11):
            assert _geometric_error_for_level(lv) > 0


class TestBuildFullTileset:
    def test_structure(self):
        tile_index = make_tile_index(BBOX, LEVELS[-1])
        ts = build_full_tileset(BBOX, LEVELS, tile_index)

        assert "asset" in ts
        assert ts["asset"]["version"] == "1.1"
        assert "geometricError" in ts
        assert "root" in ts

        root = ts["root"]
        assert "boundingVolume" in root
        assert "region" in root["boundingVolume"]
        assert len(root["boundingVolume"]["region"]) == 6
        assert "geometricError" in root
        assert "refine" in root
        assert root["refine"] == "REPLACE"

    def test_root_region_in_radians(self):
        tile_index = make_tile_index(BBOX, LEVELS[-1])
        ts = build_full_tileset(BBOX, LEVELS, tile_index)
        region = ts["root"]["boundingVolume"]["region"]
        # 验证值在弧度范围（约 -π 到 π）
        assert -4 < region[0] < 4
        assert -4 < region[1] < 4

    def test_has_children(self):
        tile_index = make_tile_index(BBOX, LEVELS[-1])
        ts = build_full_tileset(BBOX, LEVELS, tile_index)
        root = ts["root"]
        assert "children" in root
        assert len(root["children"]) > 0

    def test_empty_levels_raises(self):
        with pytest.raises(ValueError):
            build_full_tileset(BBOX, [], {})

    def test_tile_content_uri_present(self):
        """tile_index 中的内容 URI 应出现在 tileset 节点中."""
        tile_index = make_tile_index(BBOX, LEVELS[-1])
        ts = build_full_tileset(BBOX, LEVELS, tile_index)

        # 收集 tileset 中所有 content.uri
        all_uris: Set[str] = set()

        def collect_uris(node: dict):
            if "content" in node:
                all_uris.add(node["content"]["uri"])
            for child in node.get("children", []):
                collect_uris(child)

        collect_uris(ts["root"])

        # tile_index 中应有部分 URI 出现在 tileset 中
        expected_uris = set(tile_index.values())
        assert len(all_uris & expected_uris) > 0, (
            "tileset 中未找到任何来自 tile_index 的 URI"
        )


class TestBuildSubTileset:
    def test_empty_input(self):
        ts = build_sub_tileset(set(), {}, LEVELS)
        assert "asset" in ts

    def test_structure(self):
        # 选取 bbox 中心的几个网格码作为命中
        lon, lat = 116.3, 39.9
        hit_codes = {lonlat_to_grid_code(lon, lat, LEVELS[-1])}
        tile_index = make_tile_index(BBOX, LEVELS[-1])

        ts = build_sub_tileset(hit_codes, tile_index, LEVELS)
        assert "asset" in ts
        assert "root" in ts
        root = ts["root"]
        assert "boundingVolume" in root

    def test_sub_tileset_contains_hit_ancestors(self):
        """子 tileset 应包含命中码的所有祖先节点."""
        lon, lat = 116.3, 39.9
        hit_level = LEVELS[-1]
        hit_code = lonlat_to_grid_code(lon, lat, hit_level)
        tile_index = make_tile_index(BBOX, hit_level)

        ts = build_sub_tileset({hit_code}, tile_index, LEVELS)

        # 递归收集所有 boundingVolume 节点
        all_nodes: list = []

        def collect_nodes(node: dict):
            all_nodes.append(node)
            for child in node.get("children", []):
                collect_nodes(child)

        collect_nodes(ts["root"])
        assert len(all_nodes) > 1, "子 tileset 应有多于1个节点（包含祖先）"

    def test_sub_tileset_refine_replace(self):
        lon, lat = 116.3, 39.9
        hit_code = lonlat_to_grid_code(lon, lat, LEVELS[-1])
        tile_index = make_tile_index(BBOX, LEVELS[-1])
        ts = build_sub_tileset({hit_code}, tile_index, LEVELS)

        def check_refine(node: dict):
            if "refine" in node:
                assert node["refine"] == "REPLACE"
            for child in node.get("children", []):
                check_refine(child)

        check_refine(ts["root"])

    def test_sub_tileset_content_uri_from_index(self):
        """命中节点的 content.uri 应来自 tile_index."""
        lon, lat = 116.3, 39.9
        hit_level = LEVELS[-1]
        hit_code = lonlat_to_grid_code(lon, lat, hit_level)
        tile_index = {hit_code: f"tiles/{hit_code[:4]}/{hit_code}.glb"}

        ts = build_sub_tileset({hit_code}, tile_index, LEVELS)

        found_uris: Set[str] = set()

        def collect(node: dict):
            if "content" in node:
                found_uris.add(node["content"]["uri"])
            for child in node.get("children", []):
                collect(child)

        collect(ts["root"])
        assert tile_index[hit_code] in found_uris
