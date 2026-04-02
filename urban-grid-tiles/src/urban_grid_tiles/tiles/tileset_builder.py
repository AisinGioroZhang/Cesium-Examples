"""tileset.json 构建（全量 + 子集）.

按北斗网格层级构建四叉树结构 tileset.json，符合 3D Tiles 1.1 规范。
boundingVolume.region 使用弧度制（Cesium 标准）。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Set, Tuple

from urban_grid_tiles.grid.beidou_grid import (
    get_children_codes,
    get_level,
    get_parent_code,
    grid_code_to_bbox,
    get_grid_size_meters,
)


# 3D Tiles 版本标识
_ASSET = {"version": "1.1"}

# geometricError 基准（城市直径量级，米）
_ROOT_GEOMETRIC_ERROR = 100_000.0


def _bbox_to_region(
    bbox: Tuple[float, float, float, float],
    min_alt: float = 0.0,
    max_alt: float = 500.0,
) -> List[float]:
    """将包围盒转换为 3D Tiles region（弧度制）.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) 单位：度
        min_alt: 最低高程（米）
        max_alt: 最高高程（米）

    Returns:
        [west, south, east, north, minHeight, maxHeight]（弧度+米）
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    return [
        math.radians(min_lon),
        math.radians(min_lat),
        math.radians(max_lon),
        math.radians(max_lat),
        min_alt,
        max_alt,
    ]


def _geometric_error_for_level(level: int) -> float:
    """返回给定北斗层级的 geometricError（米）.

    根节点最大，逐级减半。

    Args:
        level: 北斗层级（1~10）

    Returns:
        geometricError（米）
    """
    # 以 L2（~111km）为根基准，逐级减半
    base_level = 2
    error = _ROOT_GEOMETRIC_ERROR
    for _ in range(level - base_level):
        error /= 2.0
    return max(error, 0.5)


def _build_node(
    grid_code: str,
    tile_index: Dict[str, str],
    max_level: int,
    min_alt: float = 0.0,
    max_alt: float = 500.0,
) -> Dict[str, Any]:
    """递归构建单个 tileset 节点.

    Args:
        grid_code: 当前节点的北斗网格码
        tile_index: {grid_code: tile_file_path} 字典
        max_level: 最深层级（叶节点层级）
        min_alt: 最低高程（米）
        max_alt: 最高高程（米）

    Returns:
        tileset 节点 dict
    """
    level = get_level(grid_code)
    bbox = grid_code_to_bbox(grid_code)
    region = _bbox_to_region(bbox, min_alt, max_alt)
    geo_error = _geometric_error_for_level(level)

    node: Dict[str, Any] = {
        "boundingVolume": {"region": region},
        "geometricError": geo_error,
        "refine": "REPLACE",
    }

    if grid_code in tile_index:
        node["content"] = {"uri": tile_index[grid_code]}

    if level < max_level:
        children = []
        for child_code in get_children_codes(grid_code):
            child_bbox = grid_code_to_bbox(child_code)
            # 只添加有内容或有子孙有内容的节点（简单：始终添加）
            children.append(
                _build_node(child_code, tile_index, max_level, min_alt, max_alt)
            )
        if children:
            node["children"] = children

    return node


def build_full_tileset(
    root_bbox: Tuple[float, float, float, float],
    levels: List[int],
    tile_index: Dict[str, str],
    min_alt: float = 0.0,
    max_alt: float = 500.0,
) -> Dict[str, Any]:
    """构建全量 tileset.json dict.

    按北斗层级构建四叉树结构，根节点覆盖 root_bbox，
    自顶向下递归生成所有层级节点。

    Args:
        root_bbox: 根节点包围盒 (min_lon, min_lat, max_lon, max_lat)
        levels: 要生成的层级列表，例如 [4, 5, 6, 7]
        tile_index: {grid_code: tile_file_path} 字典
        min_alt: 最低高程（米）
        max_alt: 最高高程（米）

    Returns:
        tileset.json 结构的 dict
    """
    if not levels:
        raise ValueError("levels 不能为空")

    levels = sorted(levels)
    root_level = levels[0]
    max_level = levels[-1]

    # 收集 root_bbox 覆盖的所有 root_level 网格
    from urban_grid_tiles.grid.beidou_grid import lonlat_to_grid_code, _L2_LON_STEP, _L2_LAT_STEP, _SUBDIVISIONS

    lon_step = _L2_LON_STEP
    lat_step = _L2_LAT_STEP
    for _ in range(root_level - 2):
        lon_step /= _SUBDIVISIONS
        lat_step /= _SUBDIVISIONS

    min_lon, min_lat, max_lon, max_lat = root_bbox
    root_codes: list = []
    seen: set = set()
    lat = min_lat + lat_step / 2.0
    while lat < max_lat:
        lon = min_lon + lon_step / 2.0
        while lon < max_lon:
            code = lonlat_to_grid_code(lon, lat, root_level)
            if code not in seen:
                seen.add(code)
                root_codes.append(code)
            lon += lon_step
        lat += lat_step

    # 构建虚拟根节点
    root_region = _bbox_to_region(root_bbox, min_alt, max_alt)
    root_geo_error = _ROOT_GEOMETRIC_ERROR

    children = [
        _build_node(code, tile_index, max_level, min_alt, max_alt)
        for code in root_codes
    ]

    tileset = {
        "asset": _ASSET,
        "geometricError": root_geo_error,
        "root": {
            "boundingVolume": {"region": root_region},
            "geometricError": root_geo_error,
            "refine": "REPLACE",
            "children": children,
        },
    }
    return tileset


def build_sub_tileset(
    hit_grid_codes: Set[str],
    full_tile_index: Dict[str, str],
    levels: List[int],
    min_alt: float = 0.0,
    max_alt: float = 500.0,
) -> Dict[str, Any]:
    """构建子集 tileset.json dict（仅包含命中链路的稀疏四叉树）.

    自底向上反推：从命中的最细层级网格码出发，
    利用 get_parent_code() 逐层向上确定所有祖先节点。

    Args:
        hit_grid_codes: 命中的最细层级网格码集合
        full_tile_index: 完整 {grid_code: tile_file_path} 字典
        levels: 层级列表（升序），例如 [4, 5, 6, 7]
        min_alt: 最低高程（米）
        max_alt: 最高高程（米）

    Returns:
        稀疏 tileset.json 结构的 dict
    """
    if not levels or not hit_grid_codes:
        return {"asset": _ASSET, "geometricError": 0.0, "root": {}}

    levels = sorted(levels)
    root_level = levels[0]
    max_level = levels[-1]

    # 收集所有祖先节点（包含自身）
    all_needed: Set[str] = set(hit_grid_codes)
    for code in list(hit_grid_codes):
        current = code
        while True:
            try:
                parent = get_parent_code(current)
            except ValueError:
                break
            if get_level(parent) < root_level:
                break
            all_needed.add(parent)
            current = parent

    # 找出 root_level 节点
    root_codes = {c for c in all_needed if get_level(c) == root_level}

    def _build_sparse_node(grid_code: str) -> Dict[str, Any]:
        level = get_level(grid_code)
        bbox = grid_code_to_bbox(grid_code)
        region = _bbox_to_region(bbox, min_alt, max_alt)
        node: Dict[str, Any] = {
            "boundingVolume": {"region": region},
            "geometricError": _geometric_error_for_level(level),
            "refine": "REPLACE",
        }
        if grid_code in full_tile_index:
            node["content"] = {"uri": full_tile_index[grid_code]}
        if level < max_level:
            child_nodes = []
            for child_code in get_children_codes(grid_code):
                if child_code in all_needed:
                    child_nodes.append(_build_sparse_node(child_code))
            if child_nodes:
                node["children"] = child_nodes
        return node

    children = [_build_sparse_node(code) for code in sorted(root_codes)]

    # 虚拟根节点的 bbox 覆盖所有命中节点
    all_bboxes = [grid_code_to_bbox(c) for c in root_codes]
    if all_bboxes:
        min_lon = min(b[0] for b in all_bboxes)
        min_lat = min(b[1] for b in all_bboxes)
        max_lon = max(b[2] for b in all_bboxes)
        max_lat = max(b[3] for b in all_bboxes)
        root_region = _bbox_to_region(
            (min_lon, min_lat, max_lon, max_lat), min_alt, max_alt
        )
    else:
        root_region = [0, 0, 0, 0, min_alt, max_alt]

    return {
        "asset": _ASSET,
        "geometricError": _ROOT_GEOMETRIC_ERROR,
        "root": {
            "boundingVolume": {"region": root_region},
            "geometricError": _ROOT_GEOMETRIC_ERROR,
            "refine": "REPLACE",
            "children": children,
        },
    }
