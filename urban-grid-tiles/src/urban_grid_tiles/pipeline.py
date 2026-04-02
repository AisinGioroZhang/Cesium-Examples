"""主流程编排：数据读取 → 体素化 → 生成 → 写出."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def run_pipeline(config: Dict[str, Any]) -> Path:
    """串联所有步骤，完整执行端到端生成流程.

    流程：
        多源数据输入 → 统一高度场 → 北斗网格体素化 → 生成 3D Tiles → 写出

    Args:
        config: 配置字典（与 default_config.yaml 结构一致）

    Returns:
        输出目录路径
    """
    from urban_grid_tiles.data_sources.height_field import HeightField
    from urban_grid_tiles.grid.voxelizer import Voxelizer
    from urban_grid_tiles.tiles.gltf_builder import build_instanced_gltf
    from urban_grid_tiles.tiles.tileset_builder import build_full_tileset
    from urban_grid_tiles.tiles.tile_writer import TileWriter

    # ------------------------------------------------------------------
    # 1. 读取配置
    # ------------------------------------------------------------------
    bbox_cfg = config["bbox"]
    bbox = (
        float(bbox_cfg["min_lon"]),
        float(bbox_cfg["min_lat"]),
        float(bbox_cfg["max_lon"]),
        float(bbox_cfg["max_lat"]),
    )
    tile_levels: List[int] = [int(l) for l in config["tile_levels"]]
    analysis_level: int = int(config.get("analysis_level", tile_levels[-1]))

    vert_cfg = config.get("vertical", {})
    vertical_min = float(vert_cfg.get("min_alt", 0.0))
    vertical_max = float(vert_cfg.get("max_alt", 500.0))
    vertical_step = float(vert_cfg.get("step", 10.0))

    ds_cfg = config.get("data_sources", {})
    terrain_tif: str = ds_cfg.get("terrain_tif", "")
    building_vector: str = ds_cfg.get("building_vector", "")
    building_height_attr: str = ds_cfg.get("building_height_attr", "height")

    out_cfg = config.get("output", {})
    output_dir = Path(out_cfg.get("dir", "./output/tiles"))
    tile_format: str = out_cfg.get("format", "glb")

    perf_cfg = config.get("performance", {})
    tile_batch_size: int = int(perf_cfg.get("tile_batch_size", 100))

    # ------------------------------------------------------------------
    # 2. 构建统一高度场
    # ------------------------------------------------------------------
    logger.info("构建统一高度场...")

    height_field: Optional[Any] = None

    if terrain_tif:
        logger.info(f"  加载地形 GeoTIFF: {terrain_tif}")
        terrain_hf = HeightField.from_tif(terrain_tif, analysis_level, bbox)
        height_field = terrain_hf
    else:
        logger.info("  未指定地形 GeoTIFF，使用零高度场")
        terrain_hf = _make_zero_height_field(bbox, analysis_level)
        height_field = terrain_hf

    if building_vector:
        logger.info(f"  加载建筑矢量: {building_vector}")
        building_hf = HeightField.from_vector(
            building_vector, analysis_level, bbox, building_height_attr
        )
        height_field = HeightField.merge(terrain_hf, building_hf)
    else:
        logger.info("  未指定建筑矢量，仅使用地形高度场")

    # ------------------------------------------------------------------
    # 3. 体素化
    # ------------------------------------------------------------------
    logger.info(f"开始体素化（层级={analysis_level}）...")
    voxelizer = Voxelizer(
        height_field=height_field,
        vertical_min=vertical_min,
        vertical_max=vertical_max,
        vertical_step=vertical_step,
    )
    all_voxels = voxelizer.voxelize(bbox, analysis_level)
    logger.info(f"  共生成 {len(all_voxels)} 个体素")

    # ------------------------------------------------------------------
    # 4. 分 tile 生成 glTF 文件
    # ------------------------------------------------------------------
    logger.info("生成 glTF 瓦片文件...")
    writer = TileWriter(output_dir)

    # 按顶层网格码分组
    from urban_grid_tiles.grid.beidou_grid import lonlat_to_grid_code, get_parent_code, get_level

    root_level = tile_levels[0]
    tile_groups: Dict[str, List[Dict[str, Any]]] = {}
    for v in all_voxels:
        # 找到该体素属于哪个 root_level 的瓦片
        code = v["grid_code"]
        cur = code
        while get_level(cur) > root_level:
            cur = get_parent_code(cur)
        tile_groups.setdefault(cur, []).append(v)

    tile_index: Dict[str, str] = {}
    codes = list(tile_groups.keys())
    for i in range(0, len(codes), tile_batch_size):
        batch = codes[i : i + tile_batch_size]
        for code in batch:
            voxels_for_tile = tile_groups[code]
            tile_path = writer.tile_path(code, tile_format)
            build_instanced_gltf(voxels_for_tile, tile_path)
            try:
                rel = tile_path.relative_to(output_dir)
                tile_index[code] = str(rel).replace("\\", "/")
            except ValueError:
                tile_index[code] = str(tile_path).replace("\\", "/")
        logger.info(f"  已处理 {min(i + tile_batch_size, len(codes))}/{len(codes)} 瓦片")

    # ------------------------------------------------------------------
    # 5. 构建全量 tileset.json
    # ------------------------------------------------------------------
    logger.info("构建 tileset.json...")
    tileset = build_full_tileset(
        root_bbox=bbox,
        levels=tile_levels,
        tile_index=tile_index,
        min_alt=vertical_min,
        max_alt=vertical_max,
    )

    # ------------------------------------------------------------------
    # 6. 写出
    # ------------------------------------------------------------------
    ts_path = writer.write_tileset(tileset)
    logger.info(f"tileset.json 写出至: {ts_path}")
    logger.info(f"输出目录: {output_dir}")

    return output_dir


def _make_zero_height_field(
    bbox: tuple,
    level: int,
) -> Any:
    """构造一个全零的高度场（无数据时的默认值）."""
    from urban_grid_tiles.data_sources.height_field import HeightField, _grid_steps

    lon_step, lat_step = _grid_steps(level)
    min_lon, min_lat, max_lon, max_lat = bbox
    n_cols = max(1, int(round((max_lon - min_lon) / lon_step)))
    n_rows = max(1, int(round((max_lat - min_lat) / lat_step)))
    data = np.zeros((n_rows, n_cols), dtype=np.float32)
    return HeightField(data, bbox, lon_step, lat_step)
