#!/usr/bin/env python3
"""遮蔽分析 CLI 入口.

用法示例：
    python run_los_analysis.py \\
        --radar-lon 116.3 --radar-lat 39.9 --radar-alt 100 \\
        --max-range 5000 --level 7 \\
        --output-dir ./output/los_result
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import yaml


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="配置文件路径（YAML）",
)
@click.option("--radar-lon", type=float, required=True, help="雷达经度（度）")
@click.option("--radar-lat", type=float, required=True, help="雷达纬度（度）")
@click.option("--radar-alt", type=float, default=50.0, show_default=True, help="雷达高度（米）")
@click.option("--max-range", type=float, default=5000.0, show_default=True, help="最大探测距离（米）")
@click.option("--level", type=int, default=7, show_default=True, help="北斗网格层级")
@click.option("--terrain-tif", default="", help="GeoTIFF 地形 DEM 路径")
@click.option("--building-vector", default="", help="建筑矢量文件路径（shp/geojson）")
@click.option(
    "--output-dir",
    "-o",
    default="./output/los_result",
    show_default=True,
    help="输出目录",
)
@click.option("--tileset-dir", default="./output/tiles", help="全量 tileset 目录（用于生成子集 tileset）")
@click.option("--verbose", "-v", is_flag=True, help="输出详细日志")
def main(
    config: Path | None,
    radar_lon: float,
    radar_lat: float,
    radar_alt: float,
    max_range: float,
    level: int,
    terrain_tif: str,
    building_vector: str,
    output_dir: str,
    tileset_dir: str,
    verbose: bool,
) -> None:
    """执行 LOS 遮蔽分析并生成子集 tileset.json."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger(__name__)

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from urban_grid_tiles.data_sources.height_field import HeightField
    from urban_grid_tiles.analysis.los_analysis import compute_los
    from urban_grid_tiles.tiles.tileset_builder import build_sub_tileset
    from urban_grid_tiles.tiles.tile_writer import TileWriter

    # 读取外部配置（可选）
    bbox = None
    if config is not None:
        with open(config, encoding="utf-8") as f:
            cfg: dict = yaml.safe_load(f)
        bbox_cfg = cfg.get("bbox", {})
        bbox = (
            float(bbox_cfg.get("min_lon", radar_lon - 0.1)),
            float(bbox_cfg.get("min_lat", radar_lat - 0.1)),
            float(bbox_cfg.get("max_lon", radar_lon + 0.1)),
            float(bbox_cfg.get("max_lat", radar_lat + 0.1)),
        )
        terrain_tif = terrain_tif or cfg.get("data_sources", {}).get("terrain_tif", "")
        building_vector = building_vector or cfg.get("data_sources", {}).get("building_vector", "")

    if bbox is None:
        # 以雷达为中心，取 max_range 对应范围作为 bbox（粗估）
        import math
        R = 6_371_000.0
        deg = math.degrees(max_range / R)
        bbox = (
            radar_lon - deg,
            radar_lat - deg,
            radar_lon + deg,
            radar_lat + deg,
        )

    # 构建高度场
    log.info("构建高度场...")
    if terrain_tif:
        hf = HeightField.from_tif(terrain_tif, level, bbox)
    else:
        import numpy as np
        from urban_grid_tiles.data_sources.height_field import _grid_steps

        lon_step, lat_step = _grid_steps(level)
        min_lon, min_lat, max_lon, max_lat = bbox
        n_cols = max(1, int(round((max_lon - min_lon) / lon_step)))
        n_rows = max(1, int(round((max_lat - min_lat) / lat_step)))
        hf = HeightField(
            np.zeros((n_rows, n_cols), dtype=np.float32),
            bbox,
            lon_step,
            lat_step,
        )

    if building_vector:
        bld_hf = HeightField.from_vector(building_vector, level, bbox)
        from urban_grid_tiles.data_sources.height_field import HeightField as HF
        hf = HF.merge(hf, bld_hf)

    # 执行 LOS 分析
    log.info(f"执行 LOS 分析（雷达: {radar_lon:.4f},{radar_lat:.4f} 高度: {radar_alt}m，距离: {max_range}m）...")
    result = compute_los(
        radar_lon=radar_lon,
        radar_lat=radar_lat,
        radar_alt=radar_alt,
        height_field=hf,
        max_range_m=max_range,
        level=level,
    )

    visible = result["visible"]
    occluded = result["occluded"]
    log.info(f"可见网格: {len(visible)}，遮挡网格: {len(occluded)}")

    # 加载 tile_index（若存在）
    tile_index: dict = {}
    ts_path = Path(tileset_dir) / "tileset.json"
    if ts_path.exists():
        import json
        full_ts = json.loads(ts_path.read_text(encoding="utf-8"))
        # 从 tileset.json 重建 tile_index（简单扫描 content.uri）
        tile_index = _extract_tile_index(full_ts)

    # 生成子集 tileset
    sub_tileset = build_sub_tileset(
        hit_grid_codes=visible,
        full_tile_index=tile_index,
        levels=[level],
    )

    # 写出结果
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub_ts_path = out_dir / "tileset.json"
    sub_ts_path.write_text(
        json.dumps(sub_tileset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(f"子集 tileset.json 写出至: {sub_ts_path}")

    # 同时保存分析结果 JSON
    summary_path = out_dir / "los_result.json"
    summary_path.write_text(
        json.dumps(
            {
                "radar": {"lon": radar_lon, "lat": radar_lat, "alt": radar_alt},
                "max_range_m": max_range,
                "level": level,
                "visible_count": len(visible),
                "occluded_count": len(occluded),
                "visible": sorted(visible),
                "occluded": sorted(occluded),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(f"分析结果写出至: {summary_path}")
    click.echo(f"分析完成，输出目录: {out_dir}")


def _extract_tile_index(tileset: dict, index: dict | None = None) -> dict:
    """递归从 tileset dict 中提取 {grid_code: uri} 索引（简化实现）."""
    if index is None:
        index = {}
    root = tileset.get("root", {})
    _walk_node(root, index)
    return index


def _walk_node(node: dict, index: dict) -> None:
    content = node.get("content")
    if content and "uri" in content:
        uri = content["uri"]
        # 尝试从 URI 中提取 grid_code（文件名去扩展名）
        stem = Path(uri).stem
        index[stem] = uri
    for child in node.get("children", []):
        _walk_node(child, index)


if __name__ == "__main__":
    main()
