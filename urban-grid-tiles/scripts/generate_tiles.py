#!/usr/bin/env python3
"""离线生成 3D Tiles 的 CLI 入口.

用法示例：
    python generate_tiles.py --config config/default_config.yaml
    python generate_tiles.py --bbox 116.0,39.7,116.7,40.1 --levels 5,6,7
"""

from __future__ import annotations

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
    help="配置文件路径（YAML），优先于其他参数",
)
@click.option("--bbox", default=None, help="包围盒 min_lon,min_lat,max_lon,max_lat")
@click.option(
    "--levels",
    default="4,5,6,7",
    show_default=True,
    help="北斗网格层级列表，逗号分隔",
)
@click.option("--terrain-tif", default="", help="GeoTIFF 地形 DEM 路径")
@click.option("--building-vector", default="", help="建筑矢量文件路径（shp/geojson）")
@click.option(
    "--output-dir",
    "-o",
    default="./output/tiles",
    show_default=True,
    help="输出目录",
)
@click.option(
    "--vertical-min", default=0.0, type=float, show_default=True, help="最低高程（米）"
)
@click.option(
    "--vertical-max", default=500.0, type=float, show_default=True, help="最高高程（米）"
)
@click.option(
    "--vertical-step",
    default=10.0,
    type=float,
    show_default=True,
    help="垂直步长（米）",
)
@click.option("--verbose", "-v", is_flag=True, help="输出详细日志")
def main(
    config: Path | None,
    bbox: str | None,
    levels: str,
    terrain_tif: str,
    building_vector: str,
    output_dir: str,
    vertical_min: float,
    vertical_max: float,
    vertical_step: float,
    verbose: bool,
) -> None:
    """离线生成北斗网格体素 3D Tiles."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 加载配置
    if config is not None:
        with open(config, encoding="utf-8") as f:
            cfg: dict = yaml.safe_load(f)
    else:
        # 从命令行参数构建配置
        if bbox is None:
            click.echo("错误：必须提供 --bbox 或 --config 参数", err=True)
            sys.exit(1)
        min_lon, min_lat, max_lon, max_lat = [float(x) for x in bbox.split(",")]
        cfg = {
            "bbox": {
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
            },
            "tile_levels": [int(l) for l in levels.split(",")],
            "analysis_level": max(int(l) for l in levels.split(",")),
            "vertical": {
                "min_alt": vertical_min,
                "max_alt": vertical_max,
                "step": vertical_step,
            },
            "data_sources": {
                "terrain_tif": terrain_tif,
                "building_vector": building_vector,
                "building_height_attr": "height",
            },
            "output": {
                "dir": output_dir,
                "format": "glb",
            },
            "performance": {
                "tile_batch_size": 100,
                "workers": 4,
            },
        }

    # 执行流程
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from urban_grid_tiles.pipeline import run_pipeline

    out = run_pipeline(cfg)
    click.echo(f"生成完成，输出目录: {out}")


if __name__ == "__main__":
    main()
