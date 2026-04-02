"""瓦片文件写入（输出到磁盘/对象存储）."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class TileWriter:
    """将生成的瓦片文件和 tileset.json 写出到目标目录.

    Args:
        output_dir: 输出根目录
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_tileset(
        self,
        tileset: Dict[str, Any],
        filename: str = "tileset.json",
    ) -> Path:
        """将 tileset dict 序列化为 JSON 文件.

        Args:
            tileset: tileset.json 结构 dict
            filename: 输出文件名

        Returns:
            写出的文件路径
        """
        out_path = self.output_dir / filename
        out_path.write_text(
            json.dumps(tileset, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out_path

    def write_sub_tileset(
        self,
        tileset: Dict[str, Any],
        sub_id: str,
    ) -> Path:
        """写出子集 tileset.json.

        文件存放在 ``<output_dir>/sub/<sub_id>/tileset.json``。

        Args:
            tileset: 子集 tileset dict
            sub_id: 子集唯一标识（例如任务 ID 或时间戳）

        Returns:
            写出的文件路径
        """
        sub_dir = self.output_dir / "sub" / sub_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        out_path = sub_dir / "tileset.json"
        out_path.write_text(
            json.dumps(tileset, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out_path

    def tile_path(self, grid_code: str, ext: str = "glb") -> Path:
        """返回指定网格码对应的瓦片文件路径（不创建文件）.

        目录结构：``<output_dir>/tiles/<grid_code[0:4]>/<grid_code>.{ext}``

        Args:
            grid_code: 北斗网格码
            ext: 文件扩展名（glb 或 gltf）

        Returns:
            目标路径（Path 对象）
        """
        prefix = grid_code[:4] if len(grid_code) >= 4 else grid_code
        tile_dir = self.output_dir / "tiles" / prefix
        tile_dir.mkdir(parents=True, exist_ok=True)
        return tile_dir / f"{grid_code}.{ext}"

    def build_tile_index(
        self,
        grid_codes: List[str],
        ext: str = "glb",
        relative_to: Optional[Path] = None,
    ) -> Dict[str, str]:
        """为一批网格码构建 tile_index（{grid_code: 相对URI}）.

        Args:
            grid_codes: 网格码列表
            ext: 瓦片文件扩展名
            relative_to: 相对路径基准目录（默认使用 output_dir）

        Returns:
            {grid_code: relative_uri} 字典
        """
        base = relative_to or self.output_dir
        index: Dict[str, str] = {}
        for code in grid_codes:
            path = self.tile_path(code, ext)
            try:
                rel = path.relative_to(base)
                index[code] = str(rel).replace("\\", "/")
            except ValueError:
                index[code] = str(path).replace("\\", "/")
        return index
