"""glTF instancing 瓦片生成（EXT_mesh_gpu_instancing）.

使用 pygltflib 构建 glTF 2.0 格式的体素实例化瓦片（.glb），
几何为共享单位立方体，每个体素通过 translation/scale 实例属性定位。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


# 占用类型 → 整数编码
OCCUPANCY_INT: Dict[str, int] = {
    "air": 0,
    "terrain": 1,
    "building": 2,
}


def _unit_cube_geometry() -> Dict[str, Any]:
    """生成单位立方体的顶点/法线/索引数据.

    Returns:
        dict 包含 positions (N,3), normals (N,3), indices (M,3) numpy 数组
    """
    # 8 个顶点（中心在原点，边长 1）
    v = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    # 6 个面，每面 2 个三角形，法线朝外
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],  # -Z
            [4, 5, 6], [4, 6, 7],  # +Z
            [0, 1, 5], [0, 5, 4],  # -Y
            [2, 3, 7], [2, 7, 6],  # +Y
            [0, 4, 7], [0, 7, 3],  # -X
            [1, 2, 6], [1, 6, 5],  # +X
        ],
        dtype=np.uint16,
    )

    normals = np.zeros_like(v)
    face_normals = [
        [0, 0, -1], [0, 0, -1],
        [0, 0, 1], [0, 0, 1],
        [0, -1, 0], [0, -1, 0],
        [0, 1, 0], [0, 1, 0],
        [-1, 0, 0], [-1, 0, 0],
        [1, 0, 0], [1, 0, 0],
    ]
    for fi, face in enumerate(faces):
        for vi in face:
            normals[vi] += face_normals[fi]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = (normals / norms).astype(np.float32)

    return {"positions": v, "normals": normals, "indices": faces}


def build_instanced_gltf(
    voxels: List[Dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """生成体素实例化 glTF 瓦片文件（.glb）.

    几何为共享单位立方体，使用 EXT_mesh_gpu_instancing 扩展进行实例化。
    每个实例附带 Batch Table 属性：grid_code、layer、occupancy。

    Args:
        voxels: 体素字典列表（来自 Voxelizer.voxelize()）
        output_path: 输出 .glb 文件路径

    Returns:
        输出文件路径
    """
    import pygltflib

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(voxels)
    if n == 0:
        # 空瓦片：写出最小 glb
        gltf = pygltflib.GLTF2()
        gltf.save_binary(str(output_path))
        return output_path

    # ------------------------------------------------------------------
    # 1. 构建实例属性数组
    # ------------------------------------------------------------------
    translations = np.zeros((n, 3), dtype=np.float32)
    scales = np.zeros((n, 3), dtype=np.float32)
    layers = np.zeros(n, dtype=np.int32)
    occupancies = np.zeros(n, dtype=np.int32)

    for i, v in enumerate(voxels):
        # 将经纬度 + 高程转换为局部笛卡尔坐标（简化：以第一个体素为原点）
        # 实际生产应使用 pyproj 转换为 ECEF 或 ENU
        translations[i] = [
            v["center_lon"],
            v["center_lat"],
            v["center_alt"],
        ]
        scales[i] = [v["size_h"], v["size_h"], v["size_v"]]
        layers[i] = v["layer"]
        occupancies[i] = OCCUPANCY_INT.get(v["occupancy"], 0)

    # ------------------------------------------------------------------
    # 2. 构建单位立方体几何 bufferView / accessor
    # ------------------------------------------------------------------
    cube = _unit_cube_geometry()
    positions: np.ndarray = cube["positions"]
    normals: np.ndarray = cube["normals"]
    indices: np.ndarray = cube["indices"]

    pos_bytes = positions.tobytes()
    norm_bytes = normals.tobytes()
    idx_bytes = indices.tobytes()
    trans_bytes = translations.tobytes()
    scale_bytes = scales.tobytes()
    layer_bytes = layers.tobytes()
    occ_bytes = occupancies.tobytes()

    # 合并所有二进制数据
    all_bytes = pos_bytes + norm_bytes + idx_bytes + trans_bytes + scale_bytes + layer_bytes + occ_bytes
    buffer_data = all_bytes

    def _offset(arrays_before):
        return sum(len(b) for b in arrays_before)

    chunks = [pos_bytes, norm_bytes, idx_bytes, trans_bytes, scale_bytes, layer_bytes, occ_bytes]

    gltf = pygltflib.GLTF2()
    gltf.extensionsUsed = ["EXT_mesh_gpu_instancing"]
    gltf.extensionsRequired = ["EXT_mesh_gpu_instancing"]

    # Buffer
    buf = pygltflib.Buffer(byteLength=len(buffer_data))
    gltf.buffers.append(buf)

    # BufferViews
    bv_idx = 0
    buffer_views = []
    for chunk in chunks:
        bv = pygltflib.BufferView(
            buffer=0,
            byteOffset=_offset(chunks[:bv_idx]),
            byteLength=len(chunk),
        )
        buffer_views.append(bv)
        bv_idx += 1
    gltf.bufferViews = buffer_views

    # Accessors
    n_verts = len(positions)
    n_idx = len(indices) * 3

    pos_acc = pygltflib.Accessor(
        bufferView=0,
        componentType=pygltflib.FLOAT,
        count=n_verts,
        type=pygltflib.VEC3,
        max=positions.max(axis=0).tolist(),
        min=positions.min(axis=0).tolist(),
    )
    norm_acc = pygltflib.Accessor(
        bufferView=1,
        componentType=pygltflib.FLOAT,
        count=n_verts,
        type=pygltflib.VEC3,
    )
    idx_acc = pygltflib.Accessor(
        bufferView=2,
        componentType=pygltflib.UNSIGNED_SHORT,
        count=n_idx,
        type=pygltflib.SCALAR,
    )
    trans_acc = pygltflib.Accessor(
        bufferView=3,
        componentType=pygltflib.FLOAT,
        count=n,
        type=pygltflib.VEC3,
    )
    scale_acc = pygltflib.Accessor(
        bufferView=4,
        componentType=pygltflib.FLOAT,
        count=n,
        type=pygltflib.VEC3,
    )
    layer_acc = pygltflib.Accessor(
        bufferView=5,
        componentType=pygltflib.UNSIGNED_INT,
        count=n,
        type=pygltflib.SCALAR,
    )
    occ_acc = pygltflib.Accessor(
        bufferView=6,
        componentType=pygltflib.UNSIGNED_INT,
        count=n,
        type=pygltflib.SCALAR,
    )

    gltf.accessors = [pos_acc, norm_acc, idx_acc, trans_acc, scale_acc, layer_acc, occ_acc]

    # Mesh primitive（共享单位立方体）
    prim = pygltflib.Primitive(
        attributes=pygltflib.Attributes(POSITION=0, NORMAL=1),
        indices=2,
    )
    mesh = pygltflib.Mesh(primitives=[prim])
    gltf.meshes.append(mesh)

    # Node with EXT_mesh_gpu_instancing
    node_extensions = {
        "EXT_mesh_gpu_instancing": {
            "attributes": {
                "TRANSLATION": 3,
                "SCALE": 4,
                "_LAYER": 5,
                "_OCCUPANCY": 6,
            }
        }
    }
    node = pygltflib.Node(mesh=0, extensions=node_extensions)
    gltf.nodes.append(node)

    scene = pygltflib.Scene(nodes=[0])
    gltf.scenes.append(scene)
    gltf.scene = 0

    # 设置 binary blob
    gltf.set_binary_blob(buffer_data)

    gltf.save_binary(str(output_path))
    return output_path
