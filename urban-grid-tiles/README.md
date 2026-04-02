# urban-grid-tiles

## 项目简介

**urban-grid-tiles** 是一个 Python 工具链，用于将多源城市数据（地形 GeoTIFF、建筑 shp/geojson、白膜 3D Tiles、倾斜摄影 osgb）统一处理，生成基于**北斗网格编码（GB/T 40087）**的多层级三维体素 3D Tiles 数据底座，用于**城市近地空间可视化**。

核心思路：
- **离线生成全量网格 3D Tiles**（instancing，规则正方体体素）
- **在线 LOS 遮蔽分析** → 返回命中网格的子集 `tileset.json`（前端只加载可见/遮挡网格）
- 前端（CesiumJS 3D Tiles 1.1）按层级 LOD 加载，支持交互式样式过滤

---

## 技术方案概述

```
多源数据输入
    ├── GeoTIFF DEM（地形高程）
    ├── shp/geojson（建筑矢量，带高度属性）
    ├── 3D Tiles（白模/精模）
    └── Cesium 地形切片

         ↓
    统一高度场（HeightField）
    北斗网格对齐，numpy 数组存储

         ↓
    三维体素化（Voxelizer）
    水平：北斗网格码（L4~L10）
    垂直：固定步长分层（默认 10m/层）
    每体素：grid_code / layer / occupancy

         ↓
    glTF 实例化瓦片生成（EXT_mesh_gpu_instancing）
    共享单位立方体 + 每实例 translation/scale
    输出 .glb 文件

         ↓
    tileset.json 构建（3D Tiles 1.1）
    四叉树结构，按北斗层级递减 geometricError
    boundingVolume.region（弧度制）

         ↓
    输出目录
    tileset.json + tiles/**/*.glb

         ↓（在线分析）
    LOS 射线步进分析（numpy 向量化）
    雷达位置 + 高度场 → 可见/遮挡网格码集合
    → 子集 tileset.json（仅命中节点）
```

```
架构图
┌─────────────────────────────────────────────────────┐
│                   urban-grid-tiles                  │
├─────────────┬───────────────┬───────────────────────┤
│ data_sources│     grid      │        tiles          │
│  TifLoader  │ BeidouGrid    │  GltfBuilder          │
│  VectorLoad │ Voxelizer     │  TilesetBuilder       │
│  HeightField│               │  TileWriter           │
├─────────────┴───────────────┴───────────────────────┤
│                  analysis / pipeline                │
│  LosAnalysis              Pipeline                  │
└─────────────────────────────────────────────────────┘
           ↓
    CesiumJS 3D Tiles 1.1 前端可视化
```

---

## 北斗网格层级说明

| 层级 | 编码位数 | 水平尺寸 | 典型用途 |
|------|----------|----------|----------|
| L1   | 5 位     | ~100 万图幅范围  | 全国索引 |
| L2   | 8 位     | 1°×1°，~111 km  | 省级 |
| L3   | 11 位    | 1:5 万图幅       | 市级 |
| L4   | 14 位    | 1′×1′，~1.85 km | 区县级 |
| L5   | 17 位    | 4″×4″，~123.69 m | 街区 |
| L6   | 20 位    | 2″×2″，~61.84 m  | 楼组 |
| L7   | 23 位    | 1/4″×1/4″，~7.73 m | 建筑 |
| L8   | 26 位    | 1/32″×1/32″，~1 m  | 精细 |
| L9   | 29 位    | 1/256″×1/256″，~12.5 cm | 超精细 |
| L10  | 32 位    | 1/2048″×1/2048″，~1.5 cm | 厘米级 |

> 每级在父码基础上追加 3 位子码（行+列+层标），2×2 细分（4 个子格）。

---

## 安装说明

### 环境要求

- Python 3.10+
- GDAL（系统级安装）

### 安装依赖

```bash
# 建议使用 conda 环境安装 GDAL
conda create -n urban-grid python=3.10
conda activate urban-grid
conda install -c conda-forge gdal rasterio geopandas

pip install -r requirements.txt
```

### 开发模式安装

```bash
cd urban-grid-tiles
pip install -e .
```

---

## 快速开始

### 1. 使用默认配置离线生成瓦片

```bash
cd urban-grid-tiles

# 使用默认配置（北京五环范围，无真实数据源，生成空高度场体素）
python scripts/generate_tiles.py --config config/default_config.yaml

# 指定真实数据源
python scripts/generate_tiles.py \
    --config config/default_config.yaml \
    --terrain-tif /data/beijing_dem.tif \
    --building-vector /data/beijing_buildings.shp \
    --output-dir ./output/beijing_tiles
```

### 2. 执行 LOS 遮蔽分析

```bash
python scripts/run_los_analysis.py \
    --radar-lon 116.3 \
    --radar-lat 39.9 \
    --radar-alt 100 \
    --max-range 5000 \
    --level 7 \
    --tileset-dir ./output/beijing_tiles \
    --output-dir ./output/los_result
```

输出：
- `./output/los_result/tileset.json`：子集 tileset，只包含可见网格
- `./output/los_result/los_result.json`：分析结果摘要（可见/遮挡网格码列表）

### 3. 运行单元测试

```bash
cd urban-grid-tiles
pip install pytest
pytest tests/ -v
```

---

## 配置说明

配置文件位于 `config/default_config.yaml`，主要参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `bbox` | 城市范围（经纬度） | 北京五环 |
| `tile_levels` | 生成的北斗层级列表 | [4, 5, 6, 7] |
| `analysis_level` | LOS 分析使用的层级 | 7 |
| `vertical.min_alt` | 最低高程（米） | 0 |
| `vertical.max_alt` | 最高高程（米） | 500 |
| `vertical.step` | 垂直层步长（米） | 10 |
| `data_sources.terrain_tif` | GeoTIFF DEM 路径 | 空 |
| `data_sources.building_vector` | 建筑矢量路径 | 空 |
| `output.dir` | 输出目录 | ./output/tiles |

---

## 模块说明

| 模块 | 位置 | 说明 |
|------|------|------|
| `beidou_grid` | `grid/beidou_grid.py` | 北斗网格编码/解码/层级操作 |
| `voxelizer` | `grid/voxelizer.py` | 三维体素化（水平×垂直） |
| `base` | `data_sources/base.py` | 数据源抽象基类 |
| `tif_loader` | `data_sources/tif_loader.py` | GeoTIFF 地形读取 |
| `vector_loader` | `data_sources/vector_loader.py` | shp/geojson 建筑矢量读取 |
| `height_field` | `data_sources/height_field.py` | 统一高度场（合并地形+建筑） |
| `gltf_builder` | `tiles/gltf_builder.py` | glTF instancing 瓦片生成 |
| `tileset_builder` | `tiles/tileset_builder.py` | tileset.json 全量/子集构建 |
| `tile_writer` | `tiles/tile_writer.py` | 瓦片文件写出 |
| `los_analysis` | `analysis/los_analysis.py` | 射线视域分析（LOS） |
| `pipeline` | `pipeline.py` | 主流程编排 |

---

## 路线图

### 第一阶段（当前）：端到端逻辑验证

- [x] 北斗网格编码（L1~L10）
- [x] 统一高度场（GeoTIFF + 矢量建筑）
- [x] 三维体素化
- [x] glTF EXT_mesh_gpu_instancing 瓦片
- [x] tileset.json（全量 + 子集）
- [x] LOS 射线遮蔽分析
- [x] CLI 工具（generate_tiles / run_los_analysis）

### 第二阶段：Rust 加速

- [ ] 北斗网格编码/解码 Rust 扩展（PyO3）
- [ ] 射线步进加速（Rust + rayon 并行）
- [ ] 高度场查询 KD-Tree / 空间索引

### 第三阶段：生产化

- [ ] FastAPI 在线服务（动态子集 tileset 接口）
- [ ] 支持 Cesium 地形切片（quantized-mesh）输入
- [ ] 支持 3D Tiles 白模/倾斜摄影作为遮挡体
- [ ] Draco/KTX2 压缩输出
- [ ] 分布式切片（多进程/云函数）
- [ ] 前端样式接口（按占用类型/层级/强度着色）
