# 数据集与数据处理技术报告（技术草稿）

> 本报告从工程实现与实验复现的角度，总结本项目中数据目录结构、CANOPY HEIGHT MODEL（CHM）生成流水线（`data-processing/gen-chm`）、训练数据预处理流水线（`data-processing/preprocessing`）、运行步骤与关键统计结果。文本偏“技术说明 + 提纲”，方便后续改写为正式学术语言，用于论文第 4 章：
>
> - 4.1 TreeSatAI Dataset Overview  
> - 4.2 CHM Generations  
> - 4.3 Data Preprocessing Pipeline, Dataset Splitting and Validation Strategy

---

## 1. 数据目录结构与原始数据

### 1.1 顶层数据与代码目录

- **原始与中间数据（`data/`）**
  - `data/lgln-opengeodata/`
    - `lgln-opengeodata-dgm1.geojson`：数字地面模型（DGM1）多边形索引（1 m 分辨率 DEM 底面）。
    - `lgln-opengeodata-dom1.geojson`：数字表面模型（DOM1）多边形索引（1 m 分辨率 DSM 顶面）。
  - `data/treesat_ori/`
    - `geojson/`
      - `bb_60m.GeoJSON`、`bb_200m.GeoJSON`：TreeSatAI 规则网格（60 m / 200 m）空间单元的多边形边界。  
        额外还包含 train/test/val 子集的 `bb_*_train.GeoJSON` / `bb_*_test.GeoJSON` 等，用于原始基准划分。
    - `labels/TreeSatBA_v9_60m_multi_labels.json`：TreeSatAI 60 m 栅格多标签文件，每个 patch 文件名映射到一组 `[species_label, area_fraction]` 对。
    - `s1/60m`, `s1/200m`：Sentinel‑1 后处理强度图（VV、VH）在 60 m / 200 m 网格上的栅格切片（原 TreeSatAI 基准提供）。
    - `s2/60m`, `s2/200m`：Sentinel‑2 影像切片（本工作中主要侧重 SAR，S2 不在当前流水线中使用）。
  - **CHM 与派生产品**（由流水线生成）：
    - `data/dtm1_tif/`：下载后的 DTM1 栅格瓦片。
    - `data/dsm1_tif/`：下载后的 DSM1 栅格瓦片。
    - `data/chm/`：以原始 1 m 分辨率生成的 CHM 瓦片（DSM − DTM）。
    - `data/chm_60m/`、`data/chm_200m/`：重投影到 TreeSatAI UTM 坐标系并经过块状最大池化后，对齐 60 m / 200 m 模板的 CHM patch。
    - `data/chm_stacked_treesat_60m/`（以及对应 200 m 版本）或 `data/stacked_treesat_60m/`：将 S1（VV、VH、VV/VH）与 CHM 堆叠后的 4‑band GeoTIFF patch。
  - **预处理与训练数据**：
    - `data/treesatai_data/labels/TreeSatBA_v9_60m_multi_labels.json`：多标签源文件（与 `treesat_ori/labels/` 中内容一致，放在统一的数据根下便于调用）。
    - `data/treesatai_data/s1_60m_stratified/`：预处理后的标签与统计信息：
      - `labels_train_filtered.json`, `labels_val_filtered.json`, `labels_test_filtered.json`
      - `classes.json`：物种类别列表
      - `label_stats.json`：样本数与类别频次
      - `class_weights.json`：多种类别权重策略（逆频率、median freq、effective number 等）。
    - `data/training-data-60m/`：最终用于模型训练的 numpy 数组：
      - `train_x.npy`, `val_x.npy`, `test_x.npy`：形状为 `(N, 4, H, W)` 的 S1+CHM patch
      - `train_y.npy`, `val_y.npy`, `test_y.npy`：形状 `(N, C)` 的 multi‑hot 标签
      - `classes.npy`：类别名
      - `*_filenames.npy`：对应 patch 文件名，便于回溯。

- **处理代码（`data-processing/`）**
  - `gen-chm/`：完整的 CHM 生成与堆叠流水线，包括 DTM/DSM 链接构建、反向映射、瓦片下载、CHM 计算、重投影+max‑pooling，以及 S1+CHM 堆叠与可视化。
  - `preprocessing/`：面向下游深度学习的预处理模块，包括面积阈值过滤和分层划分、类别权重计算、训练数组构建（含缺失值插值）、以及训练数据分布的可视化。

- **可视化输出（`plots/`）**
  - `plots/`：CHM 相关可视化，例如：
    - `dtm_dsm_chm.png`：单个 tile 上 DTM/DSM/CHM 的三联图，用于检查 DSM/DTM 对齐及 CHM 质量。  
      `![DTM/DSM/CHM 示例](../plots/dtm_dsm_chm.png)`
    - `stacked_s1_chm_60m.png`（以及可能的 200 m 版本）：多列样本、4 行 band（VV/VH/VV/VH/CHM）的可视化，用于直观理解堆叠后 patch 的外观与动态范围。  
      `![S1+CHM 堆叠示例（60 m）](../plots/stacked_s1_chm_60m.png)`
  - `plots/training-data-insight/`：训练数据分布诊断图：
    - `sample_grid_60m.png`：train/val/test × band 的样本网格。  
      `![训练样本网格（60 m）](../plots/training-data-insight/sample_grid_60m.png)`
    - `train_histograms_60m.png`, `val_histograms_60m.png`, `test_histograms_60m.png`：三种划分下四个 band 的直方图。  
      `![Train 直方图](../plots/training-data-insight/train_histograms_60m.png)` 等。
    - `class_distribution_60m.png`：train/val/test 主类分布条形图。  
      `![主类分布对比](../plots/training-data-insight/class_distribution_60m.png)`

这些图像在后续撰写第 4 章时，可分别放入：

- 4.2 CHM Generations：`dtm_dsm_chm.png`、`stacked_s1_chm_60m.png`
- 4.3 Data Preprocessing Pipeline, Dataset Splitting and Validation Strategy：`sample_grid_60m.png`、三张 histogram、`class_distribution_60m.png`

---

## 2. CHM 生成流水线（`data-processing/gen-chm`）

### 2.1 DTM/DSM – TreeSat 网格链接构建（gen1）

脚本：`gen1_get_dtm_dsm_treesat_links.py`

- **输入**：
  - LGLN DGM1/DOM1 多边形：`data/lgln-opengeodata/lgln-opengeodata-dgm1.geojson` / `...-dom1.geojson`
  - TreeSat 网格：`data/treesat_ori/geojson/bb_60m.GeoJSON`, `bb_200m.GeoJSON`
- **方法**：
  - 利用 `geopandas` 读入大瓦片与网格多边形，构建 DGM1/DOM1 瓦片的 R‑tree 索引。
  - 对每个 TreeSat 网格单元，基于包围盒与精确几何相交关系，找到所有覆盖它的 DTM/DSM 瓦片，并记录对应的 `tile_id` 与下载 URL。
- **输出**：
  - `data-processing/dtm-dsm-treesat-links/dtm1_links_{60,200}m.json`
  - `data-processing/dtm-dsm-treesat-links/dsm1_links_{60,200}m.json`

根据 `data-processing/output.txt`，以 DTM 为例，链接结果可整理为表 1。

**表 1 DTM 链接统计（节选）**

| 分辨率 (m) | DGM1 多边形数 | TreeSat 网格数 | 覆盖到至少 1 DTM 瓦片的网格数 | 有覆盖的 DTM 瓦片数 | TreeSat 覆盖比例 |
|-----------:|---------------:|---------------:|--------------------------------:|--------------------:|------------------:|
| 60         | 49,573         | 50,381         | 50,381                           | 6,210               | 100%              |
| 200        | 49,573         | 50,381         | 50,381                           | 6,810               | 100%              |

（DSM 的统计量相似，DOM1 多边形数为 49,609，60 m 与 200 m 均实现对 50,381 个 TreeSat 网格 100% 的覆盖。）

该步骤保证了后续每个 TreeSat patch 至少可以从一个 DTM/DSM 瓦片对中推导出 CHM。

### 2.2 反向映射：TreeSat → {tile_id}（gen2）

脚本：`gen2_reverse_mapping.py`

- **输入**：`dtm1_links_{res}m.json`, `dsm1_links_{res}m.json`
- **目标**：把 tile→TreeSat 的映射反转为 `treesat_id → {dtm_tile_ids, dsm_tile_ids}` 的结构，识别哪些网格需要多瓦片拼接（mosaic）。
- **输出**：`treesat_to_tiles_{60,200}m.json`

根据运行日志，60 m 与 200 m 的映射结果如下（表 2）。

**表 2 TreeSat 网格对应 DTM/DSM 瓦片数量统计**

| 分辨率 (m) | TreeSat 样本数 | 单瓦片样本数 (%) | 多瓦片样本数 (%) | 典型 DTM 瓦片数分布 |
|-----------:|----------------:|------------------:|------------------:|---------------------|
| 60         | 50,381          | 44,191 (87.7%)    | 6,190 (12.3%)     | 1 瓦片: 44,191; 2 瓦片: 5,957; 4 瓦片: 233 |
| 200        | 50,381          | 32,410 (64.3%)    | 17,971 (35.7%)    | 1 瓦片: 32,410; 2 瓦片: 15,904; 3 瓦片: 4; 4 瓦片: 2,063 |

60 m 下绝大多数 patch 仅依赖单一 DTM/DSM 瓦片，而 200 m 由于空间采样粒度较粗，多瓦片 mosaic 样本比例显著升高，这对后续 mosaic 策略与计算效率有直接影响。

### 2.3 DTM/DSM 批量下载（gen3）

脚本：`gen3_download_dtm_dsm_tiles.py`

- 读取 `treesat_to_tiles_{res}m.json`，提取所有唯一的 `tile_id`。
- 根据 LGLN 提供的 HTTP/HTTPS 下载链接批量获取 GeoTIFF 文件，分别保存至：
  - `data/dtm1_tif/`
  - `data/dsm1_tif/`
- 跳过已存在瓦片，支持失败重试与简单的错误日志记录，保证多次运行具有幂等性。

### 2.4 CHM 生成与 NoData 填补（gen4）

脚本：`gen4_generate_chm.py`

- **输入**：`data/dtm1_tif/`, `data/dsm1_tif/`
- **核心步骤**：
  1. 对每一对 DSM/DTM 瓦片：
     - 保证空间分辨率与栅格对齐（必要时做 resampling）。
     - 识别并填补 DSM/DTM 中的 NoData 像素（例如通过多尺度邻域平均或迭代扩散的方法）。
  2. 计算 CHM = DSM − DTM，得到单位为米的树高估计。
  3. 将结果保存到 `data/chm/`，文件命名沿用 tile_id，便于后续索引与 mosaic。

- **质量检查**：配合可视化脚本 `vis_triplet_dtm_dsm_chm`：

  ```markdown
  ![DTM/DSM/CHM 示例](../plots/dtm_dsm_chm.png)
  ```

  该图为若干示例瓦片的 DTM、DSM 与 CHM 三联图，方便检查：

  - DTM/DSM 是否空间对齐；
  - DSM 是否在树冠区域显著高于 DTM；
  - CHM 是否在水体或裸地处接近 0，在林地内部呈现合理的高度梯度。

### 2.5 CHM 重投影、max‑pooling 与 S1+CHM 堆叠（gen5）

脚本：`gen5_chm_reproj_maxpool_crop_stack.py`

该脚本整合了原先的 “CHM 重投影+裁剪” 与 “S1+CHM 堆叠” 两个步骤，通过 `--resolution` 与 `--stack` 参数控制.

- **重投影与 max‑pooling**：
  - 利用 `treesat_to_tiles_{res}m.json` 读取每个 TreeSat 网格对应的一个或多个 CHM 瓦片，在必要时对多个 1 m CHM 瓦片进行 mosaic。
  - 将 mosaic 后的 CHM 从 LGLN 坐标系重投影到 TreeSatAI 使用的 UTM 网格。
  - 在 CHM 上执行块状最大池化（max‑pooling），空间步长匹配 S1 patch 的分辨率（例如 10 m 或 60 m），以保留林冠最高点信息，抑制局部噪声。
  - 最终裁剪出与 TreeSat S1 模板完全对齐的 CHM patch，保存到 `data/chm_{60,200}m/`。

- **S1+CHM 堆叠**（`--stack`）：
  - 从 `data/treesatai_data/s1/{60,200}m/` 读取 S1 patch（VV、VH）。
  - 计算第三个 SAR feature band：`VV/VH` 比值，用于增强对散射机制差异的刻画。
  - 依据预先统计的 band 分布（均值、方差、分位数）对 S1 band 做裁剪与归一化（参见 §3 的 histogram 结果）。
  - 将 `[VV, VH, VV/VH, CHM]` 沿 channel 维度堆叠为 4‑band GeoTIFF，存入 `data/chm_stacked_treesat_{60,200}m/` 或 `data/stacked_treesat_{60,200}m/`。

- **运行示例**：

  ```bash
  # 60 m：CHM 重投影 + 堆叠
  cd data-processing/gen-chm
  python gen5_chm_reproj_maxpool_crop_stack.py --resolution 60 --stack

  # 200 m：可选
  python gen5_chm_reproj_maxpool_crop_stack.py --resolution 200 --stack
  ```

- **堆叠结果可视化**：

  使用 `vis_stacked_s1_chm` 函数绘制若干样本的 4 行 × N 列图像：

  ```markdown
  ![S1+CHM 堆叠示例（60 m）](../plots/stacked_s1_chm_60m.png)
  ```

  该图有助于直观理解：

  - S1 VV/VH 的动态范围与噪声模式；
  - VV/VH 比值在不同树种与背景上的对比度；
  - CHM 在林冠与空地之间的结构差异。

### 2.6 S1+CHM band 统计（来自 `output.txt`）

在完成 60 m 堆叠后，对 50,381 个 patch 的 band 统计结果可整理为表 3。

**表 3 S1+CHM band 全局统计（60 m，节选）**

| Band   | NoData 文件数 / 总数 | NoData 像素数 / 总数 | Min    | Max    | Mean    | Std    | P1     | P99    |
|--------|----------------------|----------------------|--------|--------|---------|--------|--------|--------|
| VV     | 26 / 50,381 (~0.05%) | 145 / 1,813,716      | −31.22 | 22.74  | −6.17   | 3.71   | −14.48 | 2.42   |
| VH     | 8 / 50,381 (~0.02%)  | 41 / 1,813,716       | −49.73 | 21.94  | −12.41  | 3.48   | −21.17 | −4.61  |
| VV/VH  | 18 / 50,381 (~0.04%) | 104 / 1,813,716      | −288.35| 1,259.18| 0.48   | 1.56   | −0.36  | 1.10   |
| CHM    | 0 / 50,381           | 0 / 1,813,716        | 0.00   | 41.12  | 21.79   | 8.67   | 0.00   | 37.07  |

可以看出：

- CHM band 完全无 NoData，且高度分布集中在 0–40 m 之间，符合德国森林高度的合理范围。  
- VV/VH 原始比值存在极端 outlier，后续在 histogram 中对其剪裁到 [−3, 3] 区间，以获得更稳定的分布可视化。

---

## 3. 训练数据预处理流水线（`data-processing/preprocessing`）

### 3.1 标签面积过滤与分层 8:1:1 划分

脚本：`labels_area_filter_and_split.py`

- **输入**：`data/treesatai_data/labels/TreeSatBA_v9_60m_multi_labels.json`
  - 结构：`{"filename.tif": [["Quercus", 0.35], ["Fagus", 0.28], ...], ...}`
  - 每个元素包含 species 标签及其在 60 m patch 中的面积占比。

- **面积阈值过滤**：
  - 设定阈值 `AREA_THRESHOLD = 0.07`（7%）。
  - 对每个 patch，仅保留 `area_fraction > 0.07` 的 species，形成新的多标签集合 `{filename: [label1, label2, ...]}`。

- **主类定义与分层划分**：
  - 主类（primary class）定义为过滤后标签列表中的第一个标签，即面积占比最大的物种。
  - 以主类为分层键，对每个类别分别做 8:1:1 的随机划分：
    - train:val:test 比例为 0.8:0.1:0.1，类内随机打乱；
    - 保证每个类别在 train 中至少有 1 个样本（若类内样本极少则略微打破精确比例）。

- **输出**：
  - `labels_train_filtered.json`, `labels_val_filtered.json`, `labels_test_filtered.json`
  - `classes.json`：按训练集中出现的类别排序后的物种列表。
  - `label_stats.json`：总样本数与各划分、各类别的统计汇总。

该步骤在不依赖原 TreeSatAI 官方划分的前提下，构建了*面积加权*的分层 8:1:1 划分，更符合以主导物种为目标的树种分类任务设定。

### 3.2 类别权重计算

脚本：`compute_class_weights.py`

- **输入**：
  - `classes.json`
  - `labels_train_filtered.json`
- **输出**：`class_weights.json`
  - 包含多种权重方案：
    - `class_freq`：类别相对频率
    - `class_weights_invfreq_normalized`：归一化逆频率权重
    - `class_weights_median_freq_normalized`：median frequency balancing
    - `class_weights_inv_sqrt_freq_normalized`：1/√f 型权重
    - `class_weights_effective_num_beta_0_99_normalized` 等。

根据 `output.txt`，训练集主频统计（节选）可以组织为表 4。

**表 4 训练集类别频率与逆频率权重（面积过滤后）**

| 类别       | 计数  | 频率 `class_freq` | 归一化逆频率权重 `w_invfreq_norm` |
|------------|------:|------------------:|------------------------------------:|
| Abies      | 1,039 | 0.0169           | 0.0673 |
| Acer       | 2,641 | 0.0431           | 0.0265 |
| Alnus      | 2,704 | 0.0441           | 0.0258 |
| Betula     | 2,815 | 0.0459           | 0.0248 |
| Cleared    | 4,538 | 0.0740           | 0.0154 |
| Fagus      | 8,864 | 0.1446           | 0.0079 |
| Fraxinus   | 2,370 | 0.0387           | 0.0295 |
| Larix      | 3,914 | 0.0638           | 0.0179 |
| Picea      | 8,837 | 0.1442           | 0.0079 |
| Pinus      | 9,264 | 0.1511           | 0.0075 |
| Populus    |   450 | 0.0073           | 0.1553 |
| Prunus     |   315 | 0.0051           | 0.2218 |
| Pseudotsuga| 3,593 | 0.0586           | 0.0194 |
| Quercus    | 9,767 | 0.1593           | 0.0072 |
| Tilia      |   191 | 0.0031           | 0.3658 |

可以看到长尾类别（如 *Tilia*, *Prunus*）在逆频率归一化权重下获得显著放大，这对于缓解类别不平衡、提升小样本物种的召回率具有潜在作用。

### 3.3 训练数组构建与缺失值插值

脚本：`build_training_arrays.py`

- **输入**：
  - 标签：`data/treesatai_data/s1_60m_stratified/labels_{train,val,test}_filtered.json` 与 `classes.json`
  - 影像：`data/stacked_treesat_{res}m/*.tif`（或 `data/chm_stacked_treesat_{res}m/*.tif`），每个文件为 4‑band patch。
- **输出**：
  - `data/training-data-{res}m/` 下的 `*_x.npy`, `*_y.npy`, `classes.npy`, `*_filenames.npy`。

- **缺失值插值（S1 band）**：
  - 对于 VV/VH/VV/VH 三个 SAR band：
    - 将 NaN 与 `-9999` 视为 NoData，构建掩膜。
    - 使用 3×3 加权核执行局部“类双线性”卷积插值：

      \[
      K = \frac{1}{16}
      \begin{bmatrix}
      1 & 2 & 1 \\
      2 & 4 & 2 \\
      1 & 2 & 1
      \end{bmatrix}
      \]

    - 在每次迭代中，利用 `num = \text{convolve(values, K)}` 与 `den = \text{convolve(valid_mask, K)}`，仅对具有有效邻域的 NoData 像素更新，最多迭代 5 次。
    - 若仍有残余 NoData，统一置 0，保证训练数组中不存在 NaN 或极端 nodata 值。
  - 对于 CHM band：认为 CHM 已在生成阶段处理完 NoData，仅通过 `nan_to_num` 移除 NaN，并在后续 histogram 中过滤掉接近 0 的值（例如 `> 0.5 m`），从而突出林冠高度分布。

- **多标签编码**：
  - 对每个样本，根据过滤后的标签列表，构建长度为 `num_classes` 的 multi‑hot 向量 `y`；
  - 在类分布统计中，主类定义为 `argmax(y)` 对应的索引。

### 3.4 训练数据分布可视化

脚本：`preprocessing/vis.py`

1. **样本网格（`vis_sample_grid`）**
   - 以 3×4 子图的形式展示一个 train/val/test 样本（行）在四个 band（列）上的空间模式：

     ```markdown
     ![训练样本网格（60 m）](../plots/training-data-insight/sample_grid_60m.png)
     ```

   - 每个子图配有单独的 colorbar，tick label 使用类似 `stacked_s1_chm` 的格式（小于 1 使用一位小数，大于 1 使用整数）。

2. **band 直方图（`vis_histograms`）**
   - 对 train/val/test 三个划分分别生成 1×4 子图的 histogram：

     ```markdown
     ![Train 直方图](../plots/training-data-insight/train_histograms_60m.png)
     ![Val 直方图](../plots/training-data-insight/val_histograms_60m.png)
     ![Test 直方图](../plots/training-data-insight/test_histograms_60m.png)
     ```

   - 直方图设置：
     - bin 数默认 50；
     - VV/VH 比值 band 限制在 [−3, 3]；
     - CHM 过滤高度小于 0.5 m 的像素，以突出林冠高度分布；
     - y 轴使用科学计数法刻度，便于在较大样本量下阅读。

3. **类别分布柱状图（`vis_class_distribution`）**
   - 展示 train/val/test 三个划分在主类计数上的对比：

     ```markdown
     ![主类分布对比](../plots/training-data-insight/class_distribution_60m.png)
     ```

   - 三组柱状图采用统一的类别顺序，train/val/test 在 x 轴上略微错开，便于比较，每个柱子带有黑色边框以增强可读性。

这些可视化为后续章节提供：

- 对输入 feature 空间分布（SAR backscatter 与 CHM）的直观描述；
- 对类别不平衡与分层抽样效果的定量展示；
- 对 train/val/test 一致性的 sanity check。

---

## 4. 运行指南与复现实验步骤

本节从工程角度总结推荐的运行顺序，帮助读者从原始数据出发重现 CHM 生成与训练数据构建全流程。

### 4.1 CHM 生成阶段

1. **DTM/DSM – TreeSat 链接构建**

   ```bash
   cd data-processing/gen-chm

   # 60 m
   python gen1_get_dtm_dsm_treesat_links.py --type dtm --resolution 60
   python gen1_get_dtm_dsm_treesat_links.py --type dsm --resolution 60

   # 200 m（可选）
   python gen1_get_dtm_dsm_treesat_links.py --type dtm --resolution 200
   python gen1_get_dtm_dsm_treesat_links.py --type dsm --resolution 200
   ```

2. **反向映射（TreeSat → {tile_id}）**

   ```bash
   python gen2_reverse_mapping.py --resolution 60
   # 可选
   python gen2_reverse_mapping.py --resolution 200
   ```

3. **DTM/DSM 瓦片下载**

   ```bash
   python gen3_download_dtm_dsm_tiles.py --types dtm dsm
   ```

4. **CHM 计算（DSM − DTM）**

   ```bash
   python gen4_generate_chm.py \
       --dsm-dir ../../data/dsm1_tif \
       --dtm-dir ../../data/dtm1_tif \
       --out-dir ../../data/chm
   ```

5. **CHM 重投影 + max‑pooling + S1+CHM 堆叠**

   ```bash
   python gen5_chm_reproj_maxpool_crop_stack.py --resolution 60 --stack
   # 可选 200 m
   python gen5_chm_reproj_maxpool_crop_stack.py --resolution 200 --stack
   ```

### 4.2 预处理与训练数据构建

1. **标签面积过滤 + 分层划分 + 类别权重 + 训练数组**

   ```bash
   cd /path/to/repo
   python data-processing/preprocessing/run_all_preprocessing.py --resolution 60
   ```

   运行完成后，可在 `data/treesatai_data/s1_60m_stratified/` 与 `data/training-data-60m/` 下检查生成的 JSON 与 NPY 文件。

2. **训练数据可视化**

   ```bash
   python data-processing/preprocessing/vis.py
   ```

   所有诊断图将输出至 `plots/training-data-insight/`，并可直接在论文第 4 章中引用。

---

## 5. 相关研究与参考文献（建议引用）

以下文献可为第 4 章的数据与方法部分提供背景支撑（TreeSatAI 数据集、SAR 与森林参数估计、深度学习与多任务学习等）。

1. **TreeSatAI 数据集与多标签树种制图**  
   Kellenberger, B., Lang, N., Follmann, T., et al. (2022). *The TreeSatAI Benchmark Archive: A multi-sensor, multi-label dataset for tree species mapping*. ISPRS Journal of Photogrammetry and Remote Sensing, 183, 236–250.

2. **SAR 遥感理论基础**  
   Woodhouse, I. H. (2017). *Introduction to Microwave Remote Sensing*. CRC Press.

3. **深度学习在遥感中的综述**  
   Zhu, X. X., Tuia, D., Mou, L., et al. (2017). Deep learning in remote sensing: A comprehensive review and list of resources. *IEEE Geoscience and Remote Sensing Magazine*, 5(4), 8–36.  
   Ma, L., Liu, Y., Zhang, X., et al. (2019). Deep learning in remote sensing applications: A meta-analysis and review. *ISPRS Journal of Photogrammetry and Remote Sensing*, 152, 166–177.

4. **多任务学习综述**  
   Ruder, S. (2017). An overview of multi-task learning in deep neural networks. *arXiv preprint* arXiv:1706.05098.

5. **森林高度与生物量参数的遥感反演（示例性文献，可在正式写作时选取更贴近研究区与传感器的工作）**  
   可参考若干利用 SAR 或 LiDAR 数据反演森林高度、体积与生物量的研究，以论证引入 CHM 与 SAR feature 的合理性，并与本工作强调的 TreeSatAI 多传感器、多任务设置形成呼应。

---

本技术报告提供了第 4 章所需的关键要素：

- 数据目录与文件结构的清晰说明；
- CHM 生成与 S1+CHM 堆叠的完整工程流水线；
- 标签过滤、类别权重与训练数组构建的预处理流程；
- 基于 `data-processing/output.txt` 的核心结果表格化整理；
- 以及与 `plots/` 目录中图像一一对应的放置位置建议。

后续可在此基础上，以更正式的学术语言重写为完整的 4.1–4.3 章节。
