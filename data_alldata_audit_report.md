# `data/alldata` 数据检查报告

## 1. 检查范围

本次检查覆盖以下目录：

- `data/alldata/data`
- `data/alldata/data2`
- `data/alldata/Daten LOGIN`
- `data/alldata/newdata`
- `data/alldata/newdata2`

检查目标：

- 去掉并统计重复点
- 统计去重后各数据集的条数
- 找出 `Daten LOGIN/ReadMe.txt` 提到的脏数据
- 判断 `data/alldata/data` 下是否存在“处理前/处理后”的两份数据

---

## 2. `csv` 数据重复检查结果

### 2.1 `Messdaten 20K`

目录：

- `data/alldata/data/Messdaten 20K`

结果：

- 文件数：`35`
- 总行数：`2113`
- 去重后行数：`2113`
- 重复行数：`0`

结论：

- 没有发现整行完全重复的数据点。

### 2.2 `Alle Messdaten mit P_Fan und EEV_Opening`

目录：

- `data/alldata/data/Alle Messdaten mit P_Fan und EEV_Opening`

结果：

- 文件数：`144`
- 总行数：`8715`
- 去重后行数：`8715`
- 重复行数：`0`

结论：

- 没有发现整行完全重复的数据点。

### 2.3 `2024_08_09_OptiHorn`

文件：

- `data/alldata/data/2024_08_09_OptiHorn`

结果：

- 总行数：`140700`
- 去重后行数：`140700`
- 重复行数：`0`

结论：

- 没有发现整行完全重复的数据点。

### 2.4 `csv` 汇总

- `Messdaten 20K`：`2113` 条，无重复
- `Alle Messdaten mit P_Fan und EEV_Opening`：`8715` 条，无重复
- 两者合计：`10828` 条，无重复

---

## 3. `csv` 结构性问题

目录：

- `data/alldata/data/Alle Messdaten mit P_Fan und EEV_Opening`

发现：

- 该目录下 `144` 个文件全部存在表头重复含义问题。
- 文件中同时出现 `EEV_opening` 和 `EEV_Opening` 两列。

说明：

- 这不是“重复数据点”，但属于结构性脏点。
- 一些读取工具会把它视为重复列名或冲突列名。

---

## 4. `pkl` 数据重复检查结果

说明：

- 这里的“重复”分两层：
- 第一层：单个文件内部是否有重复行
- 第二层：不同目录下是否有完全相同的文件拷贝

### 4.1 单文件内部重复

结果：

- 本次读取到的所有 `pkl` 文件，文件内部重复行数均为 `0`。

结论：

- 各 `pkl` 数据集内部没有发现重复点。

### 4.2 去掉跨目录重复拷贝后的独立数据集数量

结果：

- 独立数据集总数：`36`

这里“独立数据集”按文件内容哈希去重，完全相同的拷贝只算一次。

---

## 5. 去重后各制冷剂条数

以下统计基于“独立数据集”口径，即跨目录完全相同的拷贝只算一次。

### 5.1 `Daten LOGIN` / `data2` / `newdata2` 这一批

- `R1234yf`：`8` 条
- `R1234yf_R32_64_36`：`4` 条
- `R1270`：`16` 条
- `R1270_R600a_92_8`：`8` 条
- `R134a`：`4` 条
- `R134a_2`：`4` 条
- `R290`：`8` 条
- `R290_R600a_63_37`：`8` 条
- `R290_R600a_82_18`：`8` 条
- `R600a`：`13` 条
- `R1270_2`：`4` 条
- `R1270_3`：`4` 条
- `R1270_4`：`4` 条

说明：

- `R1270 = 16` 条来自 4 份独立数据：
- `fixed_data_simple_35_R1270.pkl`
- `fixed_data_simple_55_R1270.pkl`
- `fixed_data_ihx_35_R1270.pkl`
- `fixed_data_ihx_55_R1270.pkl`

- `R600a = 13` 条来自 4 份独立数据：
- `fixed_data_simple_35_R600a.pkl`：`3` 条
- `fixed_data_simple_55_R600a.pkl`：`4` 条
- `fixed_data_ihx_35_R600a.pkl`：`3` 条
- `fixed_data_ihx_55_R600a.pkl`：`3` 条

### 5.2 `newdata` 这一批

全部为 `R290`，但按工况文件区分：

- `B-3W35_R290`：`6` 条
- `B-3W50_R290`：`6` 条
- `B-3W65_R290`：`5` 条
- `B12W35_R290`：`7` 条
- `B12W50_R290`：`7` 条
- `B12W65_R290`：`6` 条
- `B2W35_R290`：`7` 条
- `B2W50_R290`：`6` 条
- `B2W65_R290`：`6` 条
- `B7W35_R290`：`7` 条
- `B7W50_R290`：`7` 条
- `B7W65_R290`：`6` 条

---

## 6. `ReadMe.txt` 提到的脏数据

文件：

- `data/alldata/Daten LOGIN/ReadMe.txt`

原文意思：

- `ihx 55 R1270` 这部分数据“不干净”，需要谨慎使用。

对应数据文件位置：

- `data/alldata/newdata2/fixed_data_ihx_55_R1270.pkl`

结果：

- 该文件总条数：`4`
- 去重后条数：`4`
- 重复条数：`0`

### 6.1 为什么判断这 4 条是脏数据

读取后发现，这 `4` 条数据里多个字段出现明显不合理的异常值，例如：

- `dE_v_eva_v1` 约为 `-1e14` 到 `-1e15`
- `dE_v_eva_v2` 约为 `1e14` 到 `1e15`
- `T_m_eva_ref` 约为 `-1e13` 到 `-1e14`
- `Q_flow_eva_ref` 为负且量级异常，约为 `-2363` 到 `-5519`

结论：

- `ReadMe.txt` 指向的脏数据确实存在。
- 脏数据位置是 `newdata2/fixed_data_ihx_55_R1270.pkl`。
- 脏数据条数按当前文件内容看是 `4` 条，即整份文件都应视为可疑数据。

---

## 7. `data/alldata/data` 下是否存在“处理前/处理后”两份数据

本次重点比较了：

- `data/alldata/data/Messdaten 20K`
- `data/alldata/data/Alle Messdaten mit P_Fan und EEV_Opening`
- `data/alldata/data/2024_08_09_OptiHorn`

### 7.1 `Messdaten 20K` vs `Alle Messdaten mit P_Fan und EEV_Opening`

结论：

- 它们不是简单的“处理前/处理后”关系。

依据：

- 两者在同一频率组合下，前 `61` 列基本一致。
- `Alle Messdaten mit P_Fan und EEV_Opening` 比 `Messdaten 20K` 额外多一列 `EEV_Opening`。
- 但抽样比对同频率文件后，去掉 `EEV_Opening` 这一列，行内容重合数仍为 `0`。

说明：

- 这意味着两者不是“同一批记录，一份原始、一份清洗后”。
- 更合理的判断是：它们是结构相近、但来源不同或分组不同的两批数据。

### 7.2 更像原始数据的是谁

结论：

- `data/alldata/data/2024_08_09_OptiHorn` 更像上游原始数据。

依据：

- 该文件列数更少，只有 `31` 列。
- 含有更原始的控制或采集字段，例如 `fSetOpeningEEV_heatmode`。
- 而 `Alle Messdaten mit P_Fan und EEV_Opening` 明显是拆分整理后的派生结果。

### 7.3 最终判断

- `Messdaten 20K`：不是 `Alle Messdaten...` 的简单“处理前版本”
- `Alle Messdaten mit P_Fan und EEV_Opening`：不是 `Messdaten 20K` 的简单“处理后版本”
- `2024_08_09_OptiHorn`：更像原始数据源

---

## 8. 哪些 `pkl` 文件只是拷贝，不是“处理前/处理后”

以下文件内容完全一致，只是位于不同目录，不能视为“清理前/清理后”的不同版本：

### 8.1 `fixed_data_simple_35_R1270.pkl`

完全相同的文件：

- `Daten LOGIN/fixed_data_simple_35_R1270.pkl`
- `data2/fixed_data_simple_35_R1270.pkl`
- `newdata2/fixed_data_simple_35_R1270.pkl`
- `newdata2/fixed_data_simple_35_R1270_1.pkl`

### 8.2 `fixed_data_simple_35_R290.pkl`

- `Daten LOGIN/fixed_data_simple_35_R290.pkl`
- `data2/fixed_data_simple_35_R290.pkl`
- `newdata2/fixed_data_simple_35_R290.pkl`

### 8.3 `fixed_data_simple_35_R600a.pkl`

- `Daten LOGIN/fixed_data_simple_35_R600a.pkl`
- `data2/fixed_data_simple_35_R600a.pkl`
- `newdata2/fixed_data_simple_35_R600a.pkl`

### 8.4 `fixed_data_simple_55_R1270.pkl`

- `Daten LOGIN/fixed_data_simple_55_R1270.pkl`
- `data2/fixed_data_simple_55_R1270.pkl`
- `newdata2/fixed_data_simple_55_R1270.pkl`

### 8.5 `fixed_data_simple_55_R290.pkl`

- `Daten LOGIN/fixed_data_simple_55_R290.pkl`
- `data2/fixed_data_simple_55_R290.pkl`
- `newdata2/fixed_data_simple_55_R290.pkl`

### 8.6 `fixed_data_simple_55_R600a.pkl`

- `Daten LOGIN/fixed_data_simple_55_R600a.pkl`
- `data2/fixed_data_simple_55_R600a.pkl`
- `newdata2/fixed_data_simple_55_R600a.pkl`

### 8.7 需要单独注意的文件

- `newdata2/fixed_data_simple_35_R1270_2.pkl`
- `newdata2/fixed_data_simple_35_R1270_3.pkl`
- `newdata2/fixed_data_simple_35_R1270_4.pkl`

这三个文件与 `fixed_data_simple_35_R1270.pkl` 不同，属于独立数据集，不是简单拷贝。

---

## 9. 最终结论

### 9.1 重复点

- `csv` 数据中未发现整行重复点。
- `pkl` 数据中未发现文件内部重复点。

### 9.2 脏数据

- `Daten LOGIN/ReadMe.txt` 指向的脏数据位于：
- `data/alldata/newdata2/fixed_data_ihx_55_R1270.pkl`
- 该文件共有 `4` 条，建议整体视为脏数据。

### 9.3 “处理前/处理后”关系

- `data/alldata/data` 下现有两批主要 `csv` 数据不是简单的处理前后关系。
- `2024_08_09_OptiHorn` 更像原始数据源。

### 9.4 目录间重复文件

- `data2`、`Daten LOGIN`、`newdata2` 之间有多组完全相同的 `simple` 文件拷贝。
- 这些文件不能当作不同清洗版本重复统计。
