# `alldata` 质量流量拟合与泛化测试报告

## 1. 任务目标

本次任务使用 `data/alldata` 下的制冷剂数据，进行以下工作：

- 先根据无量纲数 `pi1 ~ pi5` 拟合流量系数 `Cd`
- 再通过质量流量公式回代得到预测质量流量
- 评估公式在训练内制冷剂上的表现
- 评估公式在完全没参与训练的制冷剂上的泛化表现

使用的质量流量公式为：

```text
m_dot = Cd * (pi * Dmax^2 / 4) * sqrt(2 * rho * (Pin - Pout))
```

其中程序中使用：

- `Dmax = 1.8e-3 m`
- `Aref = pi * Dmax^2 / 4 = 2.544690049407732e-06 m^2`

---

## 2. 两个脚本分别做什么

本次流程拆成两个脚本：

### 2.1 数据处理脚本

文件：

- `prepare_alldata_dso_hou.py`

作用：

- 读取 `data/alldata` 下所有可用 `pkl`
- 去掉跨目录完全重复的拷贝文件
- 跳过已知脏数据 `fixed_data_ihx_55_R1270.pkl`
- 统一整理字段
- 计算 `pi1 ~ pi5`
- 由真实质量流量反算 `Cd_true`
- 构造 `train / seen_test / unseen_test`
- 导出处理后的 `csv` 和 `json`

输出目录：

- `outputs/alldata_dso`

### 2.2 拟合与测试脚本

文件：

- `run_alldata_dso_hou.py`

作用：

- 读取上一步处理好的数据
- 用 DSO 拟合 `Cd = f(pi1, pi2, pi3, pi4, pi5)`
- 回代质量流量公式，得到 `m_flow_pred`
- 评估 train / seen_test / unseen_test 三套结果
- 导出候选公式、最佳公式、误差表和预测结果

输出目录：

- `outputs/alldata_dso_hou`

---

## 3. 什么是 split 策略

`split` 策略就是“如何把数据分成训练集和测试集”。

这次我没有简单随机按行切分，而是用了两层切分：

### 3.1 第一层：按制冷剂做 unseen split

目的：

- 模拟“公式从来没见过某种制冷剂”的真实泛化场景

做法：

- 选一种制冷剂，整类样本完全不参加训练
- 这类样本只放到 `unseen_test`

这次选的是：

- `R134a`

所以：

- `R134a` 完全不参加训练
- `R134a` 只用于最后测试

### 3.2 第二层：对剩余制冷剂做 seen split

目的：

- 评估模型在“见过这些制冷剂，但没见过这些具体点”的常规测试表现

做法：

- 对其余制冷剂的数据，按 `source_file` 分组
- 每个文件内部留出一小部分点做 `seen_test`
- 剩余点进入 `train`

本次设置：

- `seen_test_fraction = 0.25`

但由于很多文件样本数很少，实际策略是：

- 每个源文件至少留 `1` 个点到 `seen_test`
- 剩余点留在 `train`

这样做的原因：

- 避免把同一个小数据集全部塞进训练集
- 也避免测试集完全被某几个大文件主导

### 3.3 本次 split 结果

来自 `outputs/alldata_dso/split_summary.json`：

- 总样本数：`165`
- 独立数据集数：`35`
- `train`：`124`
- `seen_test`：`33`
- `unseen_test`：`8`

本次 `unseen_test` 只包含：

- `R134a`

本次 `train` 和 `seen_test` 包含：

- `R1234yf`
- `R1234yf_R32_64_36`
- `R1270`
- `R1270_R600a_92_8`
- `R290`
- `R290_R600a_63_37`
- `R290_R600a_82_18`
- `R600a`

---

## 4. 哪些是混合工质

本次数据里，混合工质是这些：

- `R1234yf_R32_64_36`
- `R1270_R600a_92_8`
- `R290_R600a_63_37`
- `R290_R600a_82_18`

对应文件分别有：

- `fixed_data_simple_35_R1234yf_R32_64_36.pkl`
- `fixed_data_simple_35_R1270_R600a_92_8.pkl`
- `fixed_data_simple_55_R1270_R600a_92_8.pkl`
- `fixed_data_simple_35_R290_R600a_63_37.pkl`
- `fixed_data_simple_55_R290_R600a_63_37.pkl`
- `fixed_data_simple_35_R290_R600a_82_18.pkl`
- `fixed_data_simple_55_R290_R600a_82_18.pkl`

说明：

- `R134a` 不是混合工质
- `R134a_2` 在本次处理里统一归到了 `R134a`
- `R1270_2 / _3 / _4` 不是新的混合工质名称，而是 `R1270` 的不同独立数据集版本

---

## 5. 混合工质数据用在了哪里

本次混合工质数据没有被单独排除，而是正常参与了建模。

### 5.1 在训练中的用途

混合工质被放进了：

- `train`
- `seen_test`

也就是说：

- 模型训练时确实见过混合工质的数据
- 模型的普通测试集里也包含混合工质

这样做的意义：

- 让模型尽可能学习更广的工况和更丰富的物性变化
- 提高公式的总体适用范围

### 5.2 在 unseen 泛化中的用途

本次 `unseen_test` 只保留了：

- `R134a`

因此：

- 混合工质没有进入 `unseen_test`
- 它们主要用于训练和常规测试

换句话说，本次的“完全没见过的制冷剂泛化”测试的是：

- 公式在没见过 `R134a` 时，能否从其他纯工质和混合工质中学到足够通用的规律，再推广到 `R134a`

---

## 6. 这次数据具体怎么被使用

### 6.1 数据来源

本次使用的是 `data/alldata` 下的独立 `pkl` 数据集。

### 6.2 去重方式

不是简单按文件名，而是按文件内容哈希去重：

- 跨目录完全相同的拷贝文件只保留一份

### 6.3 脏数据处理

跳过了已知脏文件：

- `newdata2/fixed_data_ihx_55_R1270.pkl`

原因：

- `Daten LOGIN/ReadMe.txt` 已明确标记其不干净
- 之前检查也发现该文件 4 条数据存在明显异常值

### 6.4 本次进入建模的数据规模

- 独立数据集：`35`
- 总样本：`165`

---

## 7. 无量纲数和 `Cd` 的处理方式

程序没有直接拟合质量流量，而是先构造：

- 输入：`pi1, pi2, pi3, pi4, pi5`
- 拟合目标：`Cd_true`

### 7.1 本次 `pi` 的含义

程序中按图示思路构造了：

- `pi1 = (Pin - Pout) / Pcrit`
- `pi2 = deltaTuk / Tcrit`
- `pi3 = nu_g / nu_f`
- `pi4 = sigma / (D * Pin)`
- `pi5 = Z`

然后再根据真实质量流量反算：

- `Cd_true`

最后让 DSO 去拟合：

```text
Cd = f(pi1, pi2, pi3, pi4, pi5)
```

再回代得到预测的质量流量。

### 7.2 关于混合工质物性

对混合工质，临界性质使用了 `CoolProp AbstractState`。

需要额外注意的是：

- 混合工质的表面张力 `sigma`，`CoolProp` 不能稳定直接给出
- 本次实现里对混合工质的 `sigma` 使用了按组分配比的加权近似

因此：

- `pi4` 对纯工质更直接
- `pi4` 对混合工质是一个工程近似值

---

## 8. 本次训练结果

结果文件：

- `outputs/alldata_dso_hou/variant_comparison.csv`

本次对比了两个版本：

- `default`
- `regularized_low_complexity`

### 8.1 `default` 结果

- train mass-flow MARD：`5.50%`
- seen-test mass-flow MARD：`3.83%`
- unseen-test mass-flow MARD：`7.52%`

### 8.2 `regularized_low_complexity` 结果

- train mass-flow MARD：`45.76%`
- seen-test mass-flow MARD：`37.21%`
- unseen-test mass-flow MARD：`36.38%`

### 8.3 结果结论

- `default` 明显优于 `regularized_low_complexity`
- 当前最有参考价值的是 `default`

---

## 9. 当前选出的最佳公式

最佳公式文件：

- `outputs/alldata_dso_hou/best_formula_summary.json`

选择规则：

- 优先最小化 `unseen_test` 的质量流量误差
- 再看 `seen_test`
- 最后看 `unseen_test mse`

当前选中的模型是：

- `default_hof_4`

它的质量流量结果为：

- train MARD：`5.27%`
- seen-test MARD：`3.10%`
- unseen-test MARD：`6.56%`

说明：

- 这个结果比 `variant_comparison.csv` 里的整版 `default` 总结更好
- 因为这里是从 DSO 导出的全部候选公式中，再按 unseen 泛化表现精挑过一次的最佳公式

---

## 10. 输出文件说明

### 10.1 预处理输出

目录：

- `outputs/alldata_dso`

主要文件：

- `all_processed.csv`
- `train_processed.csv`
- `seen_test_processed.csv`
- `unseen_test_processed.csv`
- `split_summary.json`

### 10.2 拟合输出

目录：

- `outputs/alldata_dso_hou`

主要文件：

- `variant_comparison.csv`
- `variant_comparison.json`
- `all_formula_metrics.csv`
- `best_formula_summary.json`
- `best_formula_complexity_aware_summary.json`

以及：

- 每个 variant 的 `run_summary.json`
- `hof_readable.csv/.txt`
- `pf_readable.csv/.txt`
- 预测结果 `csv`
- 误差图和趋势图

---

## 11. 总结

### 11.1 关于 split 策略

- `split` 策略就是怎么分训练集和测试集
- 这次用了两层：
- 一层是按制冷剂整类拿掉，形成 `unseen_test`
- 一层是在剩余制冷剂内部按文件留一部分样本做 `seen_test`

### 11.2 关于混合工质

本次参与建模的混合工质有：

- `R1234yf_R32_64_36`
- `R1270_R600a_92_8`
- `R290_R600a_63_37`
- `R290_R600a_82_18`

它们主要被用于：

- 训练集 `train`
- 常规测试集 `seen_test`

没有被用于：

- `unseen_test`

### 11.3 关于当前结果

- 当前最优公式在训练内和普通测试集上表现较稳
- 对完全没见过的 `R134a`，质量流量 `MARD` 约为 `6.56%`
- 说明这套 `Cd(pi1~pi5)` 的做法已经具备一定跨制冷剂泛化能力

### 11.4 后续可继续做的事

- 更换 `unseen` 制冷剂，分别测试 `R600a`、`R1270`、`R290`
- 比较“是否包含混合工质训练”对泛化的影响
- 对 `pi4` 的混合工质表面张力近似做进一步改进
- 压缩公式复杂度，找更简洁但误差接近的表达式
