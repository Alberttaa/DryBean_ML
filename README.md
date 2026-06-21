# DryBean_ML

基于 Dry Bean Dataset 脏数据版本构建的多分类机器学习项目，覆盖数据质量分析、数据预处理、模型训练、结果评估、鲁棒性分析以及 Streamlit 可视化展示。

[GitHub 仓库链接](https://github.com/Alberttaa/DryBean_ML)

## 项目简介

本项目将 Dry Bean 数据集的完整实验流程组织成一个可复现的机器学习工程，主要包括：

- 数据质量分析
- 数据清洗与特征工程
- 多模型训练与对比
- Loss、过拟合与鲁棒性分析
- 基于 Streamlit 的结果展示页面

整个项目通过统一入口 [main.py](main.py) 进行调度。

## 已实现模型

| 模型 | 类型 | 说明 |
| --- | --- | --- |
| Logistic Regression | 基础模型 | 线性分类基线 |
| KNN | 基础模型 | 基于距离的近邻分类 |
| SVM | 基础模型 | 基于间隔的分类模型 |
| MLP | 基础模型 | 多层感知机分类器 |
| Random Forest | 拓展模型 | 集成树模型 |
| XGBoost | 拓展模型 | 梯度提升树模型 |

## 数据集说明

项目使用老师提供的脏数据训练集、验证集和测试集：

```text
data/raw/Dry_Bean_Dataset_Dirty_train.csv
data/raw/Dry_Bean_Dataset_Dirty_val.csv
data/raw/Dry_Bean_Dataset_Dirty_test.csv
```

标签共包含 7 个类别：

```text
BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA
```

原始脏数据中主要包含以下问题：

- 缺失值
- 非法字符，如 `?`
- 单位污染，如 `0.8194 cm`
- 标签污染，如 `D3RMAS0N`
- 物理异常值，如 `Area <= 0`
- 训练集重复样本

## 项目结构

```text
DryBean_ML/
|-- app/
|   `-- app.py
|-- data/
|   |-- raw/
|   `-- processed/
|-- results/
|   |-- figures/
|   |   `-- confusion_matrices/
|   `-- metrics/
|-- saved_models/
|-- src/
|   |-- algorithms/
|   |-- evaluation/
|   |-- training/
|   |-- data_analysis.py
|   `-- preprocess.py
|-- .gitignore
|-- main.py
|-- README.md
`-- requirements.txt
```

## 关键结果

结果摘要来自 [results/metrics/accuracy_summary.csv](results/metrics/accuracy_summary.csv)：

| 模型 | 测试集准确率 | Macro F1 | 训练集与测试集差距 |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.9222 | 0.9327 | 0.0026 |
| KNN | 0.9247 | 0.9351 | 0.0753 |
| SVM | 0.9361 | 0.9459 | 0.0006 |
| MLP | 0.9266 | 0.9360 | 0.0040 |
| Random Forest | 0.9214 | 0.9320 | 0.0786 |
| XGBoost | 0.9255 | 0.9369 | 0.0281 |

简要结论：

- `SVM` 当前取得了最好的测试集准确率和 Macro F1。
- `Logistic Regression` 作为线性基线表现稳定。
- `KNN` 与 `Random Forest` 的训练测试差距较大，过拟合迹象更明显。

## 可视化预览

### 准确率对比图

![准确率对比图](results/figures/accuracy_comparison.png)

### 鲁棒性平均下降图

![鲁棒性平均下降图](results/figures/robustness_average_drop.png)

## 预处理流程

预处理阶段主要包含以下步骤：

1. 清洗标签污染和大小写不一致问题
2. 将非数值字符转换为缺失值
3. 修正物理意义不合理的异常值
4. 删除训练集中的重复样本
5. 使用训练集统计量填补缺失值
6. 对 7 个类别进行标签编码
7. 使用 `StandardScaler` 做特征标准化

生成的数据文件包括：

```text
data/processed/X_train.csv
data/processed/X_val.csv
data/processed/X_test.csv
data/processed/y_train.csv
data/processed/y_val.csv
data/processed/y_test.csv
```

## 快速开始

安装依赖：

```bash
python -m pip install -r requirements.txt
```

运行数据分析：

```bash
python main.py --mode analyze
```

运行数据预处理：

```bash
python main.py --mode preprocess
```

训练全部模型：

```bash
python main.py --mode train_all
```

评估全部模型：

```bash
python main.py --mode evaluate
```

运行 Loss 分析：

```bash
python main.py --mode loss_analysis
```

运行鲁棒性实验：

```bash
python main.py --mode robustness
```

启动 Streamlit 页面：

```bash
python -m streamlit run app/app.py
```

## 主要输出文件

主要指标文件：

```text
results/metrics/accuracy_summary.csv
results/metrics/speed_summary.csv
results/metrics/overfit_summary.csv
results/metrics/loss_analysis_summary.csv
results/metrics/robustness_summary.csv
results/metrics/classification_reports.json
```

主要图表文件：

```text
results/figures/accuracy_comparison.png
results/figures/loss_curve_comparison.png
results/figures/speed_comparison.png
results/figures/overfit_comparison.png
results/figures/robustness_average_drop.png
results/figures/confusion_matrices/
```

## Streamlit 展示页面

[app/app.py](app/app.py) 会读取已经生成好的实验结果，并集中展示：

- 项目概览
- 数据质量问题
- 预处理流程
- 模型对比结果
- 混淆矩阵
- Loss 分析
- 推理速度对比
- 过拟合分析
- 鲁棒性分析

## Git 跟踪说明

- 当前仓库保留了源码、处理后的数据、指标文件和结果图，便于复现与展示。
- `saved_models/` 下的训练模型文件已在 `.gitignore` 中忽略，避免仓库过大。
- Python 缓存、虚拟环境、日志和本地编辑器配置文件也已忽略。

## 后续可扩展方向

- 超参数搜索
- PCA 降维对比实验
- 特征重要性分析
- SHAP 可解释性分析
- 在线演示页面部署
