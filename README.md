# Dry Bean Dataset 多分类机器学习工程项目

## 1. 项目简介

本项目基于 Dry Bean Dataset 的脏数据版本，完成一个包含 **数据分析、数据处理、多算法实验分析、系统展示、课程总结** 的完整机器学习工程项目。

项目目标是根据干豆样本的形态学特征，对豆类品种进行多分类预测。整个项目采用工程化文件夹结构组织代码，通过统一命令行入口 `main.py` 完成数据分析、数据预处理、模型训练、模型测试、Loss 曲线分析和鲁棒性实验。算法运行阶段不使用 UI 显示，所有实验结果统一保存为表格和图片，再由 Streamlit 展示界面读取结果进行可视化展示。

本项目实现了六种多分类算法：

| 算法                  | 中文名称   | 课程内容 | 说明                |
| ------------------- | ------ | ------ | ----------------- |
| Logistic Regression | 逻辑回归   | 是      | 线性多分类基准模型         |
| KNN                 | K近邻    | 是      | 基于距离度量和邻居投票的分类算法  |
| SVM                 | 支持向量机  | 是      | 基于最大间隔和核函数的分类算法   |
| MLP                 | 多层感知机  | 是      | 基于神经网络的非线性分类模型    |
| Random Forest       | 随机森林   | 否      | 集成学习算法，作为课堂外拓展算法  |
| XGBoost             | 极端梯度提升 | 否      | 梯度提升树算法，作为课堂外拓展算法 |

其中，Random Forest 和 XGBoost 是课堂上没有系统讲解的算法，用于满足课程项目中“至少包含一种课堂外算法”的要求。

---

## 2. 数据集说明

本项目使用的数据文件包括：

```text
data/raw/Dry_Bean_Dataset_Dirty_train.csv
data/raw/Dry_Bean_Dataset_Dirty_val.csv
data/raw/Dry_Bean_Dataset_Dirty_test.csv
```

数据集包含 16 个输入特征和 1 个标签列 `Class`。输入特征主要描述干豆图像中的面积、周长、长轴长度、短轴长度、偏心率、凸包面积、圆度、紧致度和形状因子等形态学信息。

标签类别包括 7 类：

```text
BARBUNYA
BOMBAY
CALI
DERMASON
HOROZ
SEKER
SIRA
```

本项目直接使用老师提供的 train、val、test 三份数据文件，不重新随机划分数据集，以保证实验流程稳定、可复现。

---

## 3. 项目结构

```text
DryBean_ML/
│
├── data/
│   ├── raw/                         # 原始脏数据
│   ├── processed/                   # 清洗和标准化后的数据
│   └── noise/                       # 噪声实验相关数据
│
├── saved_models/                    # 保存训练好的模型
│
├── src/
│   ├── data_analysis.py             # 数据分析模块
│   ├── preprocess.py                # 数据清洗与特征工程模块
│   ├── algorithms/                  # 六种分类算法定义
│   ├── training/                    # 模型训练模块
│   └── evaluation/                  # 测试、速度、鲁棒性、过拟合和 Loss 分析
│
├── results/
│   ├── metrics/                     # 实验结果表格
│   └── figures/                     # 实验可视化图表
│
├── app/
│   └── app.py                       # Streamlit 展示系统
│
├── reports/                         # 论文和截图材料
├── main.py                          # 统一命令行入口
├── requirements.txt                 # 项目依赖
├── README.md                        # 项目说明文档
└── .gitignore
```

---

## 4. 数据分析

数据分析阶段主要观察输入数据的基本情况和污染情况，包括：

* 样本数量和字段数量；
* 类别分布情况；
* 缺失值情况；
* 重复样本情况；
* 标签污染情况；
* 物理异常值情况。

在原始脏数据中，主要发现以下问题：

| 污染类型  | 具体表现                                           |
| ----- | ---------------------------------------------- |
| 缺失值   | `Perimeter` 和 `Solidity` 存在缺失                  |
| 非法字符  | `Solidity` 中存在 `?`                             |
| 单位污染  | `Compactness` 中存在类似 `0.8194 cm` 的字符串           |
| 标签污染  | 存在大小写、空格、数字替换等问题，例如 `D3RMAS0N`、`S3K3R`、`H0R0Z` |
| 物理异常值 | 验证集和测试集中存在 `Area <= 0` 的异常面积                   |
| 重复样本  | 训练集中存在重复行                                      |

数据分析结果保存在：

```text
results/metrics/data_quality_report.csv
results/metrics/feature_overview.csv
results/metrics/class_distribution.csv
results/metrics/missing_values.csv
```

对应图表保存在：

```text
results/figures/class_distribution.png
results/figures/missing_values.png
```

---

## 5. 数据处理与特征工程

本项目使用的数据处理方法包括：

| 处理步骤  | 说明                           |
| ----- | ---------------------------- |
| 标签清洗  | 统一大小写、去除空格、修正常见污染标签          |
| 数值清洗  | 将 `?` 转为缺失值，将带单位字符串提取为数值     |
| 异常值处理 | 将小于等于 0 的数值特征视为异常值并转为缺失值     |
| 重复值处理 | 删除训练集中的重复样本                  |
| 缺失值填充 | 使用训练集各特征中位数填充 train、val、test |
| 标签编码  | 将 7 个类别编码为 0 到 6             |
| 特征标准化 | 使用 StandardScaler 对数值特征进行标准化 |

为了避免数据泄露，缺失值填充参数和标准化参数均只在训练集上计算，然后应用到验证集和测试集。

预处理后生成的数据包括：

```text
data/processed/X_train.csv
data/processed/X_val.csv
data/processed/X_test.csv
data/processed/y_train.csv
data/processed/y_val.csv
data/processed/y_test.csv
```

另外，项目也保留了未标准化但已经清洗后的数据：

```text
data/processed/train_cleaned_unscaled.csv
data/processed/val_cleaned_unscaled.csv
data/processed/test_cleaned_unscaled.csv
```

这些文件主要用于论文中的数据处理前后对比。

---

## 6. 多算法实验分析

本项目从多个维度对六种多分类算法进行对比分析：

| 分析维度      | 说明                                              |
| --------- | ----------------------------------------------- |
| 测试集精度对比   | 比较不同算法在测试集上的 Accuracy、Precision、Recall、F1-score |
| Loss 曲线对比 | 对可记录训练过程的算法绘制 Loss 曲线                           |
| 推理速度对比    | 比较不同算法在测试集上的推理耗时和 samples/s                     |
| 鲁棒性对比     | 对训练数据加入不同类型和强度的噪声，观察测试精度下降情况                    |
| 过拟合分析     | 比较训练集精度和测试集精度差异                                 |
| 混淆矩阵分析    | 观察不同类别的分类错误情况                                   |

Loss 曲线说明：

* KNN 属于非训练型算法，不绘制 Loss 曲线；
* Random Forest 不以连续梯度下降方式训练，因此不绘制传统 Loss 曲线；
* Logistic Regression 和 Linear SVM 使用 SGD 版本记录训练损失；
* MLP 使用模型训练过程中的 `loss_curve_`；
* XGBoost 使用训练集和验证集上的 `mlogloss`。

---

## 7. 运行方式

### 7.1 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 7.2 数据分析

```bash
python main.py --mode analyze
```

### 7.3 数据清洗与特征工程

```bash
python main.py --mode preprocess
```

### 7.4 训练全部算法

```bash
python main.py --mode train_all
```

### 7.5 测试与结果评价

```bash
python main.py --mode evaluate
```

### 7.6 Loss 曲线分析

```bash
python main.py --mode loss_analysis
```

### 7.7 鲁棒性实验

```bash
python main.py --mode robustness
```

### 7.8 启动展示系统

```bash
python -m streamlit run app/app.py
```

---

## 8. 实验结果文件

主要结果表格保存在：

```text
results/metrics/
```

包括：

```text
data_quality_report.csv
preprocess_report.csv
train_summary.csv
accuracy_summary.csv
speed_summary.csv
overfit_summary.csv
loss_analysis_summary.csv
robustness_summary.csv
robustness_average_drop.csv
classification_reports.json
```

主要图表保存在：

```text
results/figures/
```

包括：

```text
class_distribution.png
missing_values.png
accuracy_comparison.png
accuracy_comparison_zoomed.png
loss_curve_comparison.png
loss_curve_comparison_zoomed.png
speed_comparison.png
overfit_comparison.png
robustness_average_drop.png
robustness_curve_gaussian.png
robustness_curve_uniform.png
robustness_curve_feature_dropout.png
robustness_curve_label_noise.png
```

---

## 9. Streamlit 展示系统

展示系统位于：

```text
app/app.py
```

启动命令：

```bash
python -m streamlit run app/app.py
```

展示界面包含以下内容：

* 项目概览；
* 数据集描述；
* 数据污染情况；
* 数据处理流程；
* 实现算法说明；
* 所有算法精度表；
* 测试集精度对比图；
* Loss 曲线；
* 推理速度对比；
* 过拟合分析；
* 鲁棒性分析；
* 课程总结。

展示系统只读取已经保存好的实验结果，不重新训练模型。算法训练、测试和鲁棒性实验均通过命令行完成。

---

## 10. GitHub 展示

项目上传 GitHub 后，可通过仓库首页 README 查看项目说明、运行方式和主要实验结果。

GitHub 仓库链接：

```text
TODO：在这里填写你的 GitHub 仓库链接
```
## 11. 项目总结

通过本项目，可以将课程中学习的数据清洗、特征工程、分类算法、模型评估和工程化组织方法串联起来。项目不仅实现了逻辑回归、KNN、SVM、MLP 等课堂算法，也进一步补充了随机森林和 XGBoost 等课堂外算法。通过从精度、Loss、速度、鲁棒性和过拟合等多个角度进行对比，可以更全面地理解不同算法在表格型多分类任务中的表现差异。

后续可以进一步加入：

* 参数网格搜索或贝叶斯调参；
* PCA 降维对比实验；
* 特征重要性分析；
* SHAP 可解释性分析；
* GitHub Pages 或云端部署页面。
