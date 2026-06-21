from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RAW_DIR = Path("data/raw")
METRIC_DIR = Path("results/metrics")
FIGURE_DIR = Path("results/figures")


RAW_FILES = {
    "train": RAW_DIR / "Dry_Bean_Dataset_Dirty_train.csv",
    "val": RAW_DIR / "Dry_Bean_Dataset_Dirty_val.csv",
    "test": RAW_DIR / "Dry_Bean_Dataset_Dirty_test.csv",
}


STANDARD_CLASSES = [
    "BARBUNYA",
    "BOMBAY",
    "CALI",
    "DERMASON",
    "HOROZ",
    "SEKER",
    "SIRA",
]


def read_csv_safely(path: Path) -> pd.DataFrame:
    """尽量兼容不同编码读取 CSV。"""
    if not path.exists():
        raise FileNotFoundError(f"没有找到文件：{path}")

    for encoding in ["utf-8", "utf-8-sig", "gbk"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(path)


def normalize_label(label):
    """
    统一标签：
    - 去掉首尾空格
    - 转大写
    - 修正常见污染：0 -> O, 3 -> E
    """
    if pd.isna(label):
        return np.nan

    text = str(label).strip().upper()
    text = text.replace("0", "O").replace("3", "E")

    # 去掉非字母字符
    text = re.sub(r"[^A-Z]", "", text)

    return text


def clean_numeric_value(value):
    """
    将脏数值转为 float：
    - '?' 转为 NaN
    - '0.8194 cm' 提取为 0.8194
    - 普通字符串数值转 float
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text in ["", "?", "nan", "NaN", "None", "NULL", "null"]:
        return np.nan

    match = re.search(r"-?\d+\.?\d*", text)
    if match:
        return float(match.group())

    return np.nan


def count_missing_like(series: pd.Series) -> int:
    """统计空值、?、空字符串等缺失形式。"""
    missing_count = series.isna().sum()

    str_series = series.astype(str).str.strip()
    special_missing = str_series.isin(["", "?", "nan", "NaN", "None", "NULL", "null"]).sum()

    return int(max(missing_count, special_missing))


def run_data_analysis():
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    dataframes = {}
    for split, path in RAW_FILES.items():
        dataframes[split] = read_csv_safely(path)

    quality_rows = []
    feature_rows = []
    class_rows = []
    missing_rows = []

    for split, df in dataframes.items():
        if "Class" not in df.columns:
            raise ValueError(f"{split} 数据中没有 Class 标签列，请检查文件。")

        feature_cols = [c for c in df.columns if c != "Class"]

        # 标签污染统计
        raw_labels = df["Class"].astype(str)
        normalized_labels = df["Class"].apply(normalize_label)
        label_pollution = (raw_labels.str.strip().str.upper() != normalized_labels).sum()

        # Area 异常值统计
        area_non_positive = 0
        if "Area" in df.columns:
            area_numeric = df["Area"].apply(clean_numeric_value)
            area_non_positive = int((area_numeric <= 0).sum())

        # 重复行统计
        duplicated_rows = int(df.duplicated().sum())

        # 指定列缺失统计
        perimeter_missing = count_missing_like(df["Perimeter"]) if "Perimeter" in df.columns else 0
        solidity_missing = count_missing_like(df["Solidity"]) if "Solidity" in df.columns else 0

        quality_rows.append({
            "dataset": split,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "duplicated_rows": duplicated_rows,
            "area_non_positive": area_non_positive,
            "label_pollution_count": int(label_pollution),
            "perimeter_missing": perimeter_missing,
            "solidity_missing": solidity_missing,
        })

        # 每个特征的数据类型、缺失值、非法数值情况
        for col in df.columns:
            missing_like = count_missing_like(df[col])
            invalid_numeric = ""

            if col != "Class":
                numeric_col = df[col].apply(clean_numeric_value)
                invalid_numeric = int(numeric_col.isna().sum() - missing_like)
                invalid_numeric = max(invalid_numeric, 0)

            feature_rows.append({
                "dataset": split,
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_like_count": int(missing_like),
                "invalid_numeric_count": invalid_numeric,
                "unique_count": int(df[col].nunique(dropna=True)),
            })

        # 类别分布，使用修正后的类别便于观察真实分布
        class_count = normalized_labels.value_counts(dropna=False)
        for cls, count in class_count.items():
            class_rows.append({
                "dataset": split,
                "class": cls,
                "count": int(count),
            })

        # 缺失值统计
        for col in df.columns:
            missing_rows.append({
                "dataset": split,
                "column": col,
                "missing_like_count": count_missing_like(df[col]),
            })

    quality_df = pd.DataFrame(quality_rows)
    feature_df = pd.DataFrame(feature_rows)
    class_df = pd.DataFrame(class_rows)
    missing_df = pd.DataFrame(missing_rows)

    quality_df.to_csv(METRIC_DIR / "data_quality_report.csv", index=False, encoding="utf-8-sig")
    feature_df.to_csv(METRIC_DIR / "feature_overview.csv", index=False, encoding="utf-8-sig")
    class_df.to_csv(METRIC_DIR / "class_distribution.csv", index=False, encoding="utf-8-sig")
    missing_df.to_csv(METRIC_DIR / "missing_values.csv", index=False, encoding="utf-8-sig")

    # 图1：类别分布
    class_pivot = class_df.pivot_table(
        index="class",
        columns="dataset",
        values="count",
        fill_value=0,
        aggfunc="sum"
    )

    class_pivot = class_pivot.reindex(STANDARD_CLASSES)
    ax = class_pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution.png", dpi=300)
    plt.close()

    # 图2：缺失值统计，只画有缺失的列
    missing_total = missing_df.groupby("column")["missing_like_count"].sum()
    missing_total = missing_total[missing_total > 0].sort_values(ascending=False)

    if not missing_total.empty:
        ax = missing_total.plot(kind="bar", figsize=(9, 5))
        ax.set_title("Missing Values by Column")
        ax.set_xlabel("Column")
        ax.set_ylabel("Missing-like Count")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "missing_values.png", dpi=300)
        plt.close()

    # 保存数据分析摘要，方便后面 Streamlit 和论文读取
    summary = {
        "total_rows": int(sum(df.shape[0] for df in dataframes.values())),
        "total_columns": int(next(iter(dataframes.values())).shape[1]),
        "feature_count": int(next(iter(dataframes.values())).shape[1] - 1),
        "target_column": "Class",
        "standard_classes": STANDARD_CLASSES,
        "raw_files": {k: str(v) for k, v in RAW_FILES.items()},
    }

    with open(METRIC_DIR / "data_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print("数据分析完成。")
    print(f"结果表格保存到：{METRIC_DIR}")
    print(f"结果图片保存到：{FIGURE_DIR}")