from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
METRIC_DIR = Path("results/metrics")
SAVED_MODEL_DIR = Path("saved_models")


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

LABEL_TO_ID = {label: idx for idx, label in enumerate(STANDARD_CLASSES)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def read_csv_safely(path: Path) -> pd.DataFrame:
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
    dermason / DERMASON / D3RMAS0N / DERMASON 空格
    都修正为标准大写标签。
    """
    if pd.isna(label):
        return np.nan

    text = str(label).strip().upper()
    text = text.replace("0", "O").replace("3", "E")
    text = re.sub(r"[^A-Z]", "", text)

    return text


def clean_numeric_value(value):
    """
    清洗数值字段：
    - '?' 变 NaN
    - '0.8194 cm' 提取成 0.8194
    - 字符串数字转 float
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


def clean_one_split(df: pd.DataFrame, split: str):
    df = df.copy()

    if "Class" not in df.columns:
        raise ValueError(f"{split} 数据中没有 Class 标签列。")

    feature_cols = [c for c in df.columns if c != "Class"]

    before_rows = df.shape[0]
    before_duplicates = int(df.duplicated().sum())

    # 1. 清洗标签
    df["Class"] = df["Class"].apply(normalize_label)

    # 无法识别的标签删除
    invalid_label_mask = ~df["Class"].isin(STANDARD_CLASSES)
    invalid_label_count = int(invalid_label_mask.sum())
    df = df.loc[~invalid_label_mask].copy()

    # 2. 清洗所有数值特征
    for col in feature_cols:
        df[col] = df[col].apply(clean_numeric_value)

    # 3. 物理异常值处理
    # 面积、周长、轴长、形状因子等理论上都应为正数。
    # 为了稳妥，这里将所有数值特征中 <=0 的值先视作异常值，转为缺失值。
    non_positive_count = 0
    for col in feature_cols:
        mask = df[col] <= 0
        non_positive_count += int(mask.sum())
        df.loc[mask, col] = np.nan

    # 4. 训练集删除重复行，验证集和测试集保持样本结构
    duplicated_removed = 0
    if split == "train":
        duplicated_removed = int(df.duplicated().sum())
        df = df.drop_duplicates().copy()

    after_rows = df.shape[0]

    report = {
        "split": split,
        "before_rows": int(before_rows),
        "after_rows": int(after_rows),
        "before_duplicates": before_duplicates,
        "duplicated_removed": duplicated_removed,
        "invalid_label_removed": invalid_label_count,
        "non_positive_to_nan": non_positive_count,
    }

    return df, report


def run_preprocess():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = {
        split: read_csv_safely(path)
        for split, path in RAW_FILES.items()
    }

    cleaned_data = {}
    reports = []

    for split, df in raw_data.items():
        cleaned_df, report = clean_one_split(df, split)
        cleaned_data[split] = cleaned_df
        reports.append(report)

    train_df = cleaned_data["train"]
    val_df = cleaned_data["val"]
    test_df = cleaned_data["test"]

    feature_cols = [c for c in train_df.columns if c != "Class"]

    # 5. 使用训练集的中位数填补缺失值，避免数据泄露
    train_medians = train_df[feature_cols].median(numeric_only=True)

    for split, df in cleaned_data.items():
        df[feature_cols] = df[feature_cols].fillna(train_medians)

    # 6. 标签编码
    for split, df in cleaned_data.items():
        df["Class_ID"] = df["Class"].map(LABEL_TO_ID)

    # 7. 标准化：只在训练集 fit，再应用到 val/test，避免数据泄露
    scaler = StandardScaler()
    X_train = scaler.fit_transform(cleaned_data["train"][feature_cols])
    X_val = scaler.transform(cleaned_data["val"][feature_cols])
    X_test = scaler.transform(cleaned_data["test"][feature_cols])

    y_train = cleaned_data["train"]["Class_ID"].to_numpy()
    y_val = cleaned_data["val"]["Class_ID"].to_numpy()
    y_test = cleaned_data["test"]["Class_ID"].to_numpy()

    # 8. 保存处理后的数据
    pd.DataFrame(X_train, columns=feature_cols).to_csv(PROCESSED_DIR / "X_train.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(X_val, columns=feature_cols).to_csv(PROCESSED_DIR / "X_val.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(X_test, columns=feature_cols).to_csv(PROCESSED_DIR / "X_test.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"Class_ID": y_train}).to_csv(PROCESSED_DIR / "y_train.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"Class_ID": y_val}).to_csv(PROCESSED_DIR / "y_val.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"Class_ID": y_test}).to_csv(PROCESSED_DIR / "y_test.csv", index=False, encoding="utf-8-sig")

    # 同时保存未标准化但已清洗的数据，方便论文分析
    for split, df in cleaned_data.items():
        df.to_csv(PROCESSED_DIR / f"{split}_cleaned_unscaled.csv", index=False, encoding="utf-8-sig")

    # 保存标签映射和特征名
    with open(PROCESSED_DIR / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(LABEL_TO_ID, f, ensure_ascii=False, indent=4)

    with open(PROCESSED_DIR / "id_to_label.json", "w", encoding="utf-8") as f:
        json.dump(ID_TO_LABEL, f, ensure_ascii=False, indent=4)

    with open(PROCESSED_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=4)

    # 保存预处理器
    joblib.dump(scaler, SAVED_MODEL_DIR / "standard_scaler.pkl")
    joblib.dump(train_medians, SAVED_MODEL_DIR / "train_medians.pkl")

    # 9. 保存处理报告
    preprocess_report = pd.DataFrame(reports)
    preprocess_report.to_csv(METRIC_DIR / "preprocess_report.csv", index=False, encoding="utf-8-sig")

    missing_after = []
    for split, df in cleaned_data.items():
        for col in feature_cols:
            missing_after.append({
                "split": split,
                "column": col,
                "missing_after_preprocess": int(df[col].isna().sum())
            })

    pd.DataFrame(missing_after).to_csv(
        METRIC_DIR / "missing_after_preprocess.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("数据预处理完成。")
    print(f"处理后的数据保存到：{PROCESSED_DIR}")
    print(f"预处理报告保存到：{METRIC_DIR / 'preprocess_report.csv'}")