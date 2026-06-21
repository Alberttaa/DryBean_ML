from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


PROCESSED_DIR = Path("data/processed")
SAVED_MODEL_DIR = Path("saved_models")
METRIC_DIR = Path("results/metrics")
FIGURE_DIR = Path("results/figures")
CM_DIR = FIGURE_DIR / "confusion_matrices"


ALGORITHMS = [
    "logistic_regression",
    "knn",
    "svm",
    "mlp",
    "random_forest",
    "xgboost",
]


def load_processed_data():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_val = pd.read_csv(PROCESSED_DIR / "X_val.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")

    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["Class_ID"].to_numpy()
    y_val = pd.read_csv(PROCESSED_DIR / "y_val.csv")["Class_ID"].to_numpy()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")["Class_ID"].to_numpy()

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_label_names():
    label_file = PROCESSED_DIR / "id_to_label.json"

    if label_file.exists():
        with open(label_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        # json 的 key 会变成字符串，需要转回 int 后排序
        labels = [mapping[str(i)] for i in range(len(mapping))]
        return labels

    return [str(i) for i in range(7)]


def test_inference_speed(model, X_test, repeat=5):
    """
    推理速度测试。
    多次重复预测，取平均时间。
    返回：
    - 总耗时
    - 单样本平均耗时
    - 每秒预测样本数
    """
    # 预热一次，避免第一次调用偏慢
    _ = model.predict(X_test)

    times = []
    for _ in range(repeat):
        start_time = time.perf_counter()
        _ = model.predict(X_test)
        elapsed = time.perf_counter() - start_time
        times.append(elapsed)

    avg_time = float(np.mean(times))
    per_sample_time = avg_time / len(X_test)
    samples_per_second = len(X_test) / avg_time

    return avg_time, per_sample_time, samples_per_second


def save_confusion_matrix_figure(cm, labels, model_name):
    """保存混淆矩阵图片，不使用 UI 显示。"""
    CM_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm)

    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(CM_DIR / f"{model_name}_confusion_matrix.png", dpi=300)
    plt.close()


def plot_accuracy_comparison(accuracy_df):
    """保存测试集精度对比图，同时保存一张局部放大图。"""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = accuracy_df.sort_values("test_accuracy", ascending=False)

    # 1. 原始完整范围图：0 到 1
    plt.figure(figsize=(9, 5))
    bars = plt.bar(plot_df["model"], plot_df["test_accuracy"])
    plt.title("Test Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=30, ha="right")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "accuracy_comparison.png", dpi=300)
    plt.close()

    # 2. 局部放大图：自动放大高精度区间
    min_acc = plot_df["test_accuracy"].min()
    max_acc = plot_df["test_accuracy"].max()

    y_min = max(0.0, min_acc - 0.01)
    y_max = min(1.0, max_acc + 0.01)

    # 如果模型都在 0.9 以上，就固定放大到 0.88~1.0 或 0.90~1.0
    if min_acc >= 0.85:
        y_min = max(0.85, min_acc - 0.01)
        y_max = 1.0

    plt.figure(figsize=(9, 5))
    bars = plt.bar(plot_df["model"], plot_df["test_accuracy"])
    plt.title("Test Accuracy Comparison (Zoomed)")
    plt.xlabel("Model")
    plt.ylabel("Test Accuracy")
    plt.ylim(y_min, y_max)
    plt.xticks(rotation=30, ha="right")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.001,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "accuracy_comparison_zoomed.png", dpi=300)
    plt.close()
    
    """保存测试集精度对比图。"""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = accuracy_df.sort_values("test_accuracy", ascending=False)

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["model"], plot_df["test_accuracy"])
    plt.title("Test Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "accuracy_comparison.png", dpi=300)
    plt.close()


def plot_speed_comparison(speed_df):
    """保存推理速度对比图。"""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = speed_df.sort_values("samples_per_second", ascending=False)

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["model"], plot_df["samples_per_second"])
    plt.title("Inference Speed Comparison")
    plt.xlabel("Model")
    plt.ylabel("Samples per Second")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "speed_comparison.png", dpi=300)
    plt.close()


def plot_overfit_comparison(accuracy_df):
    """保存过拟合差距图：训练集精度 - 测试集精度。"""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = accuracy_df.sort_values("train_test_gap", ascending=False)

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["model"], plot_df["train_test_gap"])
    plt.title("Overfitting Gap Comparison")
    plt.xlabel("Model")
    plt.ylabel("Train Accuracy - Test Accuracy")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "overfit_comparison.png", dpi=300)
    plt.close()


def plot_loss_curve():
    """根据训练阶段保存的 loss_history.csv 绘制 loss 曲线。"""
    loss_file = METRIC_DIR / "loss_history.csv"
    if not loss_file.exists():
        print("没有找到 loss_history.csv，跳过 loss 曲线绘制。")
        return

    loss_df = pd.read_csv(loss_file)

    if loss_df.empty:
        print("loss_history.csv 为空，跳过 loss 曲线绘制。")
        return

    plt.figure(figsize=(10, 6))

    for model_name in loss_df["model"].unique():
        sub_df = loss_df[loss_df["model"] == model_name]

        # XGBoost 有 train / validation 两条曲线
        if "dataset" in sub_df.columns:
            for dataset_name in sub_df["dataset"].dropna().unique():
                curve_df = sub_df[sub_df["dataset"] == dataset_name]
                plt.plot(
                    curve_df["epoch"],
                    curve_df["loss"],
                    label=f"{model_name}-{dataset_name}"
                )
        else:
            plt.plot(sub_df["epoch"], sub_df["loss"], label=model_name)

    plt.title("Loss Curve Comparison")
    plt.xlabel("Epoch / Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "loss_curve_comparison.png", dpi=300)
    plt.close()


def run_evaluate():
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    CM_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    labels = load_label_names()

    accuracy_rows = []
    speed_rows = []
    reports = {}

    print("开始测试所有算法...")
    print("-" * 60)

    for model_name in ALGORITHMS:
        model_path = SAVED_MODEL_DIR / f"{model_name}.pkl"

        if not model_path.exists():
            print(f"未找到模型文件：{model_path}，跳过 {model_name}")
            continue

        model = joblib.load(model_path)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)

        precision_macro = precision_score(y_test, test_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_test, test_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, test_pred, average="macro", zero_division=0)

        train_test_gap = train_acc - test_acc

        avg_time, per_sample_time, samples_per_second = test_inference_speed(model, X_test)

        accuracy_rows.append({
            "model": model_name,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "train_test_gap": train_test_gap,
        })

        speed_rows.append({
            "model": model_name,
            "test_samples": len(X_test),
            "avg_inference_time_seconds": avg_time,
            "per_sample_time_seconds": per_sample_time,
            "samples_per_second": samples_per_second,
        })

        report = classification_report(
            y_test,
            test_pred,
            target_names=labels,
            zero_division=0,
            output_dict=True,
        )
        reports[model_name] = report

        cm = confusion_matrix(y_test, test_pred)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(
            CM_DIR / f"{model_name}_confusion_matrix.csv",
            encoding="utf-8-sig"
        )
        save_confusion_matrix_figure(cm, labels, model_name)

        print(
            f"{model_name} 测试完成："
            f"test_acc={test_acc:.4f}, "
            f"f1_macro={f1_macro:.4f}, "
            f"speed={samples_per_second:.2f} samples/s"
        )
        print("-" * 60)

    accuracy_df = pd.DataFrame(accuracy_rows)
    speed_df = pd.DataFrame(speed_rows)

    accuracy_df.to_csv(
        METRIC_DIR / "accuracy_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    speed_df.to_csv(
        METRIC_DIR / "speed_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 过拟合分析表可以直接由 accuracy 表生成一份单独文件
    overfit_df = accuracy_df[[
        "model",
        "train_accuracy",
        "test_accuracy",
        "train_test_gap"
    ]].copy()
    overfit_df.to_csv(
        METRIC_DIR / "overfit_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    with open(METRIC_DIR / "classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=4)

    if not accuracy_df.empty:
        plot_accuracy_comparison(accuracy_df)
        plot_overfit_comparison(accuracy_df)

    if not speed_df.empty:
        plot_speed_comparison(speed_df)

    plot_loss_curve()

    print("全部算法测试完成。")
    print(f"精度结果保存到：{METRIC_DIR / 'accuracy_summary.csv'}")
    print(f"速度结果保存到：{METRIC_DIR / 'speed_summary.csv'}")
    print(f"过拟合分析保存到：{METRIC_DIR / 'overfit_summary.csv'}")
    print(f"混淆矩阵保存到：{CM_DIR}")