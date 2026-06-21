from pathlib import Path
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from src.algorithms.logistic_regression import build_model as build_logistic_regression
from src.algorithms.knn import build_model as build_knn
from src.algorithms.svm import build_model as build_svm
from src.algorithms.mlp import build_model as build_mlp
from src.algorithms.random_forest import build_model as build_random_forest
from src.algorithms.xgboost_model import build_model as build_xgboost


PROCESSED_DIR = Path("data/processed")
METRIC_DIR = Path("results/metrics")
FIGURE_DIR = Path("results/figures")


ALGORITHMS = {
    "logistic_regression": build_logistic_regression,
    "knn": build_knn,
    "svm": build_svm,
    "mlp": build_mlp,
    "random_forest": build_random_forest,
    "xgboost": build_xgboost,
}


NOISE_TYPES = [
    "gaussian",
    "uniform",
    "feature_dropout",
    "label_noise",
]

NOISE_STRENGTHS = [
    0.05,
    0.10,
    0.20,
]


def load_processed_data():
    """读取预处理后的数据。"""
    required_files = [
        PROCESSED_DIR / "X_train.csv",
        PROCESSED_DIR / "X_test.csv",
        PROCESSED_DIR / "y_train.csv",
        PROCESSED_DIR / "y_test.csv",
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"没有找到 {file}，请先运行：python main.py --mode preprocess"
            )

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")

    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["Class_ID"].to_numpy()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")["Class_ID"].to_numpy()

    return X_train, X_test, y_train, y_test


def add_gaussian_noise(X, strength, rng):
    """
    高斯噪声：
    因为 X 已经标准化，所以 strength 可以理解为标准化空间中的扰动强度。
    """
    noise = rng.normal(loc=0.0, scale=strength, size=X.shape)
    return X + noise


def add_uniform_noise(X, strength, rng):
    """
    均匀噪声：
    在 [-strength, strength] 范围内加入随机扰动。
    """
    noise = rng.uniform(low=-strength, high=strength, size=X.shape)
    return X + noise


def add_feature_dropout_noise(X, strength, rng):
    """
    特征缺失噪声：
    随机把一部分训练特征置为 0。
    由于标准化后 0 接近训练集均值，所以可以模拟缺失后均值填充。
    """
    X_noisy = X.copy()
    mask = rng.random(size=X_noisy.shape) < strength
    X_noisy[mask] = 0.0
    return X_noisy


def add_label_noise(y, strength, rng, num_classes=7):
    """
    标签噪声：
    随机选取一部分训练样本，把它们的标签替换成其他类别。
    """
    y_noisy = y.copy()
    sample_count = len(y_noisy)
    noise_count = int(sample_count * strength)

    if noise_count <= 0:
        return y_noisy

    noise_indices = rng.choice(sample_count, size=noise_count, replace=False)

    for idx in noise_indices:
        old_label = y_noisy[idx]
        candidate_labels = [i for i in range(num_classes) if i != old_label]
        y_noisy[idx] = rng.choice(candidate_labels)

    return y_noisy


def make_noisy_training_data(X_train, y_train, noise_type, strength, random_state=42):
    """
    根据噪声类型和强度生成带噪声训练数据。
    """
    rng = np.random.default_rng(random_state)

    X_array = X_train.to_numpy(dtype=float)
    y_array = y_train.copy()

    if noise_type == "gaussian":
        X_noisy = add_gaussian_noise(X_array, strength, rng)
        y_noisy = y_array

    elif noise_type == "uniform":
        X_noisy = add_uniform_noise(X_array, strength, rng)
        y_noisy = y_array

    elif noise_type == "feature_dropout":
        X_noisy = add_feature_dropout_noise(X_array, strength, rng)
        y_noisy = y_array

    elif noise_type == "label_noise":
        X_noisy = X_array
        y_noisy = add_label_noise(y_array, strength, rng)

    else:
        raise ValueError(f"未知噪声类型：{noise_type}")

    X_noisy = pd.DataFrame(X_noisy, columns=X_train.columns)
    return X_noisy, y_noisy


def load_baseline_accuracy():
    """
    读取原始干净数据上的测试精度。
    如果没有 accuracy_summary.csv，就返回空字典。
    """
    accuracy_file = METRIC_DIR / "accuracy_summary.csv"

    if not accuracy_file.exists():
        print("没有找到 accuracy_summary.csv，将在鲁棒性实验中重新计算 baseline。")
        return {}

    df = pd.read_csv(accuracy_file)
    return dict(zip(df["model"], df["test_accuracy"]))


def train_and_test_model(model_name, build_func, X_train, y_train, X_test, y_test):
    """
    训练一个模型并在测试集上评价。
    返回测试精度和训练时间。
    """
    model = build_func()

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time

    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    return test_acc, train_time


def plot_average_robustness(summary_df):
    """
    保存平均精度下降柱状图。
    accuracy_drop 越小，说明模型整体鲁棒性越好。
    """
    avg_df = (
        summary_df
        .groupby("model", as_index=False)["accuracy_drop"]
        .mean()
        .sort_values("accuracy_drop", ascending=True)
    )

    avg_df.to_csv(
        METRIC_DIR / "robustness_average_drop.csv",
        index=False,
        encoding="utf-8-sig"
    )

    plt.figure(figsize=(9, 5))
    bars = plt.bar(avg_df["model"], avg_df["accuracy_drop"])

    plt.title("Average Accuracy Drop under Noisy Training Data")
    plt.xlabel("Model")
    plt.ylabel("Average Accuracy Drop")
    plt.xticks(rotation=30, ha="right")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.0005,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "robustness_average_drop.png", dpi=300)
    plt.close()


def plot_robustness_curves(summary_df):
    """
    按噪声类型分别画鲁棒性曲线。
    这里固定纵轴为 0.8~1.0，用于放大高精度区间内的细微差异。
    """
    for noise_type in NOISE_TYPES:
        sub_df = summary_df[summary_df["noise_type"] == noise_type].copy()

        if sub_df.empty:
            continue

        plt.figure(figsize=(9, 5))

        for model_name in ALGORITHMS.keys():
            model_df = sub_df[sub_df["model"] == model_name].copy()

            if model_df.empty:
                continue

            model_df = model_df.sort_values("noise_strength")

            plt.plot(
                model_df["noise_strength"],
                model_df["test_accuracy"],
                marker="o",
                linewidth=1.8,
                label=model_name
            )

            for _, row in model_df.iterrows():
                y_text = min(row["test_accuracy"] + 0.004, 0.995)
                plt.text(
                    row["noise_strength"],
                    y_text,
                    f"{row['test_accuracy']:.3f}",
                    fontsize=7,
                    ha="center"
                )

        plt.title(f"Robustness Curve - {noise_type} (Y-axis: 0.8-1.0)")
        plt.xlabel("Noise Strength")
        plt.ylabel("Test Accuracy")

        # 关键设置：纵轴只显示 0.8 到 1.0
        plt.ylim(0.8, 1.0)
        plt.yticks(np.arange(0.80, 1.001, 0.025))

        # 横轴只显示三个噪声强度
        plt.xticks(NOISE_STRENGTHS)

        plt.grid(alpha=0.3)
        plt.legend(fontsize=8, loc="lower left")
        plt.tight_layout()

        output_path = FIGURE_DIR / f"robustness_curve_{noise_type}.png"
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"已保存鲁棒性曲线：{output_path}")


def plot_robustness_from_existing_summary():
    """
    只根据已有 robustness_summary.csv 重新画图，不重新训练模型。
    如果你只是想改图，不想重新跑 72 次训练，可以用这个函数。
    """
    summary_file = METRIC_DIR / "robustness_summary.csv"

    if not summary_file.exists():
        raise FileNotFoundError(
            f"没有找到 {summary_file}，请先运行：python main.py --mode robustness"
        )

    summary_df = pd.read_csv(summary_file)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    plot_average_robustness(summary_df)
    plot_robustness_curves(summary_df)

    print("已根据已有 robustness_summary.csv 重新生成鲁棒性图表。")


def run_robustness():
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_processed_data()

    baseline_accuracy = load_baseline_accuracy()

    rows = []

    print("开始鲁棒性实验。")
    print("说明：对训练集加入噪声，模型重新训练，然后在干净测试集上评估。")
    print("-" * 70)

    for model_name, build_func in ALGORITHMS.items():
        print(f"处理模型：{model_name}")

        if model_name in baseline_accuracy:
            baseline_acc = float(baseline_accuracy[model_name])
        else:
            baseline_acc, _ = train_and_test_model(
                model_name,
                build_func,
                X_train,
                y_train,
                X_test,
                y_test,
            )

        for noise_type in NOISE_TYPES:
            for strength in NOISE_STRENGTHS:
                print(f"  噪声类型={noise_type}, 强度={strength}")

                X_noisy, y_noisy = make_noisy_training_data(
                    X_train,
                    y_train,
                    noise_type=noise_type,
                    strength=strength,
                    random_state=42,
                )

                test_acc, train_time = train_and_test_model(
                    model_name,
                    build_func,
                    X_noisy,
                    y_noisy,
                    X_test,
                    y_test,
                )

                accuracy_drop = baseline_acc - test_acc
                relative_drop = accuracy_drop / baseline_acc if baseline_acc > 0 else 0

                rows.append({
                    "model": model_name,
                    "noise_type": noise_type,
                    "noise_strength": strength,
                    "baseline_accuracy": baseline_acc,
                    "test_accuracy": test_acc,
                    "accuracy_drop": accuracy_drop,
                    "relative_drop": relative_drop,
                    "train_time_seconds": train_time,
                })

                print(
                    f"    test_acc={test_acc:.4f}, "
                    f"drop={accuracy_drop:.4f}, "
                    f"time={train_time:.2f}s"
                )

        print("-" * 70)

    summary_df = pd.DataFrame(rows)

    summary_df.to_csv(
        METRIC_DIR / "robustness_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    plot_average_robustness(summary_df)
    plot_robustness_curves(summary_df)

    print("鲁棒性实验完成。")
    print(f"鲁棒性结果保存到：{METRIC_DIR / 'robustness_summary.csv'}")
    print(f"鲁棒性平均下降表保存到：{METRIC_DIR / 'robustness_average_drop.csv'}")
    print(f"鲁棒性图表保存到：{FIGURE_DIR}")


if __name__ == "__main__":
    # 直接运行这个文件时，只根据已有结果重新画图，不重新训练。
    plot_robustness_from_existing_summary()