from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, hinge_loss
from xgboost import XGBClassifier


PROCESSED_DIR = Path("data/processed")
METRIC_DIR = Path("results/metrics")
FIGURE_DIR = Path("results/figures")


RANDOM_STATE = 42
NUM_CLASSES = 7
CLASSES = np.arange(NUM_CLASSES)


def load_processed_data():
    """读取预处理后的训练集和验证集。"""
    required_files = [
        PROCESSED_DIR / "X_train.csv",
        PROCESSED_DIR / "X_val.csv",
        PROCESSED_DIR / "y_train.csv",
        PROCESSED_DIR / "y_val.csv",
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"没有找到 {file}，请先运行：python main.py --mode preprocess"
            )

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_val = pd.read_csv(PROCESSED_DIR / "X_val.csv")

    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["Class_ID"].to_numpy()
    y_val = pd.read_csv(PROCESSED_DIR / "y_val.csv")["Class_ID"].to_numpy()

    return X_train, X_val, y_train, y_val


def shuffled_batches(X, y, batch_size, rng):
    """
    按 batch 打乱并返回小批量数据。
    X 可以是 numpy array。
    """
    indices = np.arange(len(X))
    rng.shuffle(indices)

    for start in range(0, len(X), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


def train_logistic_loss_curve(X_train, X_val, y_train, y_val, epochs=60, batch_size=256):
    """
    使用 SGDClassifier(loss='log_loss') 训练逻辑回归，
    记录每个 epoch 的训练集和验证集 log loss。
    """
    rng = np.random.default_rng(RANDOM_STATE)

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        learning_rate="optimal",
        random_state=RANDOM_STATE,
    )

    rows = []

    X_train_np = X_train.to_numpy(dtype=float)
    X_val_np = X_val.to_numpy(dtype=float)

    for epoch in range(1, epochs + 1):
        for X_batch, y_batch in shuffled_batches(X_train_np, y_train, batch_size, rng):
            if epoch == 1 and len(rows) == 0:
                model.partial_fit(X_batch, y_batch, classes=CLASSES)
            else:
                model.partial_fit(X_batch, y_batch)

        train_proba = model.predict_proba(X_train_np)
        val_proba = model.predict_proba(X_val_np)

        train_loss = log_loss(y_train, train_proba, labels=CLASSES)
        val_loss = log_loss(y_val, val_proba, labels=CLASSES)

        rows.append({
            "model": "Logistic Regression",
            "epoch": epoch,
            "dataset": "train",
            "loss_type": "log_loss",
            "loss": float(train_loss),
        })
        rows.append({
            "model": "Logistic Regression",
            "epoch": epoch,
            "dataset": "val",
            "loss_type": "log_loss",
            "loss": float(val_loss),
        })

    return rows


def train_linear_svm_loss_curve(X_train, X_val, y_train, y_val, epochs=60, batch_size=256):
    """
    使用 SGDClassifier(loss='hinge') 训练线性 SVM，
    记录每个 epoch 的训练集和验证集 hinge loss。

    注意：
    这里是为了 Loss 曲线分析额外训练的 Linear SVM，
    不替代主实验中的 RBF-SVM。
    """
    rng = np.random.default_rng(RANDOM_STATE)

    model = SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=0.0001,
        learning_rate="optimal",
        random_state=RANDOM_STATE,
    )

    rows = []

    X_train_np = X_train.to_numpy(dtype=float)
    X_val_np = X_val.to_numpy(dtype=float)

    for epoch in range(1, epochs + 1):
        for X_batch, y_batch in shuffled_batches(X_train_np, y_train, batch_size, rng):
            if epoch == 1 and len(rows) == 0:
                model.partial_fit(X_batch, y_batch, classes=CLASSES)
            else:
                model.partial_fit(X_batch, y_batch)

        train_decision = model.decision_function(X_train_np)
        val_decision = model.decision_function(X_val_np)

        train_loss = hinge_loss(y_train, train_decision, labels=CLASSES)
        val_loss = hinge_loss(y_val, val_decision, labels=CLASSES)

        rows.append({
            "model": "Linear SVM",
            "epoch": epoch,
            "dataset": "train",
            "loss_type": "hinge_loss",
            "loss": float(train_loss),
        })
        rows.append({
            "model": "Linear SVM",
            "epoch": epoch,
            "dataset": "val",
            "loss_type": "hinge_loss",
            "loss": float(val_loss),
        })

    return rows


def train_mlp_loss_curve(X_train, X_val, y_train, y_val):
    """
    训练 MLPClassifier，并记录自带的 loss_curve_。
    """
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    rows = []
    if hasattr(model, "loss_curve_"):
        for epoch, loss in enumerate(model.loss_curve_, start=1):
            rows.append({
                "model": "MLP",
                "epoch": epoch,
                "dataset": "train",
                "loss_type": "cross_entropy_like_loss",
                "loss": float(loss),
            })

    return rows


def train_xgboost_loss_curve(X_train, X_val, y_train, y_val):
    """
    训练 XGBoost，并记录训练集和验证集 mlogloss。
    """
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    rows = []

    evals_result = model.evals_result()

    for dataset_name, metric_dict in evals_result.items():
        for metric_name, values in metric_dict.items():
            for epoch, loss in enumerate(values, start=1):
                rows.append({
                    "model": "XGBoost",
                    "epoch": epoch,
                    "dataset": dataset_name,
                    "loss_type": metric_name,
                    "loss": float(loss),
                })

    return rows


def plot_loss_curves(loss_df):
    """
    绘制完整 Loss 曲线。
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6))

    for (model_name, dataset_name), sub_df in loss_df.groupby(["model", "dataset"]):
        sub_df = sub_df.sort_values("epoch")

        label = f"{model_name}-{dataset_name}"

        # XGBoost 的 dataset 名通常是 validation_0 / validation_1
        label = label.replace("validation_0", "train")
        label = label.replace("validation_1", "val")

        plt.plot(
            sub_df["epoch"],
            sub_df["loss"],
            label=label,
            linewidth=1.8,
        )

    plt.title("Loss Curve Comparison")
    plt.xlabel("Epoch / Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "loss_curve_comparison.png", dpi=300)
    plt.close()


def plot_loss_curves_zoomed(loss_df):
    """
    绘制局部放大版 Loss 曲线。
    为了避免前几轮 loss 过大压缩后期变化，这里去掉前 5 轮再画。
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    zoom_df = loss_df[loss_df["epoch"] > 5].copy()

    if zoom_df.empty:
        return

    plt.figure(figsize=(11, 6))

    for (model_name, dataset_name), sub_df in zoom_df.groupby(["model", "dataset"]):
        sub_df = sub_df.sort_values("epoch")

        label = f"{model_name}-{dataset_name}"
        label = label.replace("validation_0", "train")
        label = label.replace("validation_1", "val")

        plt.plot(
            sub_df["epoch"],
            sub_df["loss"],
            label=label,
            linewidth=1.8,
        )

    plt.title("Loss Curve Comparison (After Epoch 5)")
    plt.xlabel("Epoch / Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "loss_curve_comparison_zoomed.png", dpi=300)
    plt.close()


def run_loss_analysis():
    """
    Loss 曲线分析入口。
    """
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_val, y_train, y_val = load_processed_data()

    all_rows = []

    print("开始 Loss 曲线分析...")
    print("-" * 70)

    print("训练 Logistic Regression 的 SGD 版本并记录 log loss...")
    all_rows.extend(
        train_logistic_loss_curve(
            X_train,
            X_val,
            y_train,
            y_val,
            epochs=60,
            batch_size=256,
        )
    )

    print("训练 Linear SVM 的 SGD 版本并记录 hinge loss...")
    all_rows.extend(
        train_linear_svm_loss_curve(
            X_train,
            X_val,
            y_train,
            y_val,
            epochs=60,
            batch_size=256,
        )
    )

    print("训练 MLP 并记录 loss_curve_...")
    all_rows.extend(
        train_mlp_loss_curve(
            X_train,
            X_val,
            y_train,
            y_val,
        )
    )

    print("训练 XGBoost 并记录 mlogloss...")
    all_rows.extend(
        train_xgboost_loss_curve(
            X_train,
            X_val,
            y_train,
            y_val,
        )
    )

    loss_df = pd.DataFrame(all_rows)

    loss_df.to_csv(
        METRIC_DIR / "loss_analysis_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 同时覆盖原来的 loss_history.csv，方便原有 evaluate/app 逻辑读取
    loss_df.to_csv(
        METRIC_DIR / "loss_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_loss_curves(loss_df)
    plot_loss_curves_zoomed(loss_df)

    print("-" * 70)
    print("Loss 曲线分析完成。")
    print(f"Loss 数据保存到：{METRIC_DIR / 'loss_analysis_summary.csv'}")
    print(f"Loss 曲线保存到：{FIGURE_DIR / 'loss_curve_comparison.png'}")
    print(f"Loss 局部放大图保存到：{FIGURE_DIR / 'loss_curve_comparison_zoomed.png'}")