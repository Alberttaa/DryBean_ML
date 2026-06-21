from pathlib import Path
import time
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.algorithms.logistic_regression import build_model as build_logistic_regression
from src.algorithms.knn import build_model as build_knn
from src.algorithms.svm import build_model as build_svm
from src.algorithms.mlp import build_model as build_mlp
from src.algorithms.random_forest import build_model as build_random_forest
from src.algorithms.xgboost_model import build_model as build_xgboost


PROCESSED_DIR = Path("data/processed")
SAVED_MODEL_DIR = Path("saved_models")
METRIC_DIR = Path("results/metrics")


ALGORITHMS = {
    "logistic_regression": build_logistic_regression,
    "knn": build_knn,
    "svm": build_svm,
    "mlp": build_mlp,
    "random_forest": build_random_forest,
    "xgboost": build_xgboost,
}


def load_processed_data():
    """读取预处理后的训练集、验证集、测试集。"""
    required_files = [
        PROCESSED_DIR / "X_train.csv",
        PROCESSED_DIR / "X_val.csv",
        PROCESSED_DIR / "X_test.csv",
        PROCESSED_DIR / "y_train.csv",
        PROCESSED_DIR / "y_val.csv",
        PROCESSED_DIR / "y_test.csv",
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"没有找到 {file}，请先运行：python main.py --mode preprocess"
            )

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_val = pd.read_csv(PROCESSED_DIR / "X_val.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")

    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["Class_ID"].to_numpy()
    y_val = pd.read_csv(PROCESSED_DIR / "y_val.csv")["Class_ID"].to_numpy()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")["Class_ID"].to_numpy()

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_loss_history(model_name, model):
    """
    保存部分模型的 loss 记录。
    MLP 有 loss_curve_。
    XGBoost 有 evals_result。
    其他模型暂时不保存。
    """
    loss_rows = []

    if hasattr(model, "loss_curve_"):
        for epoch, loss in enumerate(model.loss_curve_, start=1):
            loss_rows.append({
                "model": model_name,
                "epoch": epoch,
                "dataset": "train",
                "loss": float(loss),
            })

    if model_name == "xgboost" and hasattr(model, "evals_result"):
        try:
            evals_result = model.evals_result()
            for dataset_name, metric_dict in evals_result.items():
                for metric_name, values in metric_dict.items():
                    for epoch, loss in enumerate(values, start=1):
                        loss_rows.append({
                            "model": model_name,
                            "epoch": epoch,
                            "dataset": dataset_name,
                            "metric": metric_name,
                            "loss": float(loss),
                        })
        except Exception:
            pass

    return loss_rows


def run_train_all():
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()

    train_summary_rows = []
    all_loss_rows = []

    print("开始训练全部算法...")
    print("-" * 60)

    for model_name, build_func in ALGORITHMS.items():
        print(f"正在训练：{model_name}")

        model = build_func()

        start_time = time.perf_counter()

        # XGBoost 可以传入验证集，方便记录验证集 mlogloss。
        if model_name == "xgboost":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)

        train_time = time.perf_counter() - start_time

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)

        model_path = SAVED_MODEL_DIR / f"{model_name}.pkl"
        joblib.dump(model, model_path)

        train_summary_rows.append({
            "model": model_name,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "train_val_gap": train_acc - val_acc,
            "train_time_seconds": train_time,
            "model_path": str(model_path),
        })

        loss_rows = save_loss_history(model_name, model)
        all_loss_rows.extend(loss_rows)

        print(
            f"{model_name} 训练完成："
            f"train_acc={train_acc:.4f}, "
            f"val_acc={val_acc:.4f}, "
            f"time={train_time:.2f}s"
        )
        print("-" * 60)

    train_summary_df = pd.DataFrame(train_summary_rows)
    train_summary_df.to_csv(
        METRIC_DIR / "train_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if all_loss_rows:
        loss_df = pd.DataFrame(all_loss_rows)
        loss_df.to_csv(
            METRIC_DIR / "loss_history.csv",
            index=False,
            encoding="utf-8-sig"
        )

    # 保存训练配置摘要
    with open(METRIC_DIR / "algorithm_list.json", "w", encoding="utf-8") as f:
        json.dump(list(ALGORITHMS.keys()), f, ensure_ascii=False, indent=4)

    print("全部算法训练完成。")
    print(f"模型保存到：{SAVED_MODEL_DIR}")
    print(f"训练结果保存到：{METRIC_DIR / 'train_summary.csv'}")