import argparse
from pathlib import Path


def make_dirs():
    """创建项目运行需要的文件夹。"""
    dirs = [
        "data/raw",
        "data/processed",
        "data/noise",
        "results/metrics",
        "results/figures",
        "results/figures/confusion_matrices",
        "saved_models",
        "reports",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Dry Bean Dataset Machine Learning Project")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "analyze",
            "preprocess",
            "train_all",
            "evaluate",
            "robustness",
            "loss_analysis",
        ],
        help=(
            "运行模式："
            "analyze 数据分析；"
            "preprocess 数据清洗与特征工程；"
            "train_all 训练全部算法；"
            "evaluate 测试与结果评价；"
            "robustness 鲁棒性实验；"
            "loss_analysis Loss曲线分析"
        ),
    )

    args = parser.parse_args()
    make_dirs()

    if args.mode == "analyze":
        from src.data_analysis import run_data_analysis
        run_data_analysis()

    elif args.mode == "preprocess":
        from src.preprocess import run_preprocess
        run_preprocess()

    elif args.mode == "train_all":
        from src.training.train_all import run_train_all
        run_train_all()

    elif args.mode == "evaluate":
        from src.evaluation.evaluate import run_evaluate
        run_evaluate()

    elif args.mode == "robustness":
        from src.evaluation.robustness import run_robustness
        run_robustness()

    elif args.mode == "loss_analysis":
        from src.evaluation.loss_analysis import run_loss_analysis
        run_loss_analysis()


if __name__ == "__main__":
    main()