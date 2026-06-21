from pathlib import Path
import json

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
METRIC_DIR = ROOT_DIR / "results" / "metrics"
FIGURE_DIR = ROOT_DIR / "results" / "figures"
CM_DIR = FIGURE_DIR / "confusion_matrices"


st.set_page_config(
    page_title="Dry Bean 多分类机器学习项目",
    page_icon="🌱",
    layout="wide",
)


def load_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def load_json(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def show_dataframe(title: str, path: Path, height=350):
    st.subheader(title)
    df = load_csv(path)
    if df is None:
        st.warning(f"暂未找到文件：{path.relative_to(ROOT_DIR)}")
        return None
    st.dataframe(df, use_container_width=True, height=height)
    return df


def show_image(title: str, path: Path):
    st.subheader(title)
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.warning(f"暂未找到图片：{path.relative_to(ROOT_DIR)}")


def show_table_auto_height(title: str, path: Path, max_height=350):
    """
    根据表格行数自动调整高度，避免短表格下面出现大片空白。
    """
    st.subheader(title)
    df = load_csv(path)

    if df is None:
        st.warning(f"暂未找到文件：{path.relative_to(ROOT_DIR)}")
        return None

    row_count = len(df)
    height = min(max_height, max(120, row_count * 36 + 45))

    st.dataframe(df, use_container_width=True, height=height)
    return df


def get_accuracy_conclusion():
    df = load_csv(METRIC_DIR / "accuracy_summary.csv")
    if df is None or df.empty:
        return "暂未生成精度结果。"

    best = df.sort_values("test_accuracy", ascending=False).iloc[0]
    worst = df.sort_values("test_accuracy", ascending=True).iloc[0]
    gap = best["test_accuracy"] - worst["test_accuracy"]

    return (
        f"测试集上表现最好的模型是 **{best['model']}**，"
        f"准确率为 **{best['test_accuracy']:.4f}**。"
        f"最低模型为 **{worst['model']}**，准确率为 **{worst['test_accuracy']:.4f}**。"
        f"整体差距约为 **{gap:.4f}**，说明多个模型在该数据集上整体表现都较好，"
        f"但放大图可以更清楚地观察高精度区间的细微差异。"
    )


def get_speed_conclusion():
    df = load_csv(METRIC_DIR / "speed_summary.csv")
    if df is None or df.empty:
        return "暂未生成推理速度结果。"

    fastest = df.sort_values("samples_per_second", ascending=False).iloc[0]
    slowest = df.sort_values("samples_per_second", ascending=True).iloc[0]

    return (
        f"推理速度最快的模型是 **{fastest['model']}**，"
        f"约为 **{fastest['samples_per_second']:.2f} samples/s**。"
        f"速度最慢的模型是 **{slowest['model']}**。"
        f"推理速度差异主要与模型结构复杂度、是否需要距离计算、是否为集成模型有关。"
    )


def get_overfit_conclusion():
    df = load_csv(METRIC_DIR / "overfit_summary.csv")
    if df is None or df.empty:
        return "暂未生成过拟合分析结果。"

    largest = df.sort_values("train_test_gap", ascending=False).iloc[0]
    smallest = df.sort_values("train_test_gap", ascending=True).iloc[0]

    return (
        f"训练集与测试集精度差距最大的模型是 **{largest['model']}**，"
        f"差值为 **{largest['train_test_gap']:.4f}**，说明它相对更可能存在过拟合。"
        f"差距最小的模型是 **{smallest['model']}**，"
        f"差值为 **{smallest['train_test_gap']:.4f}**，泛化表现相对更稳定。"
    )


def get_robustness_conclusion(noise_type=None):
    df = load_csv(METRIC_DIR / "robustness_summary.csv")
    if df is None or df.empty:
        return "暂未生成鲁棒性实验结果。"

    if noise_type is not None:
        df = df[df["noise_type"] == noise_type]

    avg_df = (
        df.groupby("model", as_index=False)["accuracy_drop"]
        .mean()
        .sort_values("accuracy_drop", ascending=True)
    )

    best = avg_df.iloc[0]
    worst = avg_df.iloc[-1]

    if noise_type is None:
        prefix = "所有噪声综合来看，"
    else:
        prefix = f"在 **{noise_type}** 噪声下，"

    return (
        f"{prefix}平均精度下降最小的是 **{best['model']}**，"
        f"平均下降约 **{best['accuracy_drop']:.4f}**；"
        f"下降最大的是 **{worst['model']}**，"
        f"平均下降约 **{worst['accuracy_drop']:.4f}**。"
        f"精度下降越小说明模型对该类噪声越稳定。"
    )


def get_loss_conclusion():
    df = load_csv(METRIC_DIR / "loss_analysis_summary.csv")
    if df is None or df.empty:
        return "暂未生成 Loss 曲线数据。"

    models = ", ".join(df["model"].dropna().unique())

    return (
        f"当前 Loss 曲线包含：**{models}**。"
        f"KNN 属于非训练型算法，随机森林不以连续梯度下降方式训练，因此不绘制 Loss 曲线。"
        f"逻辑回归和线性 SVM 使用 SGD 版本记录训练过程，MLP 和 XGBoost 使用模型自带训练损失记录。"
    )


def image_with_conclusion(title: str, image_path: Path, conclusion: str):
    """
    左边展示图片，右边展示简要结论。
    """
    st.subheader(title)
    col_img, col_text = st.columns([2, 1])

    with col_img:
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.warning(f"暂未找到图片：{image_path.relative_to(ROOT_DIR)}")

    with col_text:
        st.markdown("**简要结论：**")
        st.info(conclusion)


def format_accuracy_table(df: pd.DataFrame):
    if df is None or df.empty:
        return df

    show_df = df.copy()
    percent_cols = [
        "train_accuracy",
        "val_accuracy",
        "test_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "train_test_gap",
    ]

    for col in percent_cols:
        if col in show_df.columns:
            show_df[col] = show_df[col].apply(lambda x: f"{x:.4f}")

    return show_df


def project_header():
    st.title("🌱 Dry Bean Dataset 多分类机器学习工程项目")
    st.caption("数据分析 · 数据处理 · 多算法实验分析 · 工程化系统展示 · 课程总结")

    st.markdown(
        """
        本项目基于 Dry Bean Dataset 的脏数据版本，完成从数据分析、数据清洗、特征工程、
        多算法训练测试，到结果可视化展示的完整机器学习工程流程。项目实现了逻辑回归、
        KNN、SVM、MLP、随机森林和 XGBoost 六种多分类算法，其中随机森林和 XGBoost
        属于课堂外拓展算法。
        """
    )


project_header()

tab_overview, tab_data, tab_process, tab_algo, tab_loss_speed, tab_robust, tab_summary = st.tabs(
    [
        "项目概览",
        "数据分析",
        "数据处理",
        "算法实验结果",
        "Loss/速度/过拟合",
        "鲁棒性分析",
        "课程总结",
    ]
)


with tab_overview:
    st.header("1. 项目概览")

    summary = load_json(METRIC_DIR / "data_analysis_summary.json")

    col1, col2, col3, col4 = st.columns(4)
    if summary:
        col1.metric("总样本数", summary.get("total_rows", "未知"))
        col2.metric("特征数量", summary.get("feature_count", "未知"))
        col3.metric("标签列", summary.get("target_column", "Class"))
        col4.metric("类别数量", len(summary.get("standard_classes", [])))
    else:
        col1.metric("总样本数", "13611")
        col2.metric("特征数量", "16")
        col3.metric("标签列", "Class")
        col4.metric("类别数量", "7")

    st.subheader("项目任务")
    st.markdown(
        """
        - **数据分析**：观察数据集用途、字段含义、类别分布和数据污染情况。
        - **数据处理**：完成标签修正、数值清洗、缺失值填充、异常值处理、重复值处理和标准化。
        - **多算法实验分析**：比较六种多分类算法在精度、Loss、推理速度、鲁棒性和过拟合方面的表现。
        - **工程化展示**：通过统一命令行运行算法模块，使用 Streamlit 单独展示实验结果。
        """
    )

    st.subheader("实现算法")
    algo_info = pd.DataFrame(
        [
            ["Logistic Regression", "逻辑回归", "课堂算法", "线性多分类基准模型"],
            ["KNN", "K近邻", "课堂算法", "基于距离度量和邻居投票的非训练型算法"],
            ["SVM", "支持向量机", "课堂算法", "基于最大间隔和核函数的分类算法"],
            ["MLP", "多层感知机", "课堂算法", "基于神经网络的非线性分类模型"],
            ["Random Forest", "随机森林", "课外拓展算法", "集成多棵决策树进行投票分类"],
            ["XGBoost", "极端梯度提升", "课外拓展算法", "基于梯度提升树的强表格数据算法"],
        ],
        columns=["英文名称", "中文名称", "来源", "说明"],
    )
    st.dataframe(algo_info, use_container_width=True, hide_index=True)


with tab_data:
    st.header("2. 数据分析")

    st.markdown(
        """
        数据分析部分主要用于说明输入数据的基本情况和污染情况，包括样本数量、字段类型、
        缺失值、重复行、标签污染和物理异常值等。
        """
    )

    show_dataframe(
        "数据质量统计表",
        METRIC_DIR / "data_quality_report.csv",
        height=250,
    )

    show_dataframe(
        "字段概览表",
        METRIC_DIR / "feature_overview.csv",
        height=320,
    )

    col1, col2 = st.columns(2)
    with col1:
        show_image(
            "类别分布图",
            FIGURE_DIR / "class_distribution.png",
        )

    with col2:
        show_image(
            "缺失值统计图",
            FIGURE_DIR / "missing_values.png",
        )

    show_dataframe(
        "类别分布统计表",
        METRIC_DIR / "class_distribution.csv",
        height=280,
    )

    show_dataframe(
        "缺失值统计表",
        METRIC_DIR / "missing_values.csv",
        height=280,
    )


with tab_process:
    st.header("3. 数据处理与特征工程")

    st.markdown(
        """
        数据处理部分针对输入脏数据进行清洗和特征工程。为了避免数据泄露，缺失值填充的中位数
        和标准化参数均只在训练集上计算，再应用到验证集和测试集。
        """
    )

    st.subheader("本项目使用的数据处理方法")
    process_df = pd.DataFrame(
        [
            ["标签清洗", "统一大小写、去除空格、修正 D3RMAS0N、S3K3R、H0R0Z、B0MBAY 等污染标签"],
            ["数值清洗", "将 '?' 识别为缺失值，将 '0.8194 cm' 提取为数值 0.8194"],
            ["异常值处理", "将 Area 等数值特征中小于等于 0 的值视为异常值并转为缺失值"],
            ["重复值处理", "删除训练集中的重复样本，验证集和测试集保持原始评估结构"],
            ["缺失值填充", "使用训练集各特征中位数填充 train/val/test 中的缺失值"],
            ["标签编码", "将 7 个豆类标签编码为 0 到 6"],
            ["特征标准化", "使用 StandardScaler 对数值特征标准化"],
        ],
        columns=["处理步骤", "具体说明"],
    )
    st.dataframe(process_df, use_container_width=True, hide_index=True)

    show_dataframe(
        "预处理报告",
        METRIC_DIR / "preprocess_report.csv",
        height=260,
    )

    show_dataframe(
        "预处理后缺失值检查",
        METRIC_DIR / "missing_after_preprocess.csv",
        height=320,
    )

    label_mapping = load_json(PROCESSED_DIR / "label_mapping.json")
    if label_mapping:
        st.subheader("标签编码映射")
        label_df = pd.DataFrame(
            [{"Class": k, "Class_ID": v} for k, v in label_mapping.items()]
        )
        st.dataframe(label_df, use_container_width=True, hide_index=True)

    feature_names = load_json(PROCESSED_DIR / "feature_names.json")
    if feature_names:
        st.subheader("最终进入模型的特征")
        st.write(", ".join(feature_names))


with tab_algo:
    st.header("4. 多算法实验结果")

    st.markdown(
        """
        本部分展示六种多分类算法在测试集上的精度、宏平均 Precision、宏平均 Recall、
        宏平均 F1-score 等指标。所有算法均使用同一份预处理后的训练、验证和测试数据。
        """
    )

    accuracy_df = load_csv(METRIC_DIR / "accuracy_summary.csv")
    st.subheader("算法精度对比表")
    if accuracy_df is not None:
        st.dataframe(
            format_accuracy_table(accuracy_df),
            use_container_width=True,
            hide_index=True,
        )

        best_row = accuracy_df.sort_values("test_accuracy", ascending=False).iloc[0]
        st.success(
            f"当前测试集精度最高的算法是：{best_row['model']}，"
            f"测试集精度为 {best_row['test_accuracy']:.4f}。"
        )
    else:
        st.warning("暂未找到 accuracy_summary.csv，请先运行 python main.py --mode evaluate")

    
    image_with_conclusion(
        "测试集精度对比图",
        FIGURE_DIR / "accuracy_comparison.png",
        get_accuracy_conclusion(),
    )

    show_image(
        "测试集精度对比图（局部放大）",
        FIGURE_DIR / "accuracy_comparison_zoomed.png",
    )
    

    st.subheader("混淆矩阵")
    cm_images = sorted(CM_DIR.glob("*_confusion_matrix.png"))
    if cm_images:
        selected_cm = st.selectbox(
            "选择要查看的算法混淆矩阵",
            cm_images,
            format_func=lambda p: p.name.replace("_confusion_matrix.png", ""),
        )
        st.image(str(selected_cm), use_container_width=True)
    else:
        st.warning("暂未找到混淆矩阵图片，请先运行 python main.py --mode evaluate")


with tab_loss_speed:
    st.header("5. Loss 曲线、推理速度与过拟合分析")

    st.subheader("Loss 曲线对比")
    st.markdown(
        """
        KNN 属于非训练型算法，不绘制 Loss 曲线。随机森林通常不以连续梯度下降方式训练，
        因此不作为 Loss 曲线分析对象。为了更完整地展示模型优化过程，本项目额外使用
        SGD 版本的 Logistic Regression 和 Linear SVM 记录训练损失，并展示 MLP 与 XGBoost
        的训练损失变化。
        """
    )

    show_dataframe(
        "Loss 曲线数据表",
        METRIC_DIR / "loss_analysis_summary.csv",
        height=260,
    )

    image_with_conclusion(
        "Loss 曲线对比图",
        FIGURE_DIR / "loss_curve_comparison.png",
        get_loss_conclusion(),
    )

    show_image(
        "Loss 曲线对比图（去除前5轮后的局部观察）",
        FIGURE_DIR / "loss_curve_comparison_zoomed.png",
    )
    st.divider()

    show_dataframe(
        "推理速度对比表",
        METRIC_DIR / "speed_summary.csv",
        height=260,
    )
    image_with_conclusion(
        "推理速度对比图",
        FIGURE_DIR / "speed_comparison.png",
        get_speed_conclusion(),
    )

    st.divider()

    st.subheader("过拟合分析")
    st.markdown(
        """
        过拟合程度通过训练集精度与测试集精度的差值进行观察。
        差值越大，说明模型越可能对训练集记忆过强，而泛化到测试集时表现下降。
        """
    )
    show_dataframe(
        "过拟合分析表",
        METRIC_DIR / "overfit_summary.csv",
        height=260,
    )
    image_with_conclusion(
        "过拟合差距对比图",
        FIGURE_DIR / "overfit_comparison.png",
        get_overfit_conclusion(),
    )


with tab_robust:
    st.header("6. 鲁棒性分析")

    st.markdown(
        """
        鲁棒性实验通过对训练数据加入不同类型、不同强度的噪声，然后重新训练模型，
        最后在干净测试集上评估精度下降情况。噪声类型包括高斯噪声、均匀噪声、
        特征缺失噪声和标签噪声。
        """
    )

    show_dataframe(
        "鲁棒性实验完整结果",
        METRIC_DIR / "robustness_summary.csv",
        height=350,
    )

    show_dataframe(
        "各算法平均精度下降表",
        METRIC_DIR / "robustness_average_drop.csv",
        height=260,
    )

    image_with_conclusion(
        "平均精度下降对比图",
        FIGURE_DIR / "robustness_average_drop.png",
        get_robustness_conclusion(),
    )

    image_with_conclusion(
        "高斯噪声鲁棒性曲线",
        FIGURE_DIR / "robustness_curve_gaussian.png",
        get_robustness_conclusion("gaussian"),
    )

    image_with_conclusion(
        "均匀噪声鲁棒性曲线",
        FIGURE_DIR / "robustness_curve_uniform.png",
        get_robustness_conclusion("uniform"),
    )

    image_with_conclusion(
        "特征缺失噪声鲁棒性曲线",
        FIGURE_DIR / "robustness_curve_feature_dropout.png",
        get_robustness_conclusion("feature_dropout"),
    )

    image_with_conclusion(
        "标签噪声鲁棒性曲线",
        FIGURE_DIR / "robustness_curve_label_noise.png",
        get_robustness_conclusion("label_noise"),
    )
        

with tab_summary:
    st.header("7. 课程总结")

    st.subheader("课程总结")
    st.markdown(
        """
        通过本项目，可以将课程中学习到的数据清洗、特征工程、分类算法、模型评估和工程化组织方法
        串联起来。项目不仅比较了逻辑回归、KNN、SVM 和 MLP 等课堂算法，也进一步引入了随机森林
        和 XGBoost 等课堂外算法，增强了对表格型多分类任务的理解。
        """
    )


    st.info("展示界面只读取已保存结果，不重新训练模型；算法训练阶段仍然通过 main.py 在命令行中运行。")