from sklearn.ensemble import RandomForestClassifier


def build_model():
    """
    随机森林多分类模型。
    课堂没有系统讲过，作为课外拓展算法。
    适合表格数据，鲁棒性通常较好。
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    return model