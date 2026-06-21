from sklearn.linear_model import LogisticRegression


def build_model():
    """
    逻辑回归多分类模型。
    适合作为线性基准模型。
    """
    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    )
    return model