from sklearn.svm import SVC


def build_model():
    """
    支持向量机多分类模型。
    使用 RBF 核函数处理非线性分类边界。
    """
    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        decision_function_shape="ovr",
        random_state=42,
    )
    return model