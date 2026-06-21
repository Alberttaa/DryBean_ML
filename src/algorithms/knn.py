from sklearn.neighbors import KNeighborsClassifier


def build_model():
    """
    KNN 多分类模型。
    非训练型算法，主要依赖距离度量和 K 值。
    """
    model = KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        metric="minkowski",
        p=2,
        n_jobs=-1,
    )
    return model