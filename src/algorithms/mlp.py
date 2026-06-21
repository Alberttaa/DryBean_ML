from sklearn.neural_network import MLPClassifier


def build_model():
    """
    MLP 多层感知机模型。
    可以记录 loss_curve_，用于后续绘制 loss 曲线。
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
        random_state=42,
    )
    return model