from xgboost import XGBClassifier


def build_model():
    """
    XGBoost 多分类模型。
    课堂没有讲过，作为进阶拓展算法。
    适合表格型数据，也可以记录多分类 logloss。
    """
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=7,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    return model