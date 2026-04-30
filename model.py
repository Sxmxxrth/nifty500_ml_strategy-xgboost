# ============================================================
# model.py — Train XGBoost classifier, generate stock scores
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

from config import TRAIN_END_DATE
from features import FEATURE_COLS


def get_train_test_split(dataset: pd.DataFrame):
    """
    Split dataset into train (up to TRAIN_END_DATE) and test.
    ⚠️  Time-based split — never shuffle financial time series!
    """
    dataset = dataset.reset_index()
    dataset["date"] = pd.to_datetime(dataset["date"])

    train = dataset[dataset["date"] <= TRAIN_END_DATE].copy()
    test  = dataset[dataset["date"] >  TRAIN_END_DATE].copy()

    print(f"\n📅 Train set: {train['date'].min().date()} → {train['date'].max().date()} ({len(train):,} rows)")
    print(f"   Test  set: {test['date'].min().date()} → {test['date'].max().date()} ({len(test):,} rows)")

    return train, test


def train_model(train: pd.DataFrame, n_iter: int = 30) -> Pipeline:
    """
    Train XGBoost classifier with hyperparameter search using
    TimeSeriesSplit cross-validation.

    Returns a fitted sklearn Pipeline (scaler + XGBoost).
    """
    print("\n🤖 Training XGBoost model...")

    X_train = train[FEATURE_COLS].values
    y_train = train["target"].values

    # Pipeline: RobustScaler (handles outliers) + XGBoost
    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("model",  xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        ))
    ])

    # Hyperparameter search space
    param_dist = {
        "model__n_estimators":    [100, 200, 300, 400],
        "model__max_depth":       [3, 4, 5, 6],
        "model__learning_rate":   [0.01, 0.05, 0.1, 0.15],
        "model__subsample":       [0.6, 0.7, 0.8, 0.9],
        "model__colsample_bytree":[0.6, 0.7, 0.8, 1.0],
        "model__min_child_weight":[1, 3, 5],
        "model__gamma":           [0, 0.1, 0.2],
        "model__reg_alpha":       [0, 0.1, 0.5],
        "model__reg_lambda":      [1, 1.5, 2],
    }

    # TimeSeriesSplit: 5 folds — respects temporal order
    tscv = TimeSeriesSplit(n_splits=5)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
        random_state=42,
    )

    search.fit(X_train, y_train)

    best_score = search.best_score_
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}

    print(f"   Best CV AUC: {best_score:.4f}")
    print(f"   Best params: {best_params}")

    return search.best_estimator_


def score_stocks_on_date(pipeline: Pipeline,
                         features_df: pd.DataFrame,
                         date: pd.Timestamp) -> pd.Series:
    """
    Given a fitted pipeline and feature matrix, score all stocks
    available on a specific rebalance date.

    Returns a Series: {ticker: probability_of_outperforming}
    """
    # Get features for this date only
    try:
        day_features = features_df.xs(date, level="date")
    except KeyError:
        # If exact date not found, find nearest previous date
        available = features_df.index.get_level_values("date").unique()
        available = available[available <= date]
        if len(available) == 0:
            return pd.Series(dtype=float)
        nearest = available[-1]
        day_features = features_df.xs(nearest, level="date")

    if day_features.empty:
        return pd.Series(dtype=float)

    # Drop tickers with missing features
    day_features = day_features[FEATURE_COLS].dropna()

    if day_features.empty:
        return pd.Series(dtype=float)

    X = day_features.values
    scores = pipeline.predict_proba(X)[:, 1]  # Prob of outperforming

    return pd.Series(scores, index=day_features.index, name="score")


def evaluate_model(pipeline: Pipeline, test: pd.DataFrame) -> dict:
    """Quick evaluation metrics on the test set."""
    from sklearn.metrics import roc_auc_score, accuracy_score

    X_test = test[FEATURE_COLS].dropna()
    y_test = test.loc[X_test.index, "target"]

    probs = pipeline.predict_proba(X_test.values)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "test_auc":      roc_auc_score(y_test, probs),
        "test_accuracy": accuracy_score(y_test, preds),
    }

    print(f"\n📈 Out-of-sample model performance:")
    print(f"   AUC Accuracy : {metrics['test_auc']:.4f}")
    print(f"   Classification accuracy: {metrics['test_accuracy']:.4f}")

    return metrics
