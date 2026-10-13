"""Churn models: an explainable logistic regression and a gradient-boosting challenger.

Logistic regression is the deployed model: on this data it matches gradient boosting on ROC-AUC,
and its per-customer explanations are exact (coefficient x standardised value), which matters when
a retention agent has to justify an offer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_copilot.data import FEATURES_CATEGORICAL, FEATURES_NUMERIC

FEATURES = FEATURES_NUMERIC + FEATURES_CATEGORICAL

REASON_TEXT = {
    "contract": "on a {value} contract",
    "tenure_months": "customer for {value} months",
    "dropped_call_rate_pct": "dropped-call rate of {value}%",
    "support_calls_90d": "{value} support calls in the last 90 days",
    "monthly_charge_eur": "pays €{value}/month",
    "payment_method": "pays by {value}",
    "family_bundle": "family bundle: {value}",
    "plan": "on the {value} plan",
    "city": "lives in {value}",
    "data_usage_gb": "uses {value} GB/month",
    "age": "age {value}",
}


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), FEATURES_NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), FEATURES_CATEGORICAL),
    ])


@dataclass
class ChurnModels:
    logistic: Pipeline
    boosting: Pipeline
    metrics: dict
    X_test: pd.DataFrame
    y_test: pd.Series


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    X["family_bundle"] = X["family_bundle"].astype(str)
    return X


def train(df: pd.DataFrame, seed: int = 0) -> ChurnModels:
    X, y = _prep(df), df["churned"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)

    logistic = Pipeline([("prep", _preprocessor()), ("clf", LogisticRegression(max_iter=2000))]).fit(X_tr, y_tr)
    boosting = Pipeline([("prep", _preprocessor()),
                         ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                random_state=seed))]).fit(X_tr, y_tr)
    metrics = {}
    for name, m in [("logistic_regression", logistic), ("gradient_boosting", boosting)]:
        p = m.predict_proba(X_te)[:, 1]
        metrics[name] = {
            "roc_auc": round(roc_auc_score(y_te, p), 3),
            "pr_auc": round(average_precision_score(y_te, p), 3),
            "brier": round(brier_score_loss(y_te, p), 4),
        }
    metrics["test_churn_rate"] = round(float(y_te.mean()), 3)
    return ChurnModels(logistic, boosting, metrics, X_te, y_te)


def global_importance(models: ChurnModels, seed: int = 0) -> pd.Series:
    """Permutation importance (drop in ROC-AUC) of each original feature for the deployed model."""
    r = permutation_importance(models.logistic, models.X_test, models.y_test, scoring="roc_auc",
                               n_repeats=10, random_state=seed)
    return pd.Series(r.importances_mean, index=FEATURES).sort_values(ascending=False)


def explain(models: ChurnModels, customer: pd.Series, top: int = 3) -> dict:
    """Churn probability plus the features pushing this customer's risk up the most.

    Contribution of a feature = sum over its encoded columns of coefficient x encoded value, relative
    to an average customer (numeric features are standardised, so 0 = average).
    """
    X = _prep(customer.to_frame().T)
    prep, clf = models.logistic.named_steps["prep"], models.logistic.named_steps["clf"]
    encoded = prep.transform(X)
    encoded = encoded.toarray()[0] if hasattr(encoded, "toarray") else np.asarray(encoded)[0]
    contrib: dict[str, float] = {}
    for col_name, value, coef in zip(prep.get_feature_names_out(), encoded, clf.coef_[0]):
        feature = col_name.split("__", 1)[1]
        feature = next(f for f in FEATURES if feature == f or feature.startswith(f + "_"))
        contrib[feature] = contrib.get(feature, 0.0) + coef * value

    MIN_EFFECT = 0.05  # ignore near-zero contributions
    drivers = sorted(((f, c) for f, c in contrib.items() if c > MIN_EFFECT), key=lambda t: -t[1])[:top]
    protective = sorted(((f, c) for f, c in contrib.items() if c < -MIN_EFFECT), key=lambda t: t[1])[:2]

    def describe(f: str) -> str:
        return REASON_TEXT[f].format(value=customer[f])

    proba = float(models.logistic.predict_proba(X)[0, 1])
    return {
        "customer_id": customer["customer_id"],
        "churn_probability": round(proba, 3),
        "risk_band": "high" if proba >= 0.5 else "medium" if proba >= 0.25 else "low",
        "top_risk_drivers": [{"feature": f, "reason": describe(f), "log_odds": round(c, 2)} for f, c in drivers],
        "protective_factors": [{"feature": f, "reason": describe(f), "log_odds": round(c, 2)} for f, c in protective],
    }


def score_all(models: ChurnModels, df: pd.DataFrame) -> pd.Series:
    return pd.Series(models.logistic.predict_proba(_prep(df))[:, 1], index=df.index, name="churn_probability")
