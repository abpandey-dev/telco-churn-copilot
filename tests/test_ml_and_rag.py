import json
from pathlib import Path

import pytest

from churn_copilot import model as churn_model
from churn_copilot.data import generate_customers
from churn_copilot.rag import Retriever, load_chunks

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def df():
    return generate_customers(5000, seed=42)


@pytest.fixture(scope="module")
def models(df):
    return churn_model.train(df)


def test_data_is_deterministic_and_realistic(df):
    again = generate_customers(5000, seed=42)
    assert df.equals(again)
    assert 0.2 < df["churned"].mean() < 0.3
    rates = df.groupby("contract")["churned"].mean()
    assert rates["month-to-month"] > rates["12-month"] > rates["24-month"]


def test_models_beat_chance_and_logistic_is_competitive(models):
    lr, gb = models.metrics["logistic_regression"], models.metrics["gradient_boosting"]
    assert lr["roc_auc"] > 0.72 and gb["roc_auc"] > 0.68
    assert lr["roc_auc"] >= gb["roc_auc"] - 0.01  # the explainable model is not worse


def test_importance_recovers_true_drivers(models):
    imp = churn_model.global_importance(models)
    top6 = set(imp.index[:6])
    assert top6 == {"tenure_months", "contract", "dropped_call_rate_pct", "monthly_charge_eur",
                    "family_bundle", "support_calls_90d"}
    assert abs(imp["city"]) < 0.005  # city has no direct effect once network quality is in


def test_explanation_for_high_risk_customer(df, models):
    risky = df.loc[churn_model.score_all(models, df).idxmax()]
    e = churn_model.explain(models, risky)
    assert e["risk_band"] == "high"
    assert e["top_risk_drivers"] and all(d["log_odds"] > 0.05 for d in e["top_risk_drivers"])
    assert all(f["log_odds"] < -0.05 for f in e["protective_factors"])


def test_chunks_have_unique_ids_and_text():
    chunks = load_chunks()
    assert len(chunks) >= 20
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.text for c in chunks)


def test_retrieval_quality_floor():
    """Regression guard on the retrieval eval set: keyword search must keep finding the right policy."""
    r, qs = Retriever(), json.loads((ROOT / "eval" / "rag_questions.json").read_text())
    hit3 = sum(q["expected"] in [h["id"] for h in r.search(q["question"], 3)] for q in qs) / len(qs)
    hit1 = sum([h["id"] for h in r.search(q["question"], 1)] == [q["expected"]] for q in qs) / len(qs)
    assert hit3 >= 0.85 and hit1 >= 0.70
