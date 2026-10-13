"""Synthetic customer base for a German mobile operator.

Churn is generated from a *known* logistic model (TRUE_EFFECTS), so we can check whether the ML
model recovers the real drivers instead of just trusting its feature-importance chart.
All data is synthetic; no real customer or operator data is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CITIES = ["Berlin", "Hamburg", "Munich", "Cologne", "Frankfurt"]
CONTRACTS = ["month-to-month", "12-month", "24-month"]
PLANS = {"Basic 10GB": (15.0, 6.0), "Plus 30GB": (30.0, 18.0), "Unlimited 5G": (50.0, 45.0)}  # price €, usage GB
PAYMENTS = ["direct debit", "credit card", "invoice"]

# True effects on the churn log-odds. Used to generate labels and to validate model explanations.
TRUE_EFFECTS = {
    "contract=month-to-month": +1.4,
    "contract=12-month": +0.5,
    "tenure_months (per month)": -0.035,
    "dropped_call_rate_pct (per pp above 0.8)": +0.7,
    "support_calls_90d (per call)": +0.35,
    "monthly_charge_eur (per € above 30)": +0.02,
    "family_bundle": -0.8,
    "payment=invoice": +0.3,
}
INTERCEPT = -1.55
FEATURES_NUMERIC = ["tenure_months", "monthly_charge_eur", "data_usage_gb", "dropped_call_rate_pct",
                    "support_calls_90d", "age"]
FEATURES_CATEGORICAL = ["contract", "plan", "payment_method", "city", "family_bundle"]


def generate_customers(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(CONTRACTS, n, p=[0.45, 0.30, 0.25])
    plan = rng.choice(list(PLANS), n, p=[0.35, 0.40, 0.25])
    price = np.array([PLANS[p][0] for p in plan])
    usage_mean = np.array([PLANS[p][1] for p in plan])
    city = rng.choice(CITIES, n)

    # Network quality differs by city: one city has a known weak spot.
    city_quality = {"Berlin": 0.8, "Hamburg": 0.75, "Munich": 0.7, "Cologne": 1.4, "Frankfurt": 0.8}
    dropped = np.round(rng.lognormal(np.log([city_quality[c] for c in city]), 0.45), 2)

    tenure = np.clip(np.round(rng.exponential(22, n)) + 1, 1, 72).astype(int)
    # Long contracts skew to longer tenure.
    tenure = np.where(contract == "24-month", np.clip(tenure + 10, 1, 72), tenure)
    support = rng.poisson(0.5 + 0.6 * np.clip(dropped - 0.8, 0, None))
    family = rng.random(n) < 0.25
    payment = rng.choice(PAYMENTS, n, p=[0.55, 0.25, 0.20])
    monthly = np.round(price * rng.uniform(0.9, 1.15, n) - family * 5, 2)
    usage = np.round(np.clip(rng.normal(usage_mean, usage_mean * 0.35), 0.2, None), 1)
    age = rng.integers(18, 80, n)

    logit = (INTERCEPT
             + TRUE_EFFECTS["contract=month-to-month"] * (contract == "month-to-month")
             + TRUE_EFFECTS["contract=12-month"] * (contract == "12-month")
             + TRUE_EFFECTS["tenure_months (per month)"] * tenure
             + TRUE_EFFECTS["dropped_call_rate_pct (per pp above 0.8)"] * (dropped - 0.8)
             + TRUE_EFFECTS["support_calls_90d (per call)"] * support
             + TRUE_EFFECTS["monthly_charge_eur (per € above 30)"] * (monthly - 30)
             + TRUE_EFFECTS["family_bundle"] * family
             + TRUE_EFFECTS["payment=invoice"] * (payment == "invoice")
             + rng.normal(0, 0.5, n))
    churned = rng.random(n) < 1 / (1 + np.exp(-logit))

    return pd.DataFrame({
        "customer_id": [f"C-{i:05d}" for i in range(1, n + 1)],
        "city": city,
        "contract": contract,
        "plan": plan,
        "tenure_months": tenure,
        "monthly_charge_eur": monthly,
        "data_usage_gb": usage,
        "dropped_call_rate_pct": dropped,
        "support_calls_90d": support,
        "payment_method": payment,
        "family_bundle": family,
        "age": age,
        "churned": churned,
    })
