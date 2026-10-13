"""Charts used by the analysis notebook and the README. Each function returns a matplotlib Figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

BLUE, ORANGE, GREY, RED = "#2563EB", "#EA580C", "#94A3B8", "#DC2626"


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def churn_by(df: pd.DataFrame, column: str, title: str | None = None, order: list | None = None):
    rates = df.groupby(column, observed=True)["churned"].agg(["mean", "size"])
    if order:
        rates = rates.reindex(order)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    overall = df["churned"].mean()
    colors = [RED if r > overall * 1.2 else BLUE for r in rates["mean"]]
    bars = ax.bar(rates.index.astype(str), rates["mean"] * 100, color=colors)
    ax.axhline(overall * 100, color=GREY, linestyle="--", linewidth=1, label=f"overall {overall:.0%}")
    for b, (rate, n) in zip(bars, rates.values):
        ax.annotate(f"{rate:.0%}\n(n={int(n)})", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(rates["mean"]) * 100 * 1.35)
    ax.legend(frameon=False, fontsize=8)
    _style(ax, title or f"Churn rate by {column}", ylabel="churn rate (%)")
    fig.tight_layout()
    return fig


def churn_by_tenure(df: pd.DataFrame):
    bins = [0, 6, 12, 24, 36, 48, 72]
    labels = ["0-6", "7-12", "13-24", "25-36", "37-48", "49-72"]
    t = df.assign(tenure_band=pd.cut(df["tenure_months"], bins, labels=labels))
    return churn_by(t, "tenure_band", "Churn falls steadily with tenure (months)", order=labels)


def dropped_calls_vs_churn(df: pd.DataFrame):
    bins = [0, 0.5, 1, 1.5, 2, 3, 4, 20]
    labels = ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4", ">4"]
    t = df.assign(dropped_band=pd.cut(df["dropped_call_rate_pct"], bins, labels=labels))
    return churn_by(t, "dropped_band", "Network quality drives churn: dropped-call rate (%)", order=labels)


def roc_curves(y_true, probs: dict[str, np.ndarray]):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for (name, p), color in zip(probs.items(), [BLUE, ORANGE]):
        fpr, tpr, _ = roc_curve(y_true, p)
        from sklearn.metrics import roc_auc_score
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC {roc_auc_score(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], color=GREY, linestyle=":")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax, "ROC curve on held-out customers", "false positive rate", "true positive rate")
    fig.tight_layout()
    return fig


def calibration(y_true, probs: dict[str, np.ndarray]):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for (name, p), color in zip(probs.items(), [BLUE, ORANGE]):
        frac, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", color=color, label=name)
    ax.plot([0, 1], [0, 1], color=GREY, linestyle=":", label="perfect")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Calibration: predicted vs actual churn", "predicted probability", "actual churn rate")
    fig.tight_layout()
    return fig


def importance_vs_truth(importance: pd.Series):
    """Learned permutation importance, with features that truly drive churn highlighted."""
    true_drivers = {"tenure_months", "contract", "dropped_call_rate_pct", "monthly_charge_eur", "family_bundle",
                    "support_calls_90d", "payment_method"}
    imp = importance.sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [BLUE if f in true_drivers else GREY for f in imp.index]
    ax.barh(imp.index, imp.values, color=colors)
    ax.set_xlabel("drop in ROC-AUC when the feature is shuffled")
    ax.set_title("What the model learned (blue = real driver in the data)", loc="left",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def risk_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(0, 1, 26)
    ax.hist(df.loc[~df["churned"], "churn_probability"], bins=bins, alpha=0.7, color=BLUE, label="stayed")
    ax.hist(df.loc[df["churned"], "churn_probability"], bins=bins, alpha=0.7, color=ORANGE, label="churned")
    for x, lbl in [(0.25, "medium"), (0.5, "high")]:
        ax.axvline(x, color=GREY, linestyle="--", linewidth=1)
        ax.text(x + 0.01, ax.get_ylim()[1] * 0.9, f"{lbl} risk →", fontsize=8, color="#475569")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Predicted churn risk by actual outcome", "predicted churn probability", "customers")
    fig.tight_layout()
    return fig
