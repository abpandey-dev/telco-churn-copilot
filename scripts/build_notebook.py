"""Build and execute notebooks/churn_analysis.ipynb, and save its charts to docs/images/ for the README.

    python scripts/build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

cells = [
    md("# Telco churn analysis\n\n"
       "Which mobile customers are about to leave, and why? This notebook explores a **synthetic** customer "
       "base of a German mobile operator, trains two churn models, and checks whether the model recovers the "
       "*true* churn drivers that were used to generate the data.\n\n"
       "> All data is synthetic. The churn drivers are known on purpose so the model's explanations can be "
       "verified, not just trusted."),
    code("import sys, pathlib\n"
         "sys.path.insert(0, str(pathlib.Path.cwd().parent / 'src'))\n"
         "import pandas as pd\n"
         "from churn_copilot.data import generate_customers, TRUE_EFFECTS\n"
         "from churn_copilot import model as churn_model, viz\n"
         "IMG = pathlib.Path.cwd().parent / 'docs' / 'images'\n"
         "IMG.mkdir(parents=True, exist_ok=True)\n"
         "import matplotlib.pyplot as plt\n"
         "from IPython.display import Image, display\n"
         "def show(fig, name):\n"
         "    path = IMG / f'{name}.png'\n"
         "    fig.savefig(path, dpi=130, bbox_inches='tight')\n"
         "    plt.close(fig)\n"
         "    display(Image(filename=str(path)))\n"
         "df = generate_customers(5000, seed=42)\n"
         "print(f'{len(df):,} customers, churn rate {df.churned.mean():.1%}')\n"
         "df.head()"),
    md("## 1. Where does churn come from?\n\nContract type is the biggest lever: month-to-month customers "
       "can leave with one month's notice."),
    code("show(viz.churn_by(df, 'contract', 'Month-to-month customers churn 3x more than 24-month', "
         "order=['month-to-month', '12-month', '24-month']), 'churn_by_contract')"),
    code("show(viz.churn_by_tenure(df), 'churn_by_tenure')"),
    md("Network quality matters: customers with a dropped-call rate above 2% churn far more."),
    code("show(viz.dropped_calls_vs_churn(df), 'churn_by_dropped_calls')"),
    md("Cologne looks like a problem city, but is it the *city* or the *network* there?"),
    code("show(viz.churn_by(df, 'city', 'Cologne churns more than other cities'), 'churn_by_city')"),
    code("df.groupby('city')[['dropped_call_rate_pct']].mean().round(2).T"),
    md("Cologne customers have roughly double the dropped-call rate. The model below shows that, once network "
       "quality is included, city itself adds almost nothing: the real cause is the network."),
    md("## 2. Models\n\nLogistic regression (explainable) vs gradient boosting (challenger), on a 25% "
       "held-out test set."),
    code("m = churn_model.train(df)\npd.DataFrame({k: v for k, v in m.metrics.items() if isinstance(v, dict)}).T"),
    code("probs = {'logistic regression': m.logistic.predict_proba(m.X_test)[:, 1],\n"
         "         'gradient boosting': m.boosting.predict_proba(m.X_test)[:, 1]}\n"
         "show(viz.roc_curves(m.y_test, probs), 'roc_curves')"),
    code("show(viz.calibration(m.y_test, probs), 'calibration')"),
    md("The simpler model wins. The true churn process is close to logistic, so boosting has nothing extra to "
       "find and slightly overfits. Logistic regression is deployed: it is as accurate, better calibrated, and "
       "its per-customer explanations are exact."),
    md("## 3. Did the model learn the real drivers?"),
    code("imp = churn_model.global_importance(m)\nshow(viz.importance_vs_truth(imp), 'importance_vs_truth')"),
    code("pd.Series(TRUE_EFFECTS, name='true log-odds effect').to_frame()"),
    md("The top six learned features are exactly the six strongest true drivers. City, age and plan, which "
       "have no direct effect in the data, come out near zero. The model is picking up real signal, not "
       "artefacts."),
    md("## 4. Who should the retention team call?"),
    code("df['churn_probability'] = churn_model.score_all(m, df)\n"
         "show(viz.risk_distribution(df), 'risk_distribution')"),
    code("bands = pd.cut(df.churn_probability, [0, .25, .5, 1], labels=['low', 'medium', 'high'], include_lowest=True)\n"
         "df.groupby(bands, observed=True).agg(customers=('customer_id', 'size'), actual_churn=('churned', 'mean')).round(3)"),
    code("import json\nrisky = df.loc[df.churn_probability.idxmax()]\n"
         "print(json.dumps(churn_model.explain(m, risky), indent=1, ensure_ascii=False))"),
    md("These per-customer explanations feed the LLM copilot (`churn ask ...`), which combines them with the "
       "company's retention, contract, network-compensation and GDPR policies through RAG."),
]

nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = ROOT / "notebooks" / "churn_analysis.ipynb"
out.parent.mkdir(exist_ok=True)
NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(out.parent)}}).execute()
nbformat.write(nb, out)
print(f"wrote {out}")
