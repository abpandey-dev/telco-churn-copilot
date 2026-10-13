"""Churn Copilot agent: Claude answers retention questions by combining ML predictions (tools over the
churn model) with company policy (RAG over the knowledge base).

Guardrail: the answer must cite policy chunk ids, and every cited id is checked against the chunks the
agent actually retrieved in this conversation. Citing something it never retrieved is flagged as a
hallucinated citation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import anthropic
import pandas as pd
from pydantic import BaseModel

from churn_copilot import model as churn_model
from churn_copilot.data import generate_customers
from churn_copilot.rag import Retriever

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
PRICES = {"claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50)}
GROUPABLE = ["contract", "plan", "payment_method", "city", "family_bundle", "support_calls_90d", "risk_band"]

SYSTEM_PROMPT = """You are a retention analyst assistant at a German mobile operator.
Answer the question using the tools: get_customer_risk gives the churn model's probability and the main
risk drivers for one customer; segment_churn gives churn rates by segment; search_policies searches the
company's retention, contract, network-compensation, customer-care and GDPR policies.

Ground every recommendation in a retrieved policy and cite the chunk ids you used. Follow the policies
exactly (eligibility, limits, approvals, consent). The churn model supports people; it does not decide
on its own. If the policies don't cover something, say so instead of guessing."""


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


TOOLS = [
    {"name": "get_customer_risk", "strict": True,
     "description": "Churn probability, risk band, top risk drivers and protective factors for one customer, "
                    "plus their profile (contract, plan, tenure, charges, network quality, support calls).",
     "input_schema": _schema({"customer_id": {"type": "string", "description": "e.g. C-00042"}}, ["customer_id"])},
    {"name": "segment_churn", "strict": True,
     "description": "Actual churn rate and average predicted risk per segment, for one grouping column.",
     "input_schema": _schema({"group_by": {"type": "string", "enum": GROUPABLE}}, ["group_by"])},
    {"name": "search_policies", "strict": True,
     "description": "Search company policies. Returns the best-matching sections with their chunk ids.",
     "input_schema": _schema({"query": {"type": "string"}}, ["query"])},
]

ANSWER_SCHEMA = _schema({
    "answer": {"type": "string", "description": "Plain-English answer for a retention agent."},
    "recommended_actions": {"type": "array", "items": _schema({
        "action": {"type": "string"},
        "policy_chunk_id": {"type": "string", "description": "Chunk id that allows this action"},
    }, ["action", "policy_chunk_id"])},
    "citations": {"type": "array", "items": {"type": "string"}, "description": "All policy chunk ids used"},
}, ["answer", "recommended_actions", "citations"])


class Action(BaseModel):
    action: str
    policy_chunk_id: str


class Answer(BaseModel):
    answer: str
    recommended_actions: list[Action]
    citations: list[str]


@dataclass
class AskResult:
    answer: Answer | None
    retrieved_ids: set[str] = field(default_factory=set)
    unsupported_citations: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float | None = None
    error: str | None = None


class Workspace:
    """Customer data, trained models and the policy retriever, built once and shared by the tools."""

    def __init__(self, n_customers: int = 5000, seed: int = 42):
        self.df = generate_customers(n_customers, seed)
        self.models = churn_model.train(self.df)
        self.df["churn_probability"] = churn_model.score_all(self.models, self.df)
        self.df["risk_band"] = pd.cut(self.df["churn_probability"], [0, 0.25, 0.5, 1.0],
                                      labels=["low", "medium", "high"], include_lowest=True).astype(str)
        self.retriever = Retriever()


class ToolError(Exception):
    pass


class Tools:
    def __init__(self, ws: Workspace):
        self.ws = ws
        self.retrieved: set[str] = set()

    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, name, None)
        if fn is None or name.startswith("_") or name == "call":
            raise ToolError(f"Unknown tool {name!r}")
        return json.dumps(fn(**args), indent=1, default=str, ensure_ascii=False)

    def get_customer_risk(self, customer_id: str) -> dict:
        rows = self.ws.df[self.ws.df["customer_id"] == customer_id.strip().upper()]
        if rows.empty:
            raise ToolError(f"No customer {customer_id!r}. IDs look like C-00042 (C-00001 to C-{len(self.ws.df):05d}).")
        row = rows.iloc[0]
        profile = {k: row[k] for k in ["contract", "plan", "tenure_months", "monthly_charge_eur", "data_usage_gb",
                                       "dropped_call_rate_pct", "support_calls_90d", "payment_method",
                                       "family_bundle", "city", "age"]}
        return {**churn_model.explain(self.ws.models, row), "profile": profile}

    def segment_churn(self, group_by: str) -> list[dict]:
        if group_by not in GROUPABLE:
            raise ToolError(f"group_by must be one of {GROUPABLE}")
        g = self.ws.df.groupby(group_by, observed=True).agg(
            customers=("customer_id", "size"), churn_rate=("churned", "mean"),
            avg_predicted_risk=("churn_probability", "mean")).round(3)
        return g.reset_index().to_dict(orient="records")

    def search_policies(self, query: str) -> list[dict]:
        hits = self.ws.retriever.search(query, k=4)
        self.retrieved |= {h["id"] for h in hits}
        return hits


def estimate_cost(m: str, i: int, o: int) -> float | None:
    return None if m not in PRICES else (i * PRICES[m][0] + o * PRICES[m][1]) / 1_000_000


class ChurnCopilot:
    def __init__(self, workspace: Workspace | None = None, model: str = DEFAULT_MODEL, effort: str = "medium",
                 max_turns: int = 10, client=None):
        self.ws = workspace or Workspace()
        self.client = client or anthropic.Anthropic()
        self.model, self.effort, self.max_turns = model, effort, max_turns

    def ask(self, question: str) -> AskResult:
        tools = Tools(self.ws)
        messages = [{"role": "user", "content": question}]
        result = AskResult(answer=None)
        extra = {"betas": [FALLBACK_BETA], "fallbacks": "default"} if self.model in FALLBACK_MODELS else {}
        start = time.perf_counter()

        for turn in range(1, self.max_turns + 1):
            resp = self.client.beta.messages.create(
                model=self.model, max_tokens=16000, system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
                **extra)
            result.input_tokens += resp.usage.input_tokens
            result.output_tokens += resp.usage.output_tokens
            if resp.stop_reason == "refusal":
                result.error = "Model declined the request"
                break
            if resp.stop_reason == "max_tokens":
                result.error = "Response hit max_tokens"
                break
            messages.append({"role": "assistant", "content": resp.content})
            uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason == "tool_use" and uses:
                results = []
                for b in uses:
                    try:
                        out, err = tools.call(b.name, dict(b.input)), False
                    except (ToolError, TypeError) as e:
                        out, err = f"Error: {e}", True
                    result.trace.append({"turn": turn, "tool": b.name, "input": dict(b.input), "is_error": err})
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": err})
                messages.append({"role": "user", "content": results})
                continue
            if resp.stop_reason == "pause_turn":
                continue
            text = "".join(b.text for b in resp.content if b.type == "text")
            try:
                result.answer = Answer.model_validate(json.loads(text))
            except (json.JSONDecodeError, ValueError) as e:
                result.error = f"Could not parse answer: {e}"
            break
        else:
            result.error = f"No final answer after {self.max_turns} turns"

        result.retrieved_ids = tools.retrieved
        if result.answer:
            cited = set(result.answer.citations) | {a.policy_chunk_id for a in result.answer.recommended_actions}
            result.unsupported_citations = sorted(cited - tools.retrieved)
        result.latency_s = round(time.perf_counter() - start, 2)
        result.cost_usd = estimate_cost(self.model, result.input_tokens, result.output_tokens)
        return result
