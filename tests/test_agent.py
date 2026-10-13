"""Agent loop with a scripted fake client: no API calls, no cost."""

import json
from types import SimpleNamespace as NS

import pytest

from churn_copilot.agent import ANSWER_SCHEMA, FALLBACK_BETA, ChurnCopilot, ToolError, Tools, Workspace


@pytest.fixture(scope="module")
def ws():
    return Workspace(n_customers=1500, seed=7)


def tool_use(id_, name, input_):
    return NS(type="tool_use", id=id_, name=name, input=input_)


def resp(content, stop_reason):
    return NS(content=content, stop_reason=stop_reason, stop_details=None, usage=NS(input_tokens=200, output_tokens=50))


class FakeClient:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})
        return self.responses.pop(0)


def final(citations, action_chunk):
    return resp([NS(type="text", text=json.dumps({
        "answer": "Offer a longer contract with a discount.",
        "recommended_actions": [{"action": "20% off for 12 months on a 12-month contract", "policy_chunk_id": action_chunk}],
        "citations": citations}))], "end_turn")


def test_tools(ws):
    t = Tools(ws)
    risk = json.loads(t.call("get_customer_risk", {"customer_id": "c-00001"}))
    assert risk["customer_id"] == "C-00001" and "profile" in risk
    seg = json.loads(t.call("segment_churn", {"group_by": "contract"}))
    assert {s["contract"] for s in seg} == {"month-to-month", "12-month", "24-month"}
    hits = json.loads(t.call("search_policies", {"query": "discount high risk customer longer contract"}))
    assert hits and t.retrieved == {h["id"] for h in hits}
    with pytest.raises(ToolError):
        t.call("get_customer_risk", {"customer_id": "C-99999"})


def test_agent_answers_with_verified_citations(ws):
    client = FakeClient([
        resp([tool_use("a", "get_customer_risk", {"customer_id": "C-00010"}),
              tool_use("b", "search_policies", {"query": "offers for high risk customers"})], "tool_use"),
        final(["retention-offers#offers-for-high-risk-customers"], "retention-offers#offers-for-high-risk-customers"),
    ])
    r = ChurnCopilot(workspace=ws, client=client).ask("What can we offer C-00010?")
    assert r.error is None and r.answer is not None
    assert r.unsupported_citations == []
    assert [t["tool"] for t in r.trace] == ["get_customer_risk", "search_policies"]
    first = client.calls[0]
    assert first["model"] == "claude-opus-5-5" and first["fallbacks"] == "default" and first["betas"] == [FALLBACK_BETA]
    assert first["output_config"]["format"]["schema"] == ANSWER_SCHEMA
    assert len(client.calls[1]["messages"][-1]["content"]) == 2  # parallel results in one message


def test_agent_flags_citation_it_never_retrieved(ws):
    client = FakeClient([
        resp([tool_use("b", "search_policies", {"query": "network compensation credit amount"})], "tool_use"),
        final(["network-compensation#compensation-amounts"], "gdpr-and-consent#legal-basis"),
    ])
    r = ChurnCopilot(workspace=ws, client=client).ask("How much credit for bad network?")
    assert "gdpr-and-consent#legal-basis" in r.unsupported_citations


def test_haiku_skips_fallbacks(ws):
    client = FakeClient([final([], "x")])
    ChurnCopilot(workspace=ws, client=client, model="claude-haiku-5-5").ask("hi")
    assert "fallbacks" not in client.calls[0] and "betas" not in client.calls[0]


def test_refusal_is_reported(ws):
    r = ChurnCopilot(workspace=ws, client=FakeClient([resp([], "refusal")])).ask("x")
    assert r.answer is None and "declined" in r.error
