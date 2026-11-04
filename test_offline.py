"""Offline check of the guardrails, no API key needed: python test_offline.py"""
import json
import re
from types import SimpleNamespace as NS

import numpy as np

import app
import evals


class FakeClient:
    """Bag-of-words 'embeddings' and a model that echoes the top excerpt, plus one made-up citation."""

    def __init__(self, cite_mode="real"):
        self.cite_mode, self.model_calls = cite_mode, 0
        self.embeddings = NS(create=self._embed)
        self.responses = NS(create=self._respond)

    def _embed(self, model, input):
        def vec(t):
            v = np.zeros(512)
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[hash(w) % 512] += 1
            return v + 1e-9
        return NS(data=[NS(embedding=vec(t)) for t in input])

    def _respond(self, model, instructions, input, text):
        self.model_calls += 1
        ids = re.findall(r"^\[([^\]]+)\]", input, re.M)
        cites = ["made-up#section"] + (ids[:1] if self.cite_mode == "real" else [])
        return NS(output_text=json.dumps({"answer": input, "citations": cites, "refused": False}))


def test():
    a = app.Assistant(FakeClient())
    chunks = {c["id"]: c for c in a.chunks}
    cases = [json.loads(l) for l in (evals.HERE / "evals.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    # 1. Branch staff never retrieve restricted chunks, whatever they ask.
    for c in cases:
        for cid in a.retrieve(c["q"], "branch"):
            assert "branch" in cid["groups"], cid["id"]
    # 2. Risk staff can reach the restricted procedure.
    assert any(h["id"].startswith("fraud-escalation#") for h in a.retrieve("FRMU fraud escalation threshold lakh", "risk"))
    # 3. Made-up citations are dropped; an answer left with no real citation is refused.
    r = a.ask("How often do we update KYC for a high-risk customer?", "branch")
    assert "made-up#section" not in r["citations"] and r["citations"], r
    r = app.Assistant(FakeClient(cite_mode="fake")).ask("shadow credit working days", "branch")
    assert r["refused"] and r["answer"] == app.NOT_FOUND, r
    # 4. Unknown role sees nothing and the model is never called.
    fc = FakeClient()
    r = app.Assistant(fc).ask("shadow credit working days", "intern")
    assert r["refused"] and fc.model_calls == 0, r
    # 5. Eval checker flags a leak and a missing citation.
    fake = {"answer": "x", "citations": [], "refused": False, "retrieved": ["fraud-escalation#escalation-thresholds"]}
    fails = evals.check({"role": "branch", "expect": "answer", "cite": "a#b", "must": []}, fake, chunks, None)
    assert any("ACCESS LEAK" in f for f in fails) and any("missing citation" in f for f in fails), fails
    print(f"offline checks passed ({len(a.chunks)} chunks, {len(cases)} eval cases)")


if __name__ == "__main__":
    test()
