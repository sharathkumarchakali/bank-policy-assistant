"""Run evals.jsonl against the assistant and write eval_report.md. Exits 1 if any case fails.

    python evals.py                          # default answer model
    ANSWER_MODEL=gpt-6.1-sol python evals.py # compare a stronger model
"""
import json
import os
import sys
from pathlib import Path

import app

HERE = Path(__file__).parent
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-6.1-sol")  # stronger model grades the cheaper one
JUDGE = """You check answers from a bank policy assistant. Given the policy excerpts and the answer,
decide whether every factual claim in the answer is supported by the excerpts. Wording may differ; facts may not."""
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"grounded": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["grounded", "reason"],
    "additionalProperties": False,
}


def judge(client, excerpts, answer):
    resp = client.responses.create(
        model=JUDGE_MODEL, instructions=JUDGE,
        input=f"Excerpts:\n{excerpts}\n\nAnswer:\n{answer}",
        text={"format": {"type": "json_schema", "name": "verdict", "schema": JUDGE_SCHEMA, "strict": True}},
    )
    return json.loads(resp.output_text)


def check(case, r, chunks, client):
    """Return a list of failure reasons (empty list = pass)."""
    fails = []
    leaked = [i for i in r["retrieved"] if case["role"] not in chunks[i]["groups"]]
    if leaked:
        fails.append(f"ACCESS LEAK {leaked}")
    if case["expect"] == "refuse":
        if not r["refused"]:
            fails.append("answered, should refuse")
        return fails
    if r["refused"]:
        return fails + ["refused" + ("" if case["cite"] in r["retrieved"] else " (retrieval miss)")]
    if case["cite"] not in r["citations"]:
        fails.append(f"missing citation {case['cite']}")
    fails += [f"missing '{k}'" for k in case["must"] if k.lower() not in r["answer"].lower()]
    if client:
        excerpts = "\n\n".join(f"[{i}] {chunks[i]['text']}" for i in r["retrieved"])
        v = judge(client, excerpts, r["answer"])
        if not v["grounded"]:
            fails.append("not grounded: " + v["reason"])
    return fails


def run(assistant, use_judge=True):
    chunks = {c["id"]: c for c in assistant.chunks}
    cases = [json.loads(l) for l in (HERE / "evals.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for case in cases:
        r = assistant.ask(case["q"], case["role"])
        fails = check(case, r, chunks, assistant.client if use_judge else None)
        rows.append((case, r, fails))
        print(f"{'PASS' if not fails else 'FAIL'}  {case['id']:<24} {'; '.join(fails)}")
    return rows


def report(rows):
    passed = sum(not f for _, _, f in rows)
    lines = [f"# Eval report", "",
             f"Answer model `{app.ANSWER_MODEL}`, judge `{JUDGE_MODEL}`, embeddings `{app.EMBED_MODEL}`.", "",
             f"**{passed}/{len(rows)} cases passed.**", "",
             "| Case | Role | Expected | Result | Notes |", "|---|---|---|---|---|"]
    for case, r, fails in rows:
        note = "; ".join(fails) or ", ".join(r["citations"]) or "refused"
        lines.append(f"| {case['id']} | {case['role']} | {case['expect']} | {'PASS' if not fails else 'FAIL'} | {note} |")
    (HERE / "eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return passed == len(rows)


if __name__ == "__main__":
    ok = report(run(app.Assistant()))
    print(f"\nWrote eval_report.md")
    sys.exit(0 if ok else 1)
