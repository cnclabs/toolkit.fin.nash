"""Reproducible, trace-based evaluation helpers for NASH components."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .extraction import extract_numeric_mentions
from .numeric_similarity import magnitude_similarity


def gold_alignment_index(sentence: str, gold: Any) -> int | None:
    """Locate NumFinE's annotated changed number in this sentence.

    NumFinE stores a number string per sentence, not a full alignment matrix.
    It therefore permits evaluation only of the annotated source/target pair,
    not a claim of exhaustive multi-number alignment gold.
    """
    if gold is None:
        return None
    needle = str(gold).replace(",", "").strip()
    for index, mention in enumerate(extract_numeric_mentions(sentence)):
        if mention.text.replace(",", "").lstrip("$€£¥") == needle:
            return index
    return None


def alignment_evaluate(records: list[dict[str, Any]], metric: Any) -> dict[str, Any]:
    rows = []
    for record in records:
        a, b = record.get("sentence_a"), record.get("sentence_b")
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        left, right = a.get("text"), b.get("text")
        ai, bi = gold_alignment_index(str(left), a.get("num")), gold_alignment_index(str(right), b.get("num"))
        if not isinstance(left, str) or not isinstance(right, str) or ai is None or bi is None:
            continue
        trace = metric.explain(left, right)
        fwd = trace["numeric_alignment"]["alignments_s1_to_s2"]
        rev = trace["numeric_alignment"]["alignments_s2_to_s1"]
        f = next((x for x in fwd if x["source_index"] == ai), None)
        r = next((x for x in rev if x["source_index"] == bi), None)
        rows.append({"id": record.get("id", record.get("pair_id")), "a_to_b_correct": bool(f and f["target_index"] == bi and f["valid"]), "b_to_a_correct": bool(r and r["target_index"] == ai and r["valid"]), "a_to_b_rejected": bool(f and not f["valid"]), "b_to_a_rejected": bool(r and not r["valid"])})
    n = len(rows)
    return {"kind": "nash_alignment_evaluation", "n_annotated_pairs": n, "note": "NumFinE supplies one annotated target number per sentence; this measures correspondence for that target rather than complete all-mention alignment.", "summary": {"a_to_b_accuracy": sum(x["a_to_b_correct"] for x in rows) / n if n else 0.0, "b_to_a_accuracy": sum(x["b_to_a_correct"] for x in rows) / n if n else 0.0, "bidirectional_correct": sum(x["a_to_b_correct"] and x["b_to_a_correct"] for x in rows) / n if n else 0.0, "target_rejected_by_tau": sum(x["a_to_b_rejected"] + x["b_to_a_rejected"] for x in rows) / (2 * n) if n else 0.0}, "rows": rows}


def baseline_evaluate(records: list[dict[str, Any]], metric: Any) -> dict[str, Any]:
    """Compare deployable text/ordered/full scores and annotated-pair oracle."""
    rows = []
    for record in records:
        a, b = record.get("sentence_a"), record.get("sentence_b")
        if not isinstance(a, dict) or not isinstance(b, dict) or not isinstance(a.get("text"), str) or not isinstance(b.get("text"), str):
            continue
        trace = metric.explain(a["text"], b["text"])
        ma, mb = extract_numeric_mentions(a["text"]), extract_numeric_mentions(b["text"])
        ordered = sum(magnitude_similarity(x.value, y.value) for x, y in zip(ma, mb)) / max(len(ma), len(mb), 1) if ma and mb else (1.0 if not ma and not mb else 0.0)
        ai, bi = gold_alignment_index(a["text"], a.get("num")), gold_alignment_index(b["text"], b.get("num"))
        oracle = magnitude_similarity(ma[ai].value, mb[bi].value) if ai is not None and bi is not None else None
        rows.append({"id": record.get("id", record.get("pair_id")), "text_only": trace["baseline_comparison"]["baseline_score"], "ordered_numeric": ordered, "full_nash": trace["baseline_comparison"]["nash_score"], "gold_target_numeric_oracle": oracle})
    def mean(key: str) -> float | None:
        values = [x[key] for x in rows if x.get(key) is not None]
        return sum(values) / len(values) if values else None
    return {"kind": "nash_numerical_baselines", "n_pairs": len(rows), "oracle_note": "gold_target_numeric_oracle uses NumFinE annotations and is an upper-bound diagnostic, not an inference-time baseline.", "summary": {key: mean(key) for key in ("text_only", "ordered_numeric", "full_nash", "gold_target_numeric_oracle")}, "rows": rows}


def benchmark(metric: Any, sentence1: str, sentence2: str, repeats: int = 10) -> dict[str, Any]:
    start = time.perf_counter(); metric.explain(sentence1, sentence2); cold = time.perf_counter() - start
    start = time.perf_counter()
    traces = [metric.explain(sentence1, sentence2) for _ in range(repeats)]
    elapsed = time.perf_counter() - start
    start = time.perf_counter(); json.dumps(traces[-1]); serialization = time.perf_counter() - start
    return {"kind": "nash_runtime_benchmark", "repeats": repeats, "cold_pair_seconds": cold, "warm_pair_seconds": elapsed / repeats, "pairs_per_second": repeats / elapsed if elapsed else 0.0, "trace_serialization_seconds": serialization, "note": "Cold timing includes first scoring call after metric construction; hardware, model cache, device, and backend determine results."}


def write_json(payload: dict[str, Any], path: str | None) -> None:
    if path:
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
