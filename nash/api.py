from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .evaluator.datasets import NashDataset
from .evaluator.metric import NASH
from .evaluator.metric_api import infer_backend


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.3763

_DATASET_FILES = {
    "numfine_triplet": {
        "test": ["triplet_test.json"],
        "easy": ["triplet_easy.json"],
        "medium": ["triplet_medium.json"],
        "hard": ["triplet_hard.json"],
        "all": ["triplet_hard.json", "triplet_medium.json", "triplet_easy.json"],
    },
    "numfine_crosspair": {
        "test": ["crosspair_test.json"],
        "all": ["crosspair.json"],
    },
    "numfine_listwise": {
        "test": ["listwise_test.json"],
        "all": ["listwise.json"],
    },
}

_TASK_DATASETS = {
    "triplet": "numfine_triplet",
    "crosspair": "numfine_crosspair",
    "listwise": "numfine_listwise",
}


def load_dataset(name: str, split: str = "test", data_dir: str | None = None) -> NashDataset:
    path = Path(name).expanduser()
    if path.exists() and path.is_file():
        return NashDataset(name=path.stem, records=_read_json_records(path), path=str(path.resolve()))

    dataset_name = _normalize_dataset_name(name)
    split_name = split.strip().lower() if split else "test"
    if dataset_name not in _DATASET_FILES:
        choices = ", ".join(sorted(_DATASET_FILES))
        raise ValueError(f"Unknown dataset '{name}'. Supported datasets: {choices}.")

    roots = [Path(data_dir)] if data_dir else [Path(__file__).resolve().parent / "data"]
    filenames = _DATASET_FILES[dataset_name].get(split_name)
    if filenames is None:
        split_choices = ", ".join(sorted(_DATASET_FILES[dataset_name]))
        raise ValueError(f"Unknown split '{split}' for {dataset_name}. Supported splits: {split_choices}.")

    records: list[dict[str, Any]] = []
    loaded_paths: list[str] = []
    expected_paths: list[str] = []
    for root in roots:
        for filename in filenames:
            path = root / filename
            expected_paths.append(str(path))
            if path.exists():
                records.extend(_read_json_records(path))
                loaded_paths.append(str(path))

    if not records:
        expected = "\n  - ".join(expected_paths)
        raise FileNotFoundError(
            f"Could not find local NumFinE data for '{dataset_name}' split '{split_name}'.\n"
            f"Expected JSON files:\n  - {expected}"
        )

    return NashDataset(name=dataset_name, records=records, path=";".join(loaded_paths))


def score(
    sentence1: str,
    sentence2: str,
    model: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    **kwargs: Any,
) -> dict[str, Any]:
    backend = kwargs.pop("backend", None) or infer_backend(model)
    metric = NASH(
        model_name=model,
        backend=backend,
        threshold=threshold,
        device=kwargs.pop("device", None),
        batch_size=int(kwargs.pop("batch_size", 32)),
    )
    return _score_with_metric(metric, sentence1, sentence2, model=model, backend=backend)


def evaluate(
    dataset: Any,
    protocol: str,
    model: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    **kwargs: Any,
) -> dict[str, Any]:
    task = _normalize_task(protocol)
    records = _records_from_dataset(dataset)
    limit = kwargs.pop("limit", None)
    if limit is not None:
        records = records[: int(limit)]

    scorer = kwargs.pop("scorer", None)
    backend = kwargs.pop("backend", None) or infer_backend(model)
    metric = None if scorer else NASH(
        model_name=model,
        backend=backend,
        threshold=threshold,
        device=kwargs.pop("device", None),
        batch_size=int(kwargs.pop("batch_size", 32)),
    )

    def run_pair(sentence1: str, sentence2: str) -> dict[str, Any]:
        if scorer:
            return scorer(sentence1, sentence2)
        assert metric is not None
        return _score_with_metric(metric, sentence1, sentence2, model=model, backend=backend)

    rows = [_evaluate_record(record, index, task, run_pair, model, backend) for index, record in enumerate(records)]
    rows = [row for row in rows if row]
    return {
        "task": task,
        "model": model,
        "baseline_label": model,
        "backend": backend,
        "rows": rows,
        "summary": {
            "n_records": len(rows),
            "n_pair_results": sum(len(row.get("pair_results", [])) for row in rows),
        },
    }


def export_web_json(result: dict[str, Any], out_path: str) -> None:
    task = _normalize_task(str(result.get("task", "")))
    model = str(result.get("model", DEFAULT_MODEL))
    backend = str(result.get("backend", infer_backend(model)))
    baseline_label = str(result.get("baseline_label", model))
    payload = {
        "kind": "nash_evaluation_results",
        "task": task,
        "rows": [
            _web_row(row, task=task, model=model, baseline_label=baseline_label, backend=backend)
            for row in result.get("rows", [])
            if isinstance(row, dict)
        ],
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _score_with_metric(metric: NASH, sentence1: str, sentence2: str, model: str, backend: str) -> dict[str, Any]:
    trace = metric.explain(sentence1, sentence2)
    return _pair_result_from_trace(trace, model=model, backend=backend)


def _pair_result_from_trace(trace: dict[str, Any], model: str, backend: str) -> dict[str, Any]:
    input_data = trace.get("input", {})
    masking = trace.get("masking", {})
    numeric = trace.get("numeric_similarity", {})
    textual = trace.get("textual_similarity", {})
    comparison = trace.get("baseline_comparison", {})
    extraction = trace.get("numeric_extraction", {})
    alignment = trace.get("numeric_alignment", {})
    aggregation = trace.get("aggregation", {})
    text_mass = float(aggregation.get("idf_text_mass", 0.0))
    numeric_mass = float(aggregation.get("idf_numeric_mass", 0.0))
    return {
        "sentence1": input_data.get("sentence1", ""),
        "sentence2": input_data.get("sentence2", ""),
        "model": model,
        "baseline_label": model,
        "backend": backend,
        "baseline_score": float(comparison.get("baseline_score", 0.0)),
        "nash_score": float(comparison.get("nash_score", 0.0)),
        "text_score": float(textual.get("score", 0.0)),
        "numeric_score": float(numeric.get("score", 0.0)),
        "alpha": text_mass / max(text_mass + numeric_mass, 1e-12),
        "numeric_score_s1_to_s2": float(numeric.get("directional_s1_to_s2", 0.0)),
        "numeric_score_s2_to_s1": float(numeric.get("directional_s2_to_s1", 0.0)),
        "masked_sentence1": masking.get("masked_sentence1", ""),
        "masked_sentence2": masking.get("masked_sentence2", ""),
        "numbers_sentence1": extraction.get("sentence1_numbers", []),
        "numbers_sentence2": extraction.get("sentence2_numbers", []),
        "alignment": {
            "matrix": alignment.get("similarity_matrix", []),
            "selected_alignment_edges": _selected_edges(alignment.get("valid_alignments_s1_to_s2", [])),
            "alignments_s1_to_s2": alignment.get("alignments_s1_to_s2", []),
            "alignments_s2_to_s1": alignment.get("alignments_s2_to_s1", []),
            "threshold": alignment.get("threshold"),
        },
        "diagnostics": trace.get("diagnostics", {}),
    }


def _selected_edges(edges: Any) -> list[dict[str, Any]]:
    selected = []
    if not isinstance(edges, list):
        return selected
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        selected.append(
            {
                "source_index": int(edge.get("source_index", 0)),
                "target_index": int(edge.get("target_index", 0)),
                "source_text": edge.get("source_text", ""),
                "target_text": edge.get("target_text", ""),
                "similarity": float(edge.get("contextual_similarity", edge.get("similarity", 0.0))),
                "aligned": bool(edge.get("valid", True)),
            }
        )
    return selected


def _evaluate_record(
    record: dict[str, Any],
    index: int,
    task: str,
    run_pair: Callable[[str, str], dict[str, Any]],
    model: str,
    backend: str,
) -> dict[str, Any]:
    if task == "triplet":
        anchor = _sentence_text(record.get("anchor") or record.get("sentence_a"))
        s1 = _sentence_text(record.get("s1") or record.get("positive") or record.get("sentence_b"))
        s2 = _sentence_text(record.get("s2") or record.get("negative") or record.get("sentence_c"))
        if not (anchor and s1 and s2):
            return {}
        return {
            **_base_row(record, index, task, model, backend),
            "anchor": anchor,
            "s1": s1,
            "s2": s2,
            "pair_results": [
                {"label": "S1", **run_pair(anchor, s1)},
                {"label": "S2", **run_pair(anchor, s2)},
            ],
        }

    if task == "crosspair":
        a1 = _sentence_text(record.get("pair_a_sentence1") or record.get("sentence_a"))
        a2 = _sentence_text(record.get("pair_a_sentence2") or record.get("sentence_b"))
        b1 = _sentence_text(record.get("pair_b_sentence1") or record.get("sentence_c"))
        b2 = _sentence_text(record.get("pair_b_sentence2") or record.get("sentence_d"))
        if not (a1 and a2 and b1 and b2):
            return {}
        return {
            **_base_row(record, index, task, model, backend),
            "pair_a_sentence1": a1,
            "pair_a_sentence2": a2,
            "pair_b_sentence1": b1,
            "pair_b_sentence2": b2,
            "pair_results": [
                {"label": "Pair A", **run_pair(a1, a2)},
                {"label": "Pair B", **run_pair(b1, b2)},
            ],
        }

    anchor = _sentence_text(record.get("anchor") or record.get("sentence_a"))
    candidates = record.get("candidates") or record.get("variants")
    if not anchor or not isinstance(candidates, list):
        return {}
    pair_results = []
    candidate_texts = []
    for candidate_index, candidate in enumerate(candidates):
        text = _sentence_text(candidate)
        if not text:
            continue
        candidate_texts.append(text)
        pair_results.append({"label": f"Candidate {candidate_index + 1}", **run_pair(anchor, text)})
    if not pair_results:
        return {}
    return {
        **_base_row(record, index, task, model, backend),
        "anchor": anchor,
        "candidates": candidate_texts,
        "pair_results": pair_results,
    }


def _base_row(record: dict[str, Any], index: int, task: str, model: str, backend: str) -> dict[str, Any]:
    row_id = record.get("id", record.get("pair_id", f"{task}_{index + 1:03d}"))
    category = str(record.get("category", task.title()))
    return {
        "id": str(row_id),
        "title": str(record.get("title", f"{category} {task} {row_id}")),
        "task": task,
        "model": model,
        "baseline_label": model,
        "backend": backend,
    }


def _web_row(row: dict[str, Any], task: str, model: str, baseline_label: str, backend: str) -> dict[str, Any]:
    exported = {
        "id": str(row.get("id", "")),
        "title": str(row.get("title", row.get("id", ""))),
        "task": task,
        "model": str(row.get("model", model)),
        "baseline_label": str(row.get("baseline_label", baseline_label)),
        "backend": str(row.get("backend", backend)),
        "pair_results": [_web_pair(pair) for pair in row.get("pair_results", []) if isinstance(pair, dict)],
    }
    for key in (
        "anchor",
        "s1",
        "s2",
        "pair_a_sentence1",
        "pair_a_sentence2",
        "pair_b_sentence1",
        "pair_b_sentence2",
        "candidates",
    ):
        if key in row:
            exported[key] = row[key]
    return exported


def _web_pair(pair: dict[str, Any]) -> dict[str, Any]:
    exported = {
        "label": str(pair.get("label", "")),
        "sentence1": str(pair.get("sentence1", "")),
        "sentence2": str(pair.get("sentence2", "")),
        "baseline_score": float(pair.get("baseline_score", 0.0)),
        "nash_score": float(pair.get("nash_score", 0.0)),
        "text_score": float(pair.get("text_score", 0.0)),
        "numeric_score": float(pair.get("numeric_score", 0.0)),
    }
    for key in ("masked_sentence1", "masked_sentence2", "numbers_sentence1", "numbers_sentence2", "alignment"):
        if key in pair:
            exported[key] = pair[key]
    return exported


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "data", "examples", "rows"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [item for item in rows if isinstance(item, dict)]
    raise ValueError(f"Unsupported dataset JSON shape in {path}.")


def _records_from_dataset(dataset: Any) -> list[dict[str, Any]]:
    records = getattr(dataset, "records", dataset)
    if not isinstance(records, list):
        records = list(records)
    return [record for record in records if isinstance(record, dict)]


def _sentence_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "sentence", "value"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
    return ""


def _normalize_dataset_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    return _TASK_DATASETS.get(normalized, normalized)


def _normalize_task(protocol: str) -> str:
    task = protocol.strip().lower().replace("-", "").replace("_", "")
    if task not in {"triplet", "crosspair", "listwise"}:
        raise ValueError("protocol/task must be one of: triplet, crosspair, listwise.")
    return task
