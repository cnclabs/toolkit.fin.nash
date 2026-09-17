from __future__ import annotations

import re
import warnings
from typing import Any

from .aggregation import combine_scores, compute_alpha, compute_idf_masses
from .alignment import align_numeric_mentions
from .backends.base import SimilarityBackend
from .backends.bertscore_backend import BERTScoreBackend
from .backends.sentence_transformer_backend import SentenceTransformerBackend
from .extraction import extract_numeric_mentions
from .masking import mask_numbers
from .numeric_similarity import bidirectional_numeric_score, scale_adaptive_normalizer
from .types import ScoreResult


DEFAULT_THRESHOLD = 0.3763


class NASH:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        backend: str | SimilarityBackend = "sentence-transformer",
        threshold: float | str | None = None,
        idf: dict[str, float] | None = None,
        idf_path: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.threshold = self._resolve_threshold(threshold)
        self.idf = self._load_idf(idf, idf_path)
        self.backend = self._resolve_backend(backend, model_name, device, batch_size)

    def score_pair(self, sentence1: str, sentence2: str) -> ScoreResult:
        numbers1 = extract_numeric_mentions(sentence1)
        numbers2 = extract_numeric_mentions(sentence2)
        masked1 = mask_numbers(sentence1, numbers1)
        masked2 = mask_numbers(sentence2, numbers2)

        baseline_score = self.baseline_score(sentence1, sentence2)
        text_score = self.backend.text_similarity([masked1], [masked2])[0]
        matrix, alignments_1_to_2, alignments_2_to_1 = align_numeric_mentions(
            sentence1,
            sentence2,
            numbers1,
            numbers2,
            self.backend,
            self.threshold,
        )
        numeric_score, directional_1_to_2, directional_2_to_1 = bidirectional_numeric_score(
            numbers1,
            numbers2,
            alignments_1_to_2,
            alignments_2_to_1,
        )
        alpha = compute_alpha(masked1, masked2, self.idf)
        final_score = combine_scores(alpha, text_score, numeric_score)

        return ScoreResult(
            sentence1=sentence1,
            sentence2=sentence2,
            final_score=final_score,
            text_score=float(text_score),
            numeric_score=float(numeric_score),
            alpha=float(alpha),
            masked_sentence1=masked1,
            masked_sentence2=masked2,
            numbers_sentence1=numbers1,
            numbers_sentence2=numbers2,
            alignment_matrix=matrix,
            alignments_s1_to_s2=alignments_1_to_2,
            alignments_s2_to_s1=alignments_2_to_1,
            baseline_score=float(baseline_score),
            numeric_score_s1_to_s2=float(directional_1_to_2),
            numeric_score_s2_to_s1=float(directional_2_to_1),
        )

    def __call__(self, sentences1: list[str], sentences2: list[str]) -> list[ScoreResult]:
        if len(sentences1) != len(sentences2):
            raise ValueError("sentences1 and sentences2 must have the same length.")
        return [self.score_pair(sentence1, sentence2) for sentence1, sentence2 in zip(sentences1, sentences2)]

    def baseline_score(self, sentence1: str, sentence2: str) -> float:
        return float(self.backend.text_similarity([sentence1], [sentence2])[0])

    def explain(self, sentence1: str, sentence2: str) -> dict[str, Any]:
        result = self.score_pair(sentence1, sentence2)
        text_mass, numeric_mass = compute_idf_masses(
            result.masked_sentence1,
            result.masked_sentence2,
            self.idf,
        )
        accepted = sum(item.valid for item in result.alignments_s1_to_s2) + sum(item.valid for item in result.alignments_s2_to_s1)
        rejected = len(result.alignments_s1_to_s2) + len(result.alignments_s2_to_s1) - accepted
        return {
            "schema_version": "1.1",
            "input": {
                "sentence1": sentence1,
                "sentence2": sentence2,
            },
            "numeric_extraction": {
                "sentence1_numbers": [mention.to_dict() for mention in result.numbers_sentence1],
                "sentence2_numbers": [mention.to_dict() for mention in result.numbers_sentence2],
            },
            "masking": {
                "masked_sentence1": result.masked_sentence1,
                "masked_sentence2": result.masked_sentence2,
            },
            "tokenization": {
                "sentence1_tokens": _tokenization_for_backend(self.backend, sentence1, result.numbers_sentence1)[0],
                "sentence2_tokens": _tokenization_for_backend(self.backend, sentence2, result.numbers_sentence2)[0],
                "number_token_spans_sentence1": _tokenization_for_backend(
                    self.backend,
                    sentence1,
                    result.numbers_sentence1,
                )[1],
                "number_token_spans_sentence2": _tokenization_for_backend(
                    self.backend,
                    sentence2,
                    result.numbers_sentence2,
                )[1],
            },
            "textual_similarity": {
                "score": result.text_score,
                "backend": self.backend.name,
            },
            "numeric_alignment": {
                "method": "directional_argmax_threshold_gating",
                "threshold": self.threshold,
                "similarity_matrix": result.alignment_matrix,
                "alignments_s1_to_s2": [
                    alignment.to_dict() for alignment in result.alignments_s1_to_s2
                ],
                "alignments_s2_to_s1": [
                    alignment.to_dict() for alignment in result.alignments_s2_to_s1
                ],
                "valid_alignments_s1_to_s2": [
                    alignment.to_dict() for alignment in result.alignments_s1_to_s2 if alignment.valid
                ],
                "valid_alignments_s2_to_s1": [
                    alignment.to_dict() for alignment in result.alignments_s2_to_s1 if alignment.valid
                ],
            },
            "numeric_similarity": {
                "pairwise_s1_to_s2": _pairwise_numeric_details(
                    result.alignments_s1_to_s2,
                    result.numbers_sentence1,
                    result.numbers_sentence2,
                ),
                "pairwise_s2_to_s1": _pairwise_numeric_details(
                    result.alignments_s2_to_s1,
                    result.numbers_sentence2,
                    result.numbers_sentence1,
                ),
                "directional_s1_to_s2": result.numeric_score_s1_to_s2,
                "directional_s2_to_s1": result.numeric_score_s2_to_s1,
                "score": result.numeric_score,
            },
            "aggregation": {
                "idf_text_mass": text_mass,
                "idf_numeric_mass": numeric_mass,
                "final_score": result.final_score,
                "identity": "final_score = alpha * text_score + (1 - alpha) * numeric_score",
                "numeric_weight": 1.0 - result.alpha,
            },
            "diagnostics": {
                "detected_mentions_sentence1": len(result.numbers_sentence1),
                "detected_mentions_sentence2": len(result.numbers_sentence2),
                "accepted_directional_alignments": accepted,
                "rejected_by_threshold": rejected,
                "summary": _diagnostic_summary(result, self.threshold),
            },
            "baseline_comparison": {
                "baseline_score": result.baseline_score,
                "nash_score": result.final_score,
                "delta": result.final_score - (result.baseline_score or 0.0),
            },
        }


    def _resolve_threshold(self, threshold: float | str | None) -> float:
        if threshold is None:
            return DEFAULT_THRESHOLD
        if threshold == "auto":
            warnings.warn(
                "threshold='auto' is not calibrated yet; using the default threshold.",
                RuntimeWarning,
                stacklevel=2,
            )
            return DEFAULT_THRESHOLD
        return float(threshold)

    def _resolve_backend(
        self,
        backend: str | SimilarityBackend,
        model_name: str,
        device: str | None,
        batch_size: int,
    ) -> SimilarityBackend:
        if not isinstance(backend, str):
            return backend
        if backend == "sentence-transformer":
            return SentenceTransformerBackend(model_name=model_name, device=device, batch_size=batch_size)
        if backend == "bertscore":
            return BERTScoreBackend(model_name=model_name, device=device, batch_size=batch_size)
        raise ValueError(
            f"Unsupported backend '{backend}'. Supported backends: sentence-transformer, bertscore."
        )

    def _load_idf(self, idf: dict[str, float] | None, idf_path: str | None) -> dict[str, float] | None:
        if idf is not None:
            return idf
        if idf_path is None:
            return None
        import json

        with open(idf_path, "r", encoding="utf-8") as file:
            loaded = json.load(file)
        return {str(key): float(value) for key, value in loaded.items()}


def _diagnostic_summary(result: ScoreResult, threshold: float) -> list[str]:
    """Trace-based observations, not assertions about semantic correctness."""
    messages: list[str] = []
    for direction, alignments in (("Sentence A → B", result.alignments_s1_to_s2), ("Sentence B → A", result.alignments_s2_to_s1)):
        for item in alignments:
            if not item.valid:
                messages.append(f"{direction}: {item.source_text!r} selected {item.target_text!r}, but contextual similarity {item.contextual_similarity:.4f} is below tau={threshold:.4f}; it contributes 0.")
    if not result.numbers_sentence1 or not result.numbers_sentence2:
        messages.append("Only one side (or neither side) contains detected numbers; the numeric channel follows NASH's documented edge-case score.")
    if result.numeric_score < result.text_score - 0.25:
        messages.append("The numeric channel is substantially lower than masked-text similarity; inspect accepted magnitude similarities and rejected matches.")
    if abs(result.alpha - 0.5) > 0.25:
        messages.append(f"IDF mass makes aggregation text-weighted at alpha={result.alpha:.3f}.")
    return messages or ["All directional argmax alignments passed tau; inspect magnitude similarities and IDF weighting for the score decomposition."]


def _basic_tokens(sentence: str) -> list[str]:
    return re.findall(r"\[NUM\]|\w+|[^\w\s]", sentence, flags=re.UNICODE)


def _tokenization_for_backend(
    backend: SimilarityBackend,
    sentence: str,
    mentions: list[Any],
) -> tuple[list[str], list[list[int]]]:
    tokenization_info = getattr(backend, "tokenization_info", None)
    if callable(tokenization_info):
        return tokenization_info(sentence, mentions)
    return _basic_tokens(sentence), _number_token_spans(sentence, mentions)


def _number_token_spans(sentence: str, mentions: list[Any]) -> list[list[int]]:
    tokens = list(re.finditer(r"\w+|[^\w\s]", sentence, flags=re.UNICODE))
    spans: list[list[int]] = []
    for mention in mentions:
        indices = [
            index
            for index, token in enumerate(tokens)
            if not (token.end() <= mention.start or token.start() >= mention.end)
        ]
        if indices:
            spans.append([indices[0], indices[-1] + 1])
        else:
            spans.append([])
    return spans


def _pairwise_numeric_details(
    alignments: list[Any],
    source_mentions: list[Any],
    target_mentions: list[Any],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for alignment in alignments:
        source_value = float(source_mentions[alignment.source_index].value)
        target_value = float(target_mentions[alignment.target_index].value)
        normalizer = scale_adaptive_normalizer(source_value, target_value)
        details.append(
            {
                "source_index": alignment.source_index,
                "target_index": alignment.target_index,
                "source_text": alignment.source_text,
                "target_text": alignment.target_text,
                "source_value": source_value,
                "target_value": target_value,
                "absolute_difference": abs(source_value - target_value),
                "normalization_factor": normalizer,
                "contextual_similarity": alignment.contextual_similarity,
                "pairwise_numeric_similarity": alignment.magnitude_similarity,
                "valid": alignment.valid,
            }
        )
    return details
