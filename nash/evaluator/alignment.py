from __future__ import annotations

import math

from .backends.base import SimilarityBackend
from .numeric_similarity import magnitude_similarity
from .types import NumericAlignment, NumericMention


def cosine_similarity_matrix(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [[_cosine(left_vector, right_vector) for right_vector in right] for left_vector in left]


def align_numeric_mentions(
    sentence1: str,
    sentence2: str,
    mentions1: list[NumericMention],
    mentions2: list[NumericMention],
    backend: SimilarityBackend,
    threshold: float,
) -> tuple[list[list[float]], list[NumericAlignment], list[NumericAlignment]]:
    if not mentions1 or not mentions2:
        return [[0.0 for _ in mentions2] for _ in mentions1], [], []

    embeddings1 = backend.number_embeddings(sentence1, mentions1)
    embeddings2 = backend.number_embeddings(sentence2, mentions2)
    matrix = cosine_similarity_matrix(embeddings1, embeddings2)

    alignments_1_to_2 = _directional_alignments(mentions1, mentions2, matrix, threshold)
    transposed = [list(row) for row in zip(*matrix)] if matrix else []
    alignments_2_to_1 = _directional_alignments(mentions2, mentions1, transposed, threshold)
    return matrix, alignments_1_to_2, alignments_2_to_1


def _directional_alignments(
    source_mentions: list[NumericMention],
    target_mentions: list[NumericMention],
    similarity_matrix: list[list[float]],
    threshold: float,
) -> list[NumericAlignment]:
    alignments: list[NumericAlignment] = []
    if not target_mentions:
        return alignments

    # NASH specifies a directional argmax for every source mention.  Do not
    # turn this into bipartite/global matching: several source mentions may
    # select the same target and unmatched sources remain in the denominator.
    for source_index, source_mention in enumerate(source_mentions):
        row = similarity_matrix[source_index] if source_index < len(similarity_matrix) else []
        if not row:
            continue
        target_index = max(range(min(len(row), len(target_mentions))), key=lambda index: row[index])
        contextual_similarity = float(row[target_index])
        target_mention = target_mentions[target_index]
        accepted = contextual_similarity >= threshold
        mag_sim = magnitude_similarity(source_mention.value, target_mention.value) if accepted else 0.0
        alignments.append(
            NumericAlignment(
                source_index=source_index,
                target_index=target_index,
                source_text=source_mention.text,
                target_text=target_mention.text,
                contextual_similarity=contextual_similarity,
                magnitude_similarity=mag_sim,
                valid=accepted,
            )
        )
    return alignments


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    dot = sum(left[index] * right[index] for index in range(length))
    left_norm = math.sqrt(sum(value * value for value in left[:length]))
    right_norm = math.sqrt(sum(value * value for value in right[:length]))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(dot / (left_norm * right_norm))
