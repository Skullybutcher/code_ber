"""
Official-style macro F_0.5 metric for the Business Entity Resolution challenge.

F_0.5 = (1.25 * P * R) / (0.25 * P + R), computed per Source-1 entity, then
averaged (macro) across all Source-1 entities. A correctly-predicted empty
set for a true singleton scores 1.0; predicting any match for a true
singleton scores 0.0.
"""
from __future__ import annotations
from typing import Dict, Set, Iterable


def _entity_f_beta(true_set: Set[str], pred_set: Set[str], beta: float = 0.5) -> float:
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0
    if len(pred_set) == 0:
        # no predictions but there were true matches -> precision undefined, recall 0
        return 0.0
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set) if len(true_set) > 0 else 0.0
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta2 = beta * beta
    denom = (beta2 * precision) + recall
    if denom == 0.0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def macro_f05(
    y_true: Dict[str, Set[str]],
    y_pred: Dict[str, Set[str]],
    entity_ids: Iterable[str] | None = None,
) -> float:
    """
    y_true / y_pred: mapping source1_entity_id -> set of matched entity_ids
    (S2-/S3- ids). Missing keys are treated as empty sets.
    entity_ids: the full universe of Source-1 ids to average over (defaults
    to the union of keys in y_true and y_pred). Always pass this explicitly
    in production so entities absent from a dict are still scored 0/1
    correctly, matching the "every Source 1 entity must appear" rule.
    """
    if entity_ids is None:
        entity_ids = set(y_true.keys()) | set(y_pred.keys())
    scores = []
    for eid in entity_ids:
        t = y_true.get(eid, set())
        p = y_pred.get(eid, set())
        scores.append(_entity_f_beta(t, p, beta=0.5))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def per_entity_scores(
    y_true: Dict[str, Set[str]],
    y_pred: Dict[str, Set[str]],
    entity_ids: Iterable[str],
) -> Dict[str, float]:
    return {
        eid: _entity_f_beta(y_true.get(eid, set()), y_pred.get(eid, set()), beta=0.5)
        for eid in entity_ids
    }
