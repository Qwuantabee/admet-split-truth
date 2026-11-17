"""Validation protocols: the five ways a team can talk itself into trusting a model.

Each returns an iterable of (train_idx, test_idx) over the TRAIN frame only. The
prospective error is measured separately, once, against the real future compounds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_split(n: int, k: int = 5, seed: int = 0):
    """Shuffled rows. The protocol most likely to flatter, because near-duplicate
    scaffolds land on both sides."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    for fold in np.array_split(idx, k):
        mask = np.ones(n, dtype=bool)
        mask[fold] = False
        yield np.where(mask)[0], fold


def scaffold_split(scaffolds: np.ndarray, k: int = 5, seed: int = 0):
    """Group by Murcko scaffold, so no scaffold is in two folds. The usual
    'honest' protocol. Scafolds with a single member are lumped into one fold."""
    groups = pd.factorize(pd.Series(scaffolds))[0]
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(groups))
    buckets: list[list[int]] = [[] for _ in range(k)]
    for i, g in enumerate(uniq):
        buckets[i % k].append(int(g))
    for b in buckets:
        te = np.where(np.isin(groups, b))[0]
        tr = np.where(~np.isin(groups, b))[0]
        if len(te) and len(tr):
            yield tr, te


def butina_split(clusters: np.ndarray, k: int = 5, seed: int = 0):
    """Group by Butina (sphere-exclusion) cluster on fingerprint similarity. Finer
    than scaffolds: separates chemotypes that share a scaffold."""
    return scaffold_split(clusters, k=k, seed=seed)


def time_split(order: np.ndarray, k: int = 5):
    """Within-history rolling-origin: train on the first m compounds by ID, test on
    the next block. This is the protocol that most resembles the real task, but it
    still sits inside the training period."""
    idx = np.argsort(order)
    blocks = np.array_split(idx, k + 1)
    for j in range(1, len(blocks)):
        tr = np.concatenate(blocks[:j])
        te = blocks[j]
        if len(tr) >= 50 and len(te) >= 20:
            yield tr, te


def frontier_split(sim_max: np.ndarray, k: int = 5, seed: int = 0):
    """Structural frontier: put the LEAST-similar compounds in the test fold.

    Ordered by maximum similarity to the rest of the training period, so the test
    fold is the part of chemical space the model has least seen. This is the split
    Zheng 2026 finds most pessimistic, and it is included precisely because a
    pessimism-versus-optimism spread across protocols is the finding.
    """
    order = np.argsort(sim_max)          # least similar first
    idx = np.array_split(order, k + 1)
    for j in range(1, len(idx)):
        te = idx[0]
        tr = np.concatenate(idx[1:j + 1]) if j > 1 else idx[1]
        if len(te) >= 20 and len(tr) >= 50:
            yield tr, te
        break                             # a single, stable frontier fold
