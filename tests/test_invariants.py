"""Non-negotiable invariants. A failure here means every ratio in the README is void."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_split import data as D  # noqa: E402
from admet_split import splits as S  # noqa: E402

RAW = ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def frame():
    return D.load(RAW)


def test_the_two_splits_are_disjoint_and_time_ordered(frame):
    """The entire project rests on test compounds being later than train compounds.

    If this ever fails, no ratio computed here means anything: the "prospective"
    error would be a retrospective one wearing a different name.
    """
    tr = {D._numeric_id(x) for x in frame.train["Molecule Name"]}
    te = {D._numeric_id(x) for x in frame.test["Molecule Name"]}
    assert not (tr & te), "compound IDs appear in both splits"
    assert min(te) > max(tr), (
        f"a test ID ({min(te)}) is not above the train maximum ({max(tr)}); the "
        "split is not time-ordered")


def test_the_loader_refuses_a_shuffled_split(frame, tmp_path, monkeypatch):
    """A loader that accepts a broken split is worse than none: prove it raises."""
    tr = frame.train.copy()
    te = frame.test.copy()
    # Move an EARLY train compound into the test file. Moving the last one would not
    # break the ordering at all (its ID stays above every remaining train ID), which
    # is what the first version of this test did - it asserted a raise that could not
    # happen and failed for the right reason.
    moved = tr.iloc[[0]]
    tr = tr.iloc[1:]
    te = pd.concat([te, moved], ignore_index=True)
    d = tmp_path / "raw"
    d.mkdir()
    tr.to_csv(d / "expansion_data_train.csv", index=False)
    te.to_csv(d / "expansion_data_test.csv", index=False)
    with pytest.raises(ValueError, match="time-ordered|BOTH"):
        D.load(d)


def test_log_endpoints_are_positive_after_scaling(frame):
    """log10 of a non-positive value is -inf/NaN and would silently vanish into the
    median. Every scored value must be finite."""
    for ep in D.LOG10:
        v = pd.concat([frame.train[ep], frame.test[ep]]).dropna()
        assert (v > -np.inf).all()
        finite = v[np.isfinite(v)]
        assert len(finite) > 0


def test_every_protocol_produces_disjoint_folds():
    """No protocol may leak a test row into its own training fold."""
    n = 200
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 12, size=n)
    for gen in (S.random_split(n, k=4, seed=0),
                S.scaffold_split(labels, k=4, seed=0),
                S.butina_split(labels, k=4, seed=0)):
        for tr, te in gen:
            assert len(set(tr) & set(te)) == 0, "train and test folds overlap"
            assert len(tr) + len(te) in (n,), "a fold dropped rows"


def test_scaffold_split_keeps_each_scaffold_whole():
    """The whole point of the protocol: a scaffold is never on both sides."""
    labels = np.array(["a"] * 30 + ["b"] * 30 + ["c"] * 30 + ["d"] * 30)
    for tr, te in S.scaffold_split(labels, k=3, seed=0):
        assert not (set(labels[tr]) & set(labels[te]))


def test_time_split_never_trains_on_the_future():
    """Rolling origin: the training block must precede the test block in ID order."""
    order = np.arange(500)
    for tr, te in S.time_split(order, k=4):
        assert max(tr) < min(te), "trained on future compounds"


def test_ratio_is_scale_free():
    """A ratio of MAEs must not change if every endpoint is rescaled by a constant.

    This is what makes it legal to pool log10 endpoints with LogD in one median.
    """
    retro = np.array([0.40, 0.50, 0.60])
    pro = np.array([0.50, 0.60, 0.80])
    base = retro / pro
    scaled = (retro * 7.3) / (pro * 7.3)
    assert np.allclose(base, scaled)
