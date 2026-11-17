"""Invariants for the third model family.

These exist because the graph featuriser has a failure mode the fingerprint path does
not: a declared feature width that drifts from the features actually produced. The
first version of `gnn.py` had exactly that bug (NODE_DIM counted one one-hot block
without its "other" bucket) and it surfaced only as a matrix-shape error deep inside
the first fit. It is an invariant now.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("rdkit")
from admet_split import gnn as G  # noqa: E402

SMI = "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"      # acetazolamide-like, rings + heteroatoms


def test_declared_feature_widths_match_what_is_produced():
    x, ei, ea = G.mol_to_graph(SMI)
    assert x.shape[1] == G.NODE_DIM, (
        f"atom_features produces {x.shape[1]} columns but NODE_DIM says {G.NODE_DIM}; "
        "the first linear layer would be built at the wrong width")
    assert ea.shape[1] == G.EDGE_DIM, (
        f"bond_features produces {ea.shape[1]} columns but EDGE_DIM says {G.EDGE_DIM}")


def test_graph_matches_the_molecule():
    from rdkit import Chem

    m = Chem.MolFromSmiles(SMI)
    x, ei, ea = G.mol_to_graph(SMI)
    assert x.shape[0] == m.GetNumAtoms()
    assert ei.shape[1] == 2 * m.GetNumBonds(), "bonds must appear in both directions"
    assert ea.shape[0] == ei.shape[1]
    assert ei.max() < x.shape[0], "an edge points at an atom that does not exist"


def test_unparseable_smiles_is_dropped_not_faked():
    assert G.mol_to_graph("not_a_molecule") is None
    graphs, keep = G.featurize(["CCO", "not_a_molecule", "c1ccccc1"])
    assert len(graphs) == 2
    assert list(keep) == [0, 2], "the kept index must point back at the original rows"


def test_batching_does_not_mix_molecules():
    torch = pytest.importorskip("torch")
    gs = [G.mol_to_graph(s) for s in ("CCO", SMI, "c1ccccc1")]
    x, ei, ea, batch, ng = G.collate(gs)
    assert ng == 3
    assert x.shape[0] == sum(g[0].shape[0] for g in gs)
    # every edge must stay inside the molecule it came from
    assert torch.equal(batch[ei[0]], batch[ei[1]]), "an edge crosses two molecules"


def test_prediction_is_deterministic_and_on_the_target_scale():
    pytest.importorskip("torch")
    smis = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "c1ccccc1C", "CCN", "CCCN", "CCOC"]
    gs = [G.mol_to_graph(s) for s in smis]
    y = np.array([-0.3, 0.2, 0.8, 2.1, 2.6, -0.9, -0.4, 0.5], dtype=float)
    a = G.fit_predict_gnn(gs, y, gs, seed=0, epochs=3)
    b = G.fit_predict_gnn(gs, y, gs, seed=0, epochs=3)
    assert np.allclose(a, b), "same seed must give the same predictions"
    assert len(a) == len(gs)
    assert np.isfinite(a).all()
    # the target standardisation must be undone: predictions sit near the data, not
    # near zero-mean unit-variance space
    assert abs(float(a.mean()) - float(y.mean())) < 2.0


def test_the_gnn_grid_defines_no_second_protocol_or_ratio():
    """A second definition of a split or of the ratio is a bug, not a convenience."""
    src = (ROOT / "scripts" / "run_gnn_grid.py").read_text()
    assert "import run_grid as RG" in src
    assert "from admet_split import splits as S" in src
    for forbidden in ("def random_split", "def scaffold_split", "def butina_split",
                      "def time_split", "def frontier_split", "def _mae",
                      "def morgan_fps"):
        assert forbidden not in src, (
            f"run_gnn_grid.py redefines {forbidden}; the protocols and the ratio have "
            "exactly one definition each")
