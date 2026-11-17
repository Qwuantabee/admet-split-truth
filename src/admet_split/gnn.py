"""A third model family: a message-passing network on the molecular graph.

Why this exists: the two families already measured (ridge and random forest) both
consume the same 2048-bit ECFP4 vector, so their errors are correlated by
construction. A protocol that flatters a fingerprint model might not flatter a model
that reads the graph directly. This module supplies that third family and nothing
else - the splits, the loader and the ratio definition are reused unchanged from
`data.py` and `splits.py`.

Deliberately small and untuned: one architecture, one set of hyper-parameters, fixed
seed. If it does not beat the forest, that is the result, not a reason to search.
"""
from __future__ import annotations

import numpy as np

ATOMS = (6, 7, 8, 9, 15, 16, 17, 35, 53)       # C N O F P S Cl Br I
HYB = ("SP", "SP2", "SP3", "SP3D", "SP3D2")
BOND = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")

# Each _onehot block is len(choices) + 1 wide (the +1 is the "other" bucket), so the
# widths below must track atom_features() exactly: atom number, total degree, formal
# charge, hybridisation, then three scalars.
NODE_DIM = (len(ATOMS) + 1) + (5 + 1) + (4 + 1) + (len(HYB) + 1) + 3
EDGE_DIM = (len(BOND) + 1) + 2


def _onehot(value, choices):
    v = [0.0] * (len(choices) + 1)
    try:
        v[choices.index(value)] = 1.0
    except ValueError:
        v[-1] = 1.0
    return v


def atom_features(a) -> list[float]:
    return (
        _onehot(a.GetAtomicNum(), list(ATOMS))                  # len(ATOMS)+1
        + _onehot(a.GetTotalDegree(), [0, 1, 2, 3, 4])          # 6
        + _onehot(a.GetFormalCharge(), [-1, 0, 1, 2])           # 5
        + _onehot(str(a.GetHybridization()), list(HYB))         # len(HYB)+1 -> 6
        + [float(a.GetIsAromatic()), float(a.IsInRing()),
           a.GetTotalNumHs() / 4.0]
    )


def bond_features(b) -> list[float]:
    return (_onehot(str(b.GetBondType()), list(BOND))
            + [float(b.GetIsConjugated()), float(b.IsInRing())])


def mol_to_graph(smiles: str):
    """(node matrix, edge index [2,E], edge attr [E,F]) or None if unparseable."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(str(smiles))
    if m is None or m.GetNumAtoms() == 0:
        return None
    x = np.array([atom_features(a) for a in m.GetAtoms()], dtype=np.float32)
    src, dst, ea = [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        src += [i, j]
        dst += [j, i]
        ea += [f, f]
    if not src:                                   # single-atom molecule
        ei = np.zeros((2, 0), dtype=np.int64)
        ea_arr = np.zeros((0, EDGE_DIM), dtype=np.float32)
    else:
        ei = np.array([src, dst], dtype=np.int64)
        ea_arr = np.array(ea, dtype=np.float32)
    return x, ei, ea_arr


def featurize(smiles_iterable):
    """Graphs for every SMILES; returns (graphs, keep_index) like morgan_fps does."""
    graphs, keep = [], []
    for i, s in enumerate(smiles_iterable):
        g = mol_to_graph(s)
        if g is None:
            continue
        graphs.append(g)
        keep.append(i)
    return graphs, np.array(keep, dtype=int)


# ------------------------------------------------------------------ torch model
def _torch():
    import torch
    return torch


def collate(graphs):
    """Disjoint-union a list of graphs into one batch."""
    import torch

    xs, eis, eas, batch = [], [], [], []
    off = 0
    for gi, (x, ei, ea) in enumerate(graphs):
        xs.append(torch.from_numpy(x))
        eis.append(torch.from_numpy(ei) + off)
        eas.append(torch.from_numpy(ea))
        batch.append(torch.full((x.shape[0],), gi, dtype=torch.long))
        off += x.shape[0]
    return (torch.cat(xs), torch.cat(eis, dim=1), torch.cat(eas),
            torch.cat(batch), len(graphs))


def build_model(hidden: int = 96, layers: int = 3, seed: int = 0):
    import torch
    from torch import nn

    torch.manual_seed(seed)

    class MPNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(NODE_DIM, hidden)
            self.msg = nn.ModuleList(
                [nn.Linear(hidden + EDGE_DIM, hidden) for _ in range(layers)])
            self.upd = nn.ModuleList(
                [nn.Linear(2 * hidden, hidden) for _ in range(layers)])
            self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                      nn.Linear(hidden, 1))

        def forward(self, x, ei, ea, batch, n_graphs):
            h = torch.relu(self.embed(x))
            for msg, upd in zip(self.msg, self.upd):
                if ei.shape[1] > 0:
                    m = torch.relu(msg(torch.cat([h[ei[0]], ea], dim=1)))
                    agg = torch.zeros_like(h).index_add_(0, ei[1], m)
                else:
                    agg = torch.zeros_like(h)
                h = torch.relu(upd(torch.cat([h, agg], dim=1)))
            s = torch.zeros(n_graphs, h.shape[1]).index_add_(0, batch, h)
            cnt = torch.zeros(n_graphs, 1).index_add_(
                0, batch, torch.ones(h.shape[0], 1)).clamp(min=1)
            return self.head(torch.cat([s / cnt, s], dim=1)).squeeze(-1)

    return MPNN()


def fit_predict_gnn(gtr, ytr, gte, seed: int = 0, epochs: int = 40,
                    batch_size: int = 64, lr: float = 1e-3):
    """Fit on (graphs, y) and predict for gte. Targets are standardised inside;
    predictions come back on the original scale so the MAE is comparable with the
    fingerprint families."""
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    y = np.asarray(ytr, dtype=np.float32)
    mu, sd = float(y.mean()), float(y.std()) or 1.0
    yz = torch.from_numpy((y - mu) / sd)

    model = build_model(seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.L1Loss()
    n = len(gtr)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            if len(idx) < 2:
                continue
            x, ei, ea, b, ng = collate([gtr[i] for i in idx])
            opt.zero_grad()
            loss = lossf(model(x, ei, ea, b, ng), yz[idx])
            loss.backward()
            opt.step()

    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(gte), 256):
            chunk = gte[s:s + 256]
            x, ei, ea, b, ng = collate(chunk)
            out.append(model(x, ei, ea, b, ng).numpy())
    return np.concatenate(out) * sd + mu if out else np.zeros(0, dtype=np.float32)
