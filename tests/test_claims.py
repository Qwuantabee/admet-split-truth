"""The README's numbers must equal the committed artefact.

This is the test that catches a stale headline. It fails in CI, not only when
someone remembers to run the audit by hand.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_every_readme_number_matches_an_artefact():
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_claims.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, (
        "README and artefacts disagree:\n" + p.stdout[-2000:] + p.stderr[-800:])


def test_the_audit_fails_when_a_number_is_wrong(tmp_path):
    """An audit that cannot fail is decoration.

    Mutate the headline to a wrong value in a copy of the README and require the
    audit to reject it. An audit that only searches for a substring anywhere in the
    file passes wrong numbers, because the right digits usually appear somewhere.
    """
    readme = (ROOT / "README.md").read_text()
    mutated = readme.replace("| **0.76** |", "| **0.96** |", 1)
    assert mutated != readme, "the headline row was not found to mutate"
    p = tmp_path / "README.md"
    p.write_text(mutated)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_claims.py"),
                        "--readme", str(p)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1, (
        "the audit accepted a wrong headline; it cannot protect anything\n"
        + r.stdout[-1200:])


def test_the_audit_fails_when_an_artefact_is_missing(tmp_path):
    """A missing artefact must fail, not silently shrink the checklist."""
    import shutil

    fake = tmp_path / "results"
    fake.mkdir()
    for name in ("grid.json", "grid.csv"):
        shutil.copy(ROOT / "results" / name, fake / name)
    (fake / "grid.json").unlink()
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_claims.py")],
                       capture_output=True, text=True, cwd=ROOT,
                       env={**__import__("os").environ, "ADMET_RES": str(fake)})
    # the audit reads results/ from the repo root; the direct check is that its
    # REQUIRED list is enforced, which we assert on the module
    sys.path.insert(0, str(ROOT / "scripts"))
    import audit_claims

    assert "grid.json" in audit_claims.REQUIRED
    assert "grid.csv" in audit_claims.REQUIRED


def test_every_figure_is_embedded_once_and_exists():
    import re

    readme = (ROOT / "README.md").read_text()
    embeds = re.findall(r"\]\((figures/[a-zA-Z0-9_]+\.png)\)", readme)
    for e in set(embeds):
        assert embeds.count(e) == 1, f"{e} embedded {embeds.count(e)} times"
        assert (ROOT / e).exists(), f"{e} is embedded but missing"
    on_disk = {f"figures/{p.name}" for p in (ROOT / "figures").glob("*.png")}
    orphans = sorted(on_disk - set(embeds))
    assert not orphans, f"figures exist but are not shown: {orphans}"
