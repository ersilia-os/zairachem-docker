"""Golden-output regression test: re-run the modelling steps on a fixed fit and compare outputs.

Opt-in, because it needs a trained model kept with its descriptor matrices (about 25 MB, not in
git). Create one once, then record its reference outputs with the code you trust::

    zairachem fit -i example/data_train.csv -m <fixture> --keep-intermediate-data
    ZAIRACHEM_GOLDEN_MODEL=<fixture> ZAIRACHEM_GOLDEN_RECORD=1 pytest tests/test_golden.py

Every later run (``ZAIRACHEM_GOLDEN_MODEL=<fixture> pytest tests/test_golden.py``) copies the
fixture, re-runs estimate -> pool -> report -> finish through the CLI and compares against that
reference. The reference lives next to the fixture as ``<fixture>.reference/``.
"""

import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

FIXTURE = os.environ.get("ZAIRACHEM_GOLDEN_MODEL")
RECORD = os.environ.get("ZAIRACHEM_GOLDEN_RECORD") == "1"
STEPS = ["estimate", "pool", "report", "finish"]

# Two runs on the same fixture agree bit for bit everywhere except the performance-table columns
# that depend on the decision cutoff: GHOST picks it from random subsets of the training rows
# without a seed, so it can land on a neighbouring grid value. Those columns are not compared;
# AUROC, AUPR and precision-at-k are.
CUTOFF_DEPENDENT = {
  "cutoff",
  "tp",
  "tn",
  "fp",
  "fn",
  "accuracy",
  "balanced_accuracy",
  "precision",
  "recall",
  "f1_score",
  "mcc",
}

pytestmark = pytest.mark.skipif(
  not FIXTURE, reason="set ZAIRACHEM_GOLDEN_MODEL to a fit made with --keep-intermediate-data"
)


def _copy_fixture(dest):
  """Copy the fixture to *dest* and point its session at the copy."""
  shutil.copytree(FIXTURE, dest, symlinks=True)
  session_path = os.path.join(dest, "session.json")
  with open(session_path) as f:
    session = json.load(f)
  session["output_dir"] = session["model_dir"] = str(dest)
  with open(session_path, "w") as f:
    json.dump(session, f, indent=4)


def _run_steps(model_dir):
  """Run each modelling step through the ``zairachem`` entry point, as a user would."""
  for step in STEPS:
    args = [step, "-m", str(model_dir)]
    if step == "finish":
      args.append("--keep-intermediate-data")
    code = (
      "import sys; from zairachem.cli import main; sys.argv = ['zairachem'] + sys.argv[1:]; main()"
    )
    subprocess.run([sys.executable, "-c", code, *args], check=True, capture_output=True)


def _collect(model_dir):
  """Gather the outputs the refactor must not change, as plain JSON-able data."""
  tables = {
    "output": "results/output.csv",
    "performance": "results/performance_table.csv",
    "pooled": "model/pool/results_unmapped.csv",
    "pool_inputs": "model/pool/data.csv",
  }
  out = {}
  for name, rel in tables.items():
    df = pd.read_csv(os.path.join(model_dir, rel))
    out[name] = {"columns": list(df.columns), "data": df.to_dict(orient="list")}
  report_dir = os.path.join(model_dir, "report")
  out["report_files"] = sorted(
    os.path.relpath(os.path.join(root, f), report_dir)
    for root, _dirs, files in os.walk(report_dir)
    for f in files
    if f.endswith((".png", ".pdf", ".svg", ".html"))
  )
  html = [f for f in out["report_files"] if f.endswith(".html")]
  ids = set()
  for f in html:
    with open(os.path.join(report_dir, f)) as fh:
      ids.update(re.findall(r'id="([^"]+)"', fh.read()))
  out["html_ids"] = sorted(ids)
  return out


def _assert_same(got, ref):
  """Everything must match exactly, except the cutoff-dependent performance columns."""
  assert got["report_files"] == ref["report_files"]
  assert got["html_ids"] == ref["html_ids"]
  for name in ("output", "performance", "pooled", "pool_inputs"):
    assert got[name]["columns"] == ref[name]["columns"], name
    for col in ref[name]["columns"]:
      if name == "performance" and col in CUTOFF_DEPENDENT:
        continue
      a, b = got[name]["data"][col], ref[name]["data"][col]
      try:
        np.testing.assert_array_equal(np.asarray(a, float), np.asarray(b, float), err_msg=col)
      except (TypeError, ValueError):
        assert a == b, f"{name}.{col}"


def test_steps_reproduce_the_reference(tmp_path):
  model_dir = tmp_path / "model"
  _copy_fixture(model_dir)
  _run_steps(model_dir)
  got = _collect(model_dir)
  ref_path = FIXTURE.rstrip("/") + ".reference/outputs.json"
  if RECORD:
    os.makedirs(os.path.dirname(ref_path), exist_ok=True)
    with open(ref_path, "w") as f:
      json.dump(got, f)
    pytest.skip(f"recorded the reference at {ref_path}")
  with open(ref_path) as f:
    ref = json.load(f)
  _assert_same(got, ref)
