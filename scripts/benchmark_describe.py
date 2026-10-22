"""Benchmark ZairaChem's describe step against the Ersilia CLI on the same molecules.

ZairaChem and ``ersilia run`` call the same model image on the same endpoint, so ZairaChem should
never be slower. Two rules are checked:

1. Per model, run alone: ZairaChem's describe takes no longer than ``ersilia serve`` + ``ersilia
   run`` + ``ersilia close``.
2. All models together: ZairaChem's describe takes no longer than the sum of the Ersilia times run
   one after another.

Both sides are timed end to end, server start-up included, with Ersilia's result cache off. Models
not fetched in the Ersilia CLI are reported and skipped there. Run it with the Docker VM otherwise
idle (no other ZairaChem run)::

    python scripts/benchmark_describe.py --out <scratch-dir>
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO / "example" / "data_train.csv"
DEFAULT_ERSILIA = Path.home() / "miniconda3" / "envs" / "ersilia" / "bin" / "ersilia"


def _run(cmd, log):
  """Run ``cmd``, appending its output to ``log``; return (seconds, returncode)."""
  t = time.perf_counter()
  with open(log, "a") as f:
    f.write(f"\n$ {' '.join(map(str, cmd))}\n")
    f.flush()
    code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
  return time.perf_counter() - t, code


def _smiles(path):
  with open(path) as f:
    rows = list(csv.DictReader(f))
  col = next(c for c in rows[0] if c.lower() in ("smiles", "input"))
  return [r[col] for r in rows]


def _ersilia_failed_rows(output_csv):
  """Rows of an ``ersilia run`` output whose values are all empty."""
  with open(output_csv) as f:
    rows = list(csv.reader(f))
  header, body = rows[0], rows[1:]
  first = next((i for i, h in enumerate(header) if h not in ("key", "input")), len(header))
  return sum(all(v.strip() == "" for v in r[first:]) for r in body)


def _zairachem_failed_rows(model_dir, model_id):
  """Rows of ZairaChem's raw descriptors that are all NaN."""
  base = Path(model_dir) / "descriptors" / model_id
  files = sorted((base / "raw_chunks").glob("*.h5")) or [base / "raw.h5"]
  n = 0
  for path in files:
    if path.exists():
      with h5py.File(path) as f:
        n += int(np.isnan(f["Values"][:]).all(axis=1).sum())
  return n


def ersilia(model_id, input_csv, out, exe):
  """Serve, run and close one model with the Ersilia CLI; return its timings, or a skip reason."""
  if not (Path.home() / "eos" / "repository" / model_id).exists():
    return {"skipped": f"not fetched (ersilia fetch {model_id} --from_dockerhub)"}
  log = out / f"ersilia_{model_id}.log"
  output = out / f"ersilia_{model_id}.csv"
  serve_s, code = _run([exe, "serve", model_id, "--disable_cache"], log)
  if code:
    return {"skipped": f"serve failed (see {log})"}
  run_s, code = _run([exe, "run", "-i", input_csv, "-o", output], log)
  close_s, _ = _run([exe, "close"], log)
  if code or not output.exists():
    return {"seconds": serve_s + run_s + close_s, "failed": "run failed", "log": str(log)}
  return {
    "seconds": round(serve_s + run_s + close_s, 1),
    "run_seconds": round(run_s, 1),
    "failed_rows": _ersilia_failed_rows(output),
  }


def _copy_run(src, dest, featurizer_ids):
  """Copy a set-up run to ``dest`` with only ``featurizer_ids`` to describe and no projections."""
  shutil.copytree(src, dest, symlinks=True)
  session = dest / "session.json"
  data = json.loads(session.read_text())
  data["output_dir"] = data["model_dir"] = str(dest)
  session.write_text(json.dumps(data, indent=4))
  params = dest / "metadata" / "parameters.json"
  p = json.loads(params.read_text())
  p["featurizer_ids"], p["projection_ids"] = list(featurizer_ids), []
  params.write_text(json.dumps(p, indent=2))


def zairachem_describe(setup_dir, dest, model_ids, exe):
  """Run ZairaChem's describe step for ``model_ids`` on a copy of the set-up run.

  The shared model servers are stopped first (not timed), so start-up is measured as on Ersilia's
  side: ZairaChem otherwise keeps them up between runs.
  """
  _copy_run(setup_dir, dest, model_ids)
  _run([exe, "stop"], dest.with_suffix(".stop.log"))
  seconds, code = _run([exe, "describe", "-m", dest], dest.with_suffix(".log"))
  if code:
    return {"seconds": round(seconds, 1), "failed": f"describe failed (see {dest}.log)"}
  done = json.loads((dest / "descriptors" / "done_eos.json").read_text())
  return {
    "seconds": round(seconds, 1),
    "dropped": [m for m in model_ids if m not in done],
    "failed_rows": {m: _zairachem_failed_rows(dest, m) for m in done},
  }


def main():
  """Run the benchmark and print the comparison table."""
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--input", default=str(DEFAULT_INPUT), help="CSV with a smiles column")
  ap.add_argument("--models", help="comma-separated model ids (default: ZairaChem's defaults)")
  ap.add_argument("--out", required=True, help="scratch folder for runs, logs and results")
  ap.add_argument("--ersilia", default=str(DEFAULT_ERSILIA), help="path of the ersilia CLI")
  ap.add_argument(
    "--zairachem",
    default=str(Path(sys.executable).with_name("zairachem")),
    help="path of the zairachem CLI (default: the one next to this Python)",
  )
  args = ap.parse_args()

  from zairachem.base.vars import DEFAULT_FEATURIZERS

  models = args.models.split(",") if args.models else list(DEFAULT_FEATURIZERS)
  out = Path(args.out).resolve()
  out.mkdir(parents=True, exist_ok=True)
  smiles_csv = out / "smiles.csv"
  with open(smiles_csv, "w") as f:
    f.write("smiles\n" + "\n".join(_smiles(args.input)) + "\n")

  setup_dir = out / "zc_setup"
  if not setup_dir.exists():
    _, code = _run(
      [args.zairachem, "setup", "-i", args.input, "-m", setup_dir, "-p", "none"],
      out / "zc_setup.log",
    )
    if code:
      sys.exit(f"zairachem setup failed; see {out / 'zc_setup.log'}")

  results = {"n_molecules": len(_smiles(args.input)), "models": {}}
  for m in models:
    print(f"[{m}] ersilia …", flush=True)
    e = ersilia(m, smiles_csv, out, args.ersilia)
    print(f"[{m}] zairachem …", flush=True)
    z = zairachem_describe(setup_dir, out / f"zc_{m}", [m], args.zairachem)
    results["models"][m] = {"ersilia": e, "zairachem": z}
  print("[all] zairachem, every model together …", flush=True)
  results["together"] = zairachem_describe(setup_dir, out / "zc_all", models, args.zairachem)
  (out / "results.json").write_text(json.dumps(results, indent=2))

  print(f"\n{results['n_molecules']} molecules · seconds, server start-up included\n")
  print("| Model | Ersilia | ZairaChem | Rule 1 | Failed rows (E / Z) |")
  print("| --- | --- | --- | --- | --- |")
  serial = 0.0
  for m, r in results["models"].items():
    e, z = r["ersilia"], r["zairachem"]
    es = e.get("seconds")
    zs = z.get("seconds")
    if es is None:
      verdict = e["skipped"]
    elif "failed" in e:
      verdict = "Ersilia failed"
    else:
      serial += es
      verdict = "ok" if zs is not None and zs <= es else "SLOWER"
    zf = z.get("failed_rows", {}).get(m, "dropped" if m in z.get("dropped", []) else "–")
    print(
      f"| {m} | {es if es is not None else '–'} | {zs} | {verdict} | {e.get('failed_rows', '–')} / {zf} |"
    )
  together = results["together"].get("seconds")
  rule2 = "ok" if together is not None and together <= serial else "SLOWER"
  print(
    f"\nAll together: ZairaChem {together} s vs {serial:.1f} s for the Ersilia runs in series "
    f"(models Ersilia ran) → rule 2 {rule2}. Results: {out / 'results.json'}"
  )


if __name__ == "__main__":
  main()
