"""Write ``constraints.txt``: exact versions of ZairaChem's dependency closure in this environment.

``install.sh`` passes the file to ``pip install -c`` so a fresh install resolves to the versions the
package was tested with. Regenerate it from a working, up-to-date environment::

    python scripts/make_constraints.py
"""

import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = "zairachem"
EXTRAS = ("isaura",)
OUT = Path(__file__).resolve().parents[1] / "constraints.txt"


def closure():
  """Installed ``{name: version}`` for ZairaChem and everything it needs, extras included."""
  found = {}
  queue = [(ROOT, set(EXTRAS))]
  while queue:
    name, extras = queue.pop()
    key = canonicalize_name(name)
    try:
      dist = distribution(name)
    except PackageNotFoundError:
      print(f"warning: {name} is not installed, skipped", file=sys.stderr)
      continue
    seen = found.setdefault(key, {"version": dist.version, "extras": set()})
    if extras <= seen["extras"] and seen["extras"]:
      continue
    seen["extras"] |= extras
    for line in dist.requires or []:
      req = Requirement(line)
      if req.marker is not None and not any(
        req.marker.evaluate({"extra": e}) for e in (extras or {""})
      ):
        continue
      queue.append((req.name, set(req.extras)))
  return {k: v["version"] for k, v in found.items() if k != canonicalize_name(ROOT)}


def main():
  pins = closure()
  lines = [f"{name}=={version}" for name, version in sorted(pins.items())]
  header = (
    "# Exact versions ZairaChem was tested with; used by install.sh (`pip install -c`).\n"
    "# Regenerate with `python scripts/make_constraints.py` from a working environment.\n"
  )
  OUT.write_text(header + "\n".join(lines) + "\n")
  print(f"wrote {len(lines)} pins to {OUT}")


if __name__ == "__main__":
  main()
