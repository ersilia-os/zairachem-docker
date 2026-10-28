"""Resolve the Ersilia Model Hub major version of a locally-pulled model image.

The canonical version is the MAJOR of the released semver tag (``vX.Y.Z``) that the locally-pulled
``ersiliaos/<id>:latest`` image corresponds to (``v1.1.0`` -> ``v1``). An image pulled from DockerHub
is matched to its tag by digest. One Ersilia re-tagged has no digest, so its layers are compared
with each released tag; an image that matches none is taken to be the registry's latest. Build times
are not used: an image is built before its tag is pushed. Defaults to ``v1`` and never raises — any failure (no docker, DockerHub unreachable, parse
error) falls back to ``v1``. :func:`resolved_model_version` returns None in those cases
instead, for callers that have to tell a real ``v1`` from a fallback.
"""

import json
import re
import subprocess

import requests

from zairachem.base.utils.logging import logger
from zairachem.base.vars import ORG

_DOCKERHUB_TAGS_URL = "https://hub.docker.com/v2/repositories/{repo}/tags?page_size=100"
_SEMVER_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
_DEFAULT_VERSION = "v1"
_HTTP_TIMEOUT = 5
_REGISTRY_AUTH_URL = "https://auth.docker.io/token"
_REGISTRY_URL = "https://registry-1.docker.io/v2/{repo}/{kind}/{ref}"
_MANIFEST_TYPES = ",".join((
  "application/vnd.oci.image.index.v1+json",
  "application/vnd.docker.distribution.manifest.list.v2+json",
  "application/vnd.oci.image.manifest.v1+json",
  "application/vnd.docker.distribution.manifest.v2+json",
))

# Per-process caches so each model resolves once (version, local image digest, DockerHub tags).
_cache = {}
_digest_cache = {}
_tags_cache = {}
_release_cache = {}
_unreachable = set()


def _local_image_digest(model_id):
  if model_id in _digest_cache:
    return _digest_cache[model_id]
  digest = None
  try:
    out = subprocess.run(
      [
        "docker",
        "image",
        "inspect",
        f"{ORG}/{model_id.lower()}",
        "--format",
        "{{index .RepoDigests 0}}",
      ],
      capture_output=True,
      text=True,
      timeout=_HTTP_TIMEOUT,
    )
    if out.returncode == 0 and "@" in out.stdout:
      digest = out.stdout.strip().split("@", 1)[1]  # ersiliaos/eos3l5f@sha256:... -> sha256:...
  except (OSError, subprocess.TimeoutExpired):
    digest = None
  _digest_cache[model_id] = digest
  return digest


def _dockerhub_tags(repo):
  # Return a list of (name, last_updated, {digests}) for every tag, following pagination.
  if repo in _tags_cache:
    return _tags_cache[repo]
  tags = []
  url = _DOCKERHUB_TAGS_URL.format(repo=repo)
  while url:
    resp = requests.get(url, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    for r in data.get("results", []):
      digests = set()
      if r.get("digest"):
        digests.add(r["digest"])
      for im in r.get("images") or []:
        if im.get("digest"):
          digests.add(im["digest"])
      tags.append((r.get("name"), r.get("last_updated"), digests))
    url = data.get("next")
  _tags_cache[repo] = tags
  return tags


def ersilia_model_version(model_id):
  """Return the model's major version as ``"v{N}"`` (default ``"v1"``); never raises."""
  return resolved_model_version(model_id) or _DEFAULT_VERSION


def resolved_model_version(model_id):
  """The model's major version, or None when it could not be determined; never raises.

  Apart from :func:`ersilia_model_version` only in what it does when the answer is unknown. That
  function defaults to ``"v1"``, which is right for choosing a transformer — a run has to pick
  something — but wrong for comparing two versions, where the default cannot be told from a
  genuine ``v1`` and would report a drift that may not have happened.
  """
  if model_id in _cache:
    return _cache[model_id]
  version = _resolve(model_id)
  _cache[model_id] = version
  return version


def registry_unreachable(model_id):
  """Whether DockerHub could not be asked for ``model_id``'s releases (after trying to resolve it).

  An unknown version is otherwise ambiguous: it is also what a model with no released tag gives,
  and that one is a real ``v1``, not an outage.
  """
  resolved_model_version(model_id)
  return model_id in _unreachable


def _resolve(model_id):
  try:
    tags = _dockerhub_tags(f"{ORG}/{model_id.lower()}")
  except Exception as e:
    # Broad on purpose: ersilia_model_version() must never raise. Log so a silent default isn't
    # mistaken for a real version when DockerHub is unreachable.
    logger.debug(f"Could not resolve version for {model_id} ({e})")
    _unreachable.add(model_id)
    return None

  semver = [
    (last_updated, int(m.group(1)), name)
    for name, last_updated, _ in tags
    if (m := _SEMVER_RE.match(name or "")) and last_updated
  ]
  if not semver:
    return None
  # Pulled from DockerHub: the tags sharing the local image's digest say when it was pushed, and
  # the release is the newest tag pushed at or before that.
  digest = _local_image_digest(model_id)
  matched = [lu for (_, lu, digs) in tags if digest and lu and digest in digs]
  if matched:
    pushed = max(matched)
    candidates = [(lu, major) for lu, major, _ in semver if lu <= pushed]
    return f"v{max(candidates)[1]}" if candidates else None
  # No digest to match (Ersilia re-tags a pulled image): the layers name the release if it is an
  # older one; an image matching the latest, or none, is taken to be the latest.
  release = _layers_release(model_id)
  if release and release != "latest":
    return f"v{int(_SEMVER_RE.match(release).group(1))}"
  return f"v{max(semver)[1]}"


def _local_image_layers_and_arch(model_id):
  # The image's layer hashes and CPU architecture, or (None, None).
  try:
    out = subprocess.run(
      [
        "docker",
        "image",
        "inspect",
        f"{ORG}/{model_id.lower()}",
        "--format",
        "{{.Architecture}} {{json .RootFS.Layers}}",
      ],
      capture_output=True,
      text=True,
      timeout=_HTTP_TIMEOUT,
    )
    if out.returncode == 0 and out.stdout.strip():
      arch, _, layers = out.stdout.strip().partition(" ")
      return json.loads(layers), arch
  except (OSError, subprocess.TimeoutExpired, ValueError):
    pass
  return None, None


def _registry_layers(model_id, arch, tag):
  # Layer hashes of the registry's ``tag`` image for ``arch``, from its config blob.
  repo = f"{ORG}/{model_id.lower()}"
  token = requests.get(
    _REGISTRY_AUTH_URL,
    params={"service": "registry.docker.io", "scope": f"repository:{repo}:pull"},
    timeout=_HTTP_TIMEOUT,
  )
  token.raise_for_status()
  headers = {"Authorization": f"Bearer {token.json()['token']}", "Accept": _MANIFEST_TYPES}

  def fetch(kind, ref):
    resp = requests.get(
      _REGISTRY_URL.format(repo=repo, kind=kind, ref=ref), headers=headers, timeout=_HTTP_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()

  manifest = fetch("manifests", tag)
  for entry in manifest.get("manifests") or []:  # a multi-platform index: take this architecture
    platform = entry.get("platform") or {}
    if platform.get("os") == "linux" and platform.get("architecture") == arch:
      manifest = fetch("manifests", entry["digest"])
      break
  return fetch("blobs", manifest["config"]["digest"]).get("rootfs", {}).get("diff_ids")


def _released_tags(model_id):
  # The released semver tag names, newest version first.
  found = []
  for name, _last_updated, _ in _dockerhub_tags(f"{ORG}/{model_id.lower()}"):
    m = _SEMVER_RE.match(name or "")
    if m:
      found.append((tuple(int(g) for g in m.groups()), name))
  return [name for _, name in sorted(found, reverse=True)]


def _layers_release(model_id):
  """The registry tag the local image's layers match: ``"latest"``, a released tag, or None.

  For an image with no registry digest (``RepoDigests`` is empty). Ersilia re-tags a pulled image
  with a label of its own (``ersilia.user.id``), which changes the image's config and so its ID but
  not its layers, so the layers are what is compared. ``latest`` is tried first, then the released
  tags newest first. None when none matches (the image was built locally) or the registry cannot
  be asked; never raises.
  """
  if model_id in _release_cache:
    return _release_cache[model_id]
  release = None
  layers, arch = _local_image_layers_and_arch(model_id)
  if layers:
    try:
      for tag in ["latest", *_released_tags(model_id)]:
        if _registry_layers(model_id, arch, tag) == layers:
          release = tag
          break
    except Exception as e:
      logger.debug(f"Could not compare {model_id} with the registry's releases ({e})")
  _release_cache[model_id] = release
  return release


def is_image_up_to_date(model_id):
  """Whether the local ``:latest`` image matches the registry's current ``:latest``.

  Returns True if the locally-pulled image's digest equals the digest the ``latest`` tag points to
  on DockerHub right now, False if it's behind (a newer image has been pushed), or None if it can't
  be determined (no local image, DockerHub unreachable, or no ``latest`` tag). An image without a
  registry digest is True if its layers match the registry's ``latest``, False if they match an
  older release, else None: it was built locally, so it cannot be called behind. A pulled image
  reuses the cached digest/tags, so it adds no extra network beyond version resolution.
  """
  local = _local_image_digest(model_id)
  if not local:
    release = _layers_release(model_id)
    return None if release is None else release == "latest"
  try:
    tags = _dockerhub_tags(f"{ORG}/{model_id.lower()}")
  except Exception as e:
    logger.debug(f"Could not check if {model_id} is up to date ({e})")
    return None
  latest_digests = set()
  for name, _last_updated, digests in tags:
    if name == "latest":
      latest_digests |= digests
  if not latest_digests:
    return None
  return local in latest_digests
