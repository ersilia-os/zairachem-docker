"""Resolve the Ersilia Model Hub major version of a locally-pulled model image.

The canonical version is the MAJOR of the released semver tag (``vX.Y.Z``) that the locally-pulled
``ersiliaos/<id>:latest`` image corresponds to, found chronologically: place the local image among
the DockerHub tags by build time, then take the most recent ``vN`` at or before it (``v1.1.0`` ->
``v1``). Defaults to ``v1`` and never raises — any failure (no docker, DockerHub unreachable, parse
error) falls back to ``v1``. :func:`resolved_model_version` returns None in those cases
instead, for callers that have to tell a real ``v1`` from a fallback.
"""

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
_REGISTRY_URL = "https://registry-1.docker.io/v2/{repo}/manifests/{ref}"
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


def _local_image_created(model_id):
  try:
    out = subprocess.run(
      ["docker", "image", "inspect", f"{ORG}/{model_id.lower()}", "--format", "{{.Created}}"],
      capture_output=True,
      text=True,
      timeout=_HTTP_TIMEOUT,
    )
    return out.stdout.strip() or None if out.returncode == 0 else None
  except (OSError, subprocess.TimeoutExpired):
    return None


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


def _resolve(model_id):
  try:
    tags = _dockerhub_tags(f"{ORG}/{model_id.lower()}")
  except Exception as e:
    # Broad on purpose: ersilia_model_version() must never raise. Log so a silent default isn't
    # mistaken for a real version when DockerHub is unreachable.
    logger.debug(f"Could not resolve version for {model_id} ({e})")
    return None

  # When was the locally-pulled image pushed? Use the tags sharing its digest; else the image's
  # created date; else (no docker) the most recent moment, i.e. consider all released tags.
  digest = _local_image_digest(model_id)
  local_time = None
  if digest:
    matched = [lu for (_, lu, digs) in tags if lu and digest in digs]
    local_time = max(matched) if matched else None
  if local_time is None:
    local_time = _local_image_created(model_id)

  # Released semver tags at or before the local build time (ISO-8601 strings sort chronologically).
  candidates = []
  for name, last_updated, _ in tags:
    m = _SEMVER_RE.match(name or "")
    if not m or not last_updated:
      continue
    if local_time is None or last_updated <= local_time:
      candidates.append((last_updated, int(m.group(1))))
  if not candidates:
    return None
  return f"v{max(candidates)[1]}"


def _local_image_id_and_arch(model_id):
  # The image's ID (the hash of its config) and CPU architecture, or (None, None).
  try:
    out = subprocess.run(
      [
        "docker",
        "image",
        "inspect",
        f"{ORG}/{model_id.lower()}",
        "--format",
        "{{.Id}} {{.Architecture}}",
      ],
      capture_output=True,
      text=True,
      timeout=_HTTP_TIMEOUT,
    )
    if out.returncode == 0 and out.stdout.strip():
      image_id, _, arch = out.stdout.strip().partition(" ")
      return image_id, arch
  except (OSError, subprocess.TimeoutExpired):
    pass
  return None, None


def _registry_latest_config(model_id, arch):
  # Config digest of the registry's ``latest`` image for ``arch``: the value a local image's ID
  # equals when it is that image.
  repo = f"{ORG}/{model_id.lower()}"
  token = requests.get(
    _REGISTRY_AUTH_URL,
    params={"service": "registry.docker.io", "scope": f"repository:{repo}:pull"},
    timeout=_HTTP_TIMEOUT,
  )
  token.raise_for_status()
  headers = {"Authorization": f"Bearer {token.json()['token']}", "Accept": _MANIFEST_TYPES}

  def manifest(ref):
    resp = requests.get(
      _REGISTRY_URL.format(repo=repo, ref=ref), headers=headers, timeout=_HTTP_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()

  top = manifest("latest")
  for entry in top.get("manifests") or []:
    platform = entry.get("platform") or {}
    if platform.get("os") == "linux" and platform.get("architecture") == arch:
      return manifest(entry["digest"]).get("config", {}).get("digest")
  return (top.get("config") or {}).get("digest")  # a single-platform image


def _matches_registry_latest(model_id):
  """Whether the local image is the registry's current ``latest``, judged by its config digest.

  For an image with no registry digest (retagged or built locally, so ``RepoDigests`` is empty): a
  local image's ID is the hash of its config, which the registry's manifest for the same
  architecture also lists. False when they differ or the registry cannot be asked; never raises.
  """
  image_id, arch = _local_image_id_and_arch(model_id)
  if not image_id:
    return False
  try:
    return _registry_latest_config(model_id, arch) == image_id
  except Exception as e:
    logger.debug(f"Could not compare {model_id} with the registry's latest ({e})")
    return False


def is_image_up_to_date(model_id):
  """Whether the local ``:latest`` image matches the registry's current ``:latest``.

  Returns True if the locally-pulled image's digest equals the digest the ``latest`` tag points to
  on DockerHub right now, False if it's behind (a newer image has been pushed), or None if it can't
  be determined (no local image, DockerHub unreachable, or no ``latest`` tag). An image without a
  registry digest is True only if its config matches the registry's ``latest``, else None: it was
  built or retagged locally, so it cannot be called behind. A pulled image reuses the cached
  digest/tags, so it adds no extra network beyond version resolution.
  """
  local = _local_image_digest(model_id)
  if not local:
    return True if _matches_registry_latest(model_id) else None
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
