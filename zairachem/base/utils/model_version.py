"""Resolve the Ersilia Model Hub major version of a locally-pulled model image.

The canonical version is the MAJOR of the released semver tag (``vX.Y.Z``) that the locally-pulled
``ersiliaos/<id>:latest`` image corresponds to (``v1.1.0`` -> ``v1``). An image pulled from DockerHub
is matched to its tag by digest. One Ersilia re-tagged has no digest, so its layers are compared
with each released tag; an image that matches none is taken to be the registry's latest. Build times
are not used: an image is built before its tag is pushed. Defaults to ``v1`` and never raises — any failure (no docker, DockerHub unreachable, parse
error) falls back to ``v1``. :func:`resolved_model_version` returns None in those cases
instead, for callers that have to tell a real ``v1`` from a fallback.

A pinned model (:func:`pin_models`, from the tables in :mod:`zairachem.base.vars` or a fitted
model's ``parameters.json``) is the exception: its version is the pinned major, and it runs one
released tag of that major (:func:`image_tag`) instead of ``:latest``.
"""

import json
import re
import subprocess

import requests

from zairachem.base.generate_config import image_name
from zairachem.base.utils.logging import logger
from zairachem.base.vars import DEFAULT_FEATURIZERS, DEFAULT_PROJECTIONS, MODEL_IMAGES_KEY, ORG

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
# Pinned models of this process, by lower-case id: [major, tag, exact]. The tag is None until it
# is chosen; exact says the fit recorded it, rather than it being chosen from the local images.
_pins = {}


def pin_models(model_ids, params=None):
  """Pin each of ``model_ids`` that has a pin, so it runs a released tag of its pinned major.

  A model whose ``params`` record an image tag (:data:`MODEL_IMAGES_KEY`, written by the run that
  fitted it) runs exactly that tag. Otherwise a model of the pin tables runs the newest local tag of
  its major: the one recorded in ``params`` (``latest_featurizer_version``, a model fitted before
  tags were recorded), else the table's. Every other model runs ``:latest``, and loses any pin an
  earlier call gave it. Ids are case-insensitive.

  Parameters
  ----------
  model_ids : list of str
      Descriptor and projection model IDs of the run.
  params : dict, optional
      The run's (or the fitted model's) ``parameters.json``.
  """
  params = params or {}
  recorded = params.get(MODEL_IMAGES_KEY) or {}
  fitted = params.get("latest_featurizer_version") or {}
  table = {m: v for m, v in {**DEFAULT_FEATURIZERS, **DEFAULT_PROJECTIONS}.items() if v}
  for m in model_ids:
    tag = (recorded.get(m) or {}).get("tag") or ""
    if match := _SEMVER_RE.match(tag):
      _pins[m.lower()] = [f"v{match.group(1)}", tag, True]
    elif m.lower() in table:
      _pins[m.lower()] = [fitted.get(m) or table[m.lower()], None, False]
    else:
      _pins.pop(m.lower(), None)
    for cache in (_cache, _digest_cache, _release_cache):
      cache.pop(m, None)


def pinned_major(model_id):
  """The major ``model_id`` is pinned to (``"vN"``), or None when it is not pinned."""
  pin = _pins.get(model_id.lower())
  return pin[0] if pin else None


def image_tag(model_id):
  """The released tag a pinned model runs, or None (not pinned, or no local tag of its major).

  The tag its fit recorded, else the newest ``vMAJOR.x.y`` of the pinned major present locally.
  Docker Hub is not asked, so a new release there changes nothing until it is pulled. Never raises.
  """
  pin = _pins.get(model_id.lower())
  if not pin:
    return None
  if pin[1] is None:
    pin[1] = _of_major(_local_tags(model_id), pin[0])
  return pin[1]


def tag_latest_as_release(model_id):
  """Tag the local ``:latest`` as the release of the pinned major it is, when that release is newer
  than every local tag of the major; return that tag, or None.

  For a model with no tag recorded by a fit, whose ``:latest`` (pulled before pins) is a release of
  its pinned major: it is matched to Docker Hub's tags by digest, or by layers for an image Ersilia
  re-tagged without one. ``docker tag`` adds the release tag, which the run then uses; nothing is
  downloaded and ``:latest`` is left as it is. None when a recorded tag runs, ``:latest`` is not a
  newer release of the major, Docker Hub cannot be asked, or the tag fails; never raises.
  """
  pin = _pins.get(model_id.lower())
  if not pin or pin[2]:
    return None
  local = image_tag(model_id)
  latest = f"{image_name(model_id)}:latest"
  try:
    release = _release_of(model_id, latest, pin[0])
    if release is None or (local and _key(release) <= _key(local)):
      return None
    out = subprocess.run(
      ["docker", "tag", latest, image_name(f"{model_id}:{release}")],
      capture_output=True,
      text=True,
      timeout=_HTTP_TIMEOUT,
    )
  except Exception as e:
    logger.debug(f"Could not tag {latest} as its release ({e})")
    return None
  if out.returncode != 0:
    logger.warning(f"Could not tag {latest} as {release}: {out.stderr.strip()}")
    return None
  pin[1] = release
  _digest_cache.pop(model_id, None)
  return release


def _release_of(model_id, ref, major):
  # The released tag of ``major`` that the local image ``ref`` is, or None. Docker Hub is asked
  # only for an image that is here, and raises when it cannot be.
  digest = _repo_digest(ref)
  layers, arch = (None, None) if digest else _local_image_layers_and_arch(model_id, ref)
  if not digest and not layers:
    return None
  tags = _dockerhub_tags(f"{ORG}/{model_id.lower()}")
  ours = _newest_first(n for n, _, _ in tags if (n or "").split(".")[0] == major)
  if digest:
    released = [n for n, _, digests in tags if _SEMVER_RE.match(n or "") and digest in digests]
    if released:  # a release by its digest: of this major, or of another (nothing to compare)
      return next((n for n in ours if n in released), None)
    layers, arch = _local_image_layers_and_arch(model_id, ref)  # a digest of this machine's own
  if layers:
    for name in ours:
      if _registry_layers(model_id, arch, name) == layers:
        return name
  return None


def hub_tag(model_id):
  """Docker Hub's newest released tag of ``model_id``'s pinned major, or None if it has none.

  Raises when Docker Hub cannot be asked.
  """
  return _of_major(_released_tags(model_id), pinned_major(model_id))


def pull_hint(model_id):
  """What gets a pinned model's missing image: ``docker pull`` of its tag, or why there is none.

  The tag is the one its fit recorded, else Docker Hub's newest of the pinned major.
  """
  major = pinned_major(model_id)
  tag = image_tag(model_id)
  if tag is None:
    try:
      tag = hub_tag(model_id)
    except Exception:
      return f"Docker Hub is unreachable; pull a {major} tag of {image_name(model_id)}"
  if tag is None:
    return f"Docker Hub has no {major} release of {model_id}"
  return f"docker pull {image_name(f'{model_id}:{tag}')}"


def newer_on_hub(model_id):
  """Docker Hub's newest tag of the pinned major when it is newer than the local one a run chose.

  None when it is not, when the tag was recorded by the fit (that exact tag runs), or when Docker
  Hub cannot be asked; never raises.
  """
  pin = _pins.get(model_id.lower())
  if not pin or pin[2] or not image_tag(model_id):
    return None
  try:
    newest = hub_tag(model_id)
  except Exception:
    return None
  return newest if newest and _key(newest) > _key(pin[1]) else None


def _of_major(tags, major):
  # The first of ``tags`` (newest first) in ``major``, or None.
  return next((t for t in tags if t.split(".")[0] == major), None)


def _key(tag):
  return tuple(int(g) for g in _SEMVER_RE.match(tag).groups())


def image_ref(model_id):
  """The image a model runs: ``ersiliaos/<id>:<tag>`` when pinned, else ``ersiliaos/<id>``.

  None for a pinned model with no tag of its major.
  """
  if not pinned_major(model_id):
    return image_name(model_id)
  tag = image_tag(model_id)
  return image_name(f"{model_id}:{tag}") if tag else None


def server_key(model_id):
  """Key of a model's server in the shared compose project: ``<id>``, or ``<id>:<tag>`` if pinned.

  Each tag of a model gets a server of its own, so runs on different tags do not share one. A
  pinned model with no local tag stops the command (``click.ClickException``) with what to pull.
  """
  if not pinned_major(model_id):
    return model_id
  tag = image_tag(model_id)
  if tag is None:
    import click

    raise click.ClickException(
      f"No local {pinned_major(model_id)} image of {model_id}. {pull_hint(model_id)}, then re-run."
    )
  return f"{model_id.lower()}:{tag}"


def install_images(model_ids):
  """Print ``<id> <tag>`` for each model, for install.sh: the tag it pulls.

  A pinned model gets Docker Hub's newest tag of its major (``?`` when it has none); when Docker
  Hub cannot be asked, the newest local tag of the major, else ``!``. An unpinned one gets ``-``
  and is fetched as ``:latest``. No ids: the default models.
  """
  model_ids = list(model_ids) or [*DEFAULT_FEATURIZERS, *DEFAULT_PROJECTIONS]
  pin_models(model_ids)
  for m in model_ids:
    tag = "-"
    if pinned_major(m):
      try:
        tag = hub_tag(m) or "?"
      except Exception:
        tag = image_tag(m) or "!"
    print(f"{m} {tag}")


def image_records(model_ids):
  """The image each model ran, for ``parameters.json``: ``{id: {"tag": ..., "digest": ...}}``.

  The tag is ``latest`` for an unpinned model.
  """
  return {
    m: {
      "tag": image_tag(m) if pinned_major(m) else "latest",
      "digest": _local_image_digest(m),
    }
    for m in model_ids
  }


def _local_tags(model_id):
  # The released tags of the model present locally, newest version first.
  try:
    out = subprocess.run(
      ["docker", "image", "ls", f"{ORG}/{model_id.lower()}", "--format", "{{.Tag}}"],
      capture_output=True,
      text=True,
      timeout=_HTTP_TIMEOUT,
    )
  except (OSError, subprocess.TimeoutExpired):
    return []
  return _newest_first(out.stdout.split() if out.returncode == 0 else [])


def _newest_first(names):
  return sorted((n for n in names if _SEMVER_RE.match(n or "")), key=_key, reverse=True)


def _local_image_digest(model_id):
  if model_id in _digest_cache:
    return _digest_cache[model_id]
  ref = image_ref(model_id)
  if ref is None:
    return None
  _digest_cache[model_id] = _repo_digest(ref)
  return _digest_cache[model_id]


def _repo_digest(ref):
  # The registry digest of the local image ``ref`` (``sha256:...``), or None.
  digest = None
  try:
    out = subprocess.run(
      [
        "docker",
        "image",
        "inspect",
        ref,
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
  return digest


def _dockerhub_tags(repo):
  # Return a list of (name, last_updated, {digests}) for every tag, following pagination. A failure
  # is kept too, so an unreachable Docker Hub costs one timeout per repo and process.
  if repo in _tags_cache:
    if isinstance(_tags_cache[repo], Exception):
      raise _tags_cache[repo]
    return _tags_cache[repo]
  tags = []
  url = _DOCKERHUB_TAGS_URL.format(repo=repo)
  try:
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
  except Exception as e:
    _tags_cache[repo] = e
    raise
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
  if pinned_major(model_id):
    return pinned_major(model_id)
  if model_id in _cache:
    return _cache[model_id]
  version = _resolve(model_id)
  _cache[model_id] = version
  return version


def registry_unreachable(model_id):
  """Whether DockerHub could not be asked for ``model_id``'s releases (after trying to resolve it).

  An unknown version is otherwise ambiguous: it is also what a model with no released tag gives,
  and that one is a real ``v1``, not an outage. A pinned model's version is never unknown.
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


def _local_image_layers_and_arch(model_id, ref=None):
  # The image's (default: the one the model runs) layer hashes and CPU architecture, or (None, None).
  ref = ref or image_ref(model_id)
  if ref is None:
    return None, None
  try:
    out = subprocess.run(
      [
        "docker",
        "image",
        "inspect",
        ref,
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
  return _newest_first(name for name, _, _ in _dockerhub_tags(f"{ORG}/{model_id.lower()}"))


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
  older release, else None: it was built locally, so it cannot be called behind. A digest that no
  registry tag lists counts as none: Docker's containerd image store gives the image Ersilia
  re-tags a digest of its own. A pulled image reuses the cached digest/tags, so it adds no extra
  network beyond version resolution.
  """
  local = _local_image_digest(model_id)
  if local:
    try:
      tags = _dockerhub_tags(f"{ORG}/{model_id.lower()}")
    except Exception as e:
      logger.debug(f"Could not check if {model_id} is up to date ({e})")
      return None
    if any(local in digests for _name, _last_updated, digests in tags):
      latest_digests = set()
      for name, _last_updated, digests in tags:
        if name == "latest":
          latest_digests |= digests
      return (local in latest_digests) if latest_digests else None
  release = _layers_release(model_id)
  return None if release is None else release == "latest"
