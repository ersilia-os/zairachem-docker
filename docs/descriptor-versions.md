# Descriptor versions

ZairaChem scales each descriptor with a transformer fitted for its major version, ranks predictions
against a reference matrix computed with that major, and keys its stored results by it. A new major
of a descriptor needs all three before a run can use it, and they are published some time after the
image. So each default descriptor is pinned to a major, in `DEFAULT_FEATURIZERS`
(`zairachem/base/vars.py`); `DEFAULT_PROJECTIONS` takes pins the same way.

## What a pin does

- **Version.** The pinned major (`v2`) is the model's version for the transformer, the isaura store
  and the rank reference, whatever image is on the machine.
- **Image.** A fit runs the newest released tag of the pinned major present locally (`v2.0.1`, as
  `ersiliaos/<id>:v2.0.1`), with or without Docker Hub. ZairaChem never pulls. When no tag of the
  major is local, setup stops with the `docker pull ersiliaos/<id>:<tag>` of Docker Hub's newest one
  (or, when Docker Hub cannot be reached, says so and names the major). That command leaves
  `:latest` alone, and so does every run. `install.sh` pulls Docker Hub's newest tag of each pinned
  major.
- **The local `:latest`.** An image pulled before pins counts too. When the local `:latest` is a
  release of the pinned major (its digest matches one of Docker Hub's tags, or, for an image Ersilia
  re-tagged without a digest, its layers do) and that release is newer than every local tag of the
  major, setup gives it that tag (`docker tag ersiliaos/<id>:latest ersiliaos/<id>:v2.1.1`), says
  so, and runs it. Nothing is downloaded and `:latest` is unchanged. An equal or newer local tag
  runs as it is. When Docker Hub cannot be
  reached, `:latest` cannot be identified: the newest local tag runs, and with none setup stops as
  above; so does a failed tag.
- **What Docker Hub does.** Nothing to a run. A newer tag of the same major gives one info line at
  setup with its `docker pull`; the next fit or predict runs it once it is pulled. A newer
  `:latest`, or a newer major, does not stop a run either.
- **Servers.** Each tag runs on a model server of its own (`eos3l5f_v2_0_1_api`), so runs on
  different tags of one model do not share one.
- **Predict.** Only a major release changes a model's output, so predict, a resumed fit and the
  step commands run the newest local tag of the major the model was fitted with
  (`latest_featurizer_version`), not necessarily the tag the fit ran. With no local tag of that
  major, setup stops with the `docker pull` of Docker Hub's newest one.
- **Record.** Each run writes the tag and image digest each model ran to `parameters.json`
  (`model_images`), for provenance. They do not choose the image of a later run.

A model with no pin, such as a descriptor given with `-f` or a default whose pin is `None`, runs
`ersiliaos/<id>:latest` as before: its major is read from the image, and a local `:latest` behind
Docker Hub's stops the run with an `ersilia fetch` command.

All tags of a major share one transformer, one rank reference and one store key.

## Moving a pin

A pin moves to a new major only in a pull request made after these are published for that major:

1. the eosframes scaler (`<id>_<vN>_transformer.json` in the reference library);
2. the rank-reference matrix.

`tests/test_pins.py` checks that the scaler of every pinned major is published (it needs the
network, and is skipped offline).
