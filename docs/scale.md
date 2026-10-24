# Running on large datasets

What a run costs, where it stops, and which knobs move it. The figures below were measured on
this pipeline, on a 10-core laptop with a 17 GB Docker VM; treat them as shape rather than as
constants.

## Where the time goes

Two steps account for nearly all of it, at every size:

| step | 700 compounds | 2,000 compounds |
| --- | --- | --- |
| setup | 11s | 20s |
| describe | 60s | 164s |
| treat | 1s | 1s |
| **estimate** | **482s** | **2,707s** |
| **evaluate** (per fold) | **524s** | **1,603s** |
| report | 6s | 7s |

Estimate is lazy-qsar fitting one model per descriptor, and `--evaluate` refits the whole
pipeline once per fold, so a menu of ten folds costs ten fits. That is the price of an honest
held-out score and there is no shortcut to it; `--repeats` and a shorter `--evaluate` list are
the fidelity-preserving ways to spend less.

Within estimate, the cost is rarely spread evenly. On the T. brucei set one descriptor took
2,564s of the 2,707s — 95% — for 16% of the pooled weight. The closing panel names that
descriptor when it happens, and the pre-screening table shows what each one *added* to the pool
rather than only its own score, so the next run can drop it. Those two numbers are the fastest
way to make a fit faster.

## What has to fit in memory

The training matrix does: lazy-qsar is handed `X_train`, which is `rows × features × 4` bytes.
A million compounds against a 1,600-feature descriptor is 5 GB, and that is a floor, not an
estimate. The rest of the pipeline streams — descriptors are written and read in chunks, the
estimate step's prediction passes are chunked, and the report caps every plot at 10,000 points.

Two steps used to hold far more than they needed and no longer do: the training matrix is filled
in place rather than concatenated from per-chunk pieces, which halves its peak, and
pre-screening scores a bounded sample instead of loading each descriptor's whole matrix.

## What gets written

Each descriptor keeps two matrices, raw and scaled, each `rows × features × 4` bytes, and both
stay until the run finishes. Five descriptors over a million compounds at 2,048 features is about
80 GB. Before scaling them, the run compares what it is about to write against the free space and
says so if it will not fit — a warning, early, rather than whatever error the filesystem raises
when it fills.

## Predicting

A prediction is almost entirely descriptor inference — 82-88% of it across every run measured,
with the estimate, pooling and report steps together under ten seconds. So its cost is the
descriptors it was fitted with, and describe runs them side by side: the time is the slowest of
them, not their sum.

Which makes the projection model worth knowing about. It embeds the molecules for the report's
2-D map and for the coordinate columns of `results/output_table.csv`, and on every prediction
measured it was the slowest model of the lot — 45.4s of a 47.5s describe on 500 molecules. `-p
none` skips it and leaves the built-in MW-vs-LogP map, which on that run took the whole
prediction from 58s to 42s with the predictions themselves byte-identical. What it costs is those
coordinate columns; what it saves is whatever the projection was adding over the slowest real
descriptor, so it is largest when the descriptors are fast.

`--skip-report` does **not** imply this. The coordinates are a deliverable of their own, so
skipping the report does not decide that you do not want them; pass both if that is what you mean.

## Where it stops

- **Butina splits** compare every pair of molecules, so their time and memory grow with the
  square of the set: 4,000 molecules cost 5s and 61 MB, and 50,000 would need about 9 GB before
  clustering starts. Above 10,000 the schema is skipped with a message. The scaffold schemas ask
  the same question of the chemistry at linear cost and have no such limit.
- **Pre-screening** scores at most 25,000 rows, sampled to keep the target's shape. Its job is to
  rank descriptors against one another, and that ranking settles long before a large set is
  exhausted. Below 25,000 rows every row is used and the selection is exactly what it always was.
- **Descriptor inference** is the Ersilia model servers' own throughput and the Docker VM's
  memory. The describe step measures each model on this machine and schedules them to fit.

## Does the order of the input rows matter?

Not in any way that reaches a decision, but it is not bit-for-bit either, and it is worth knowing
which of those two sentences applies where.

Measured on a trained regression model asked to predict its own 700 training molecules, once in
the file's order and once shuffled, then merged back on SMILES: every molecule kept its own label
and its own prediction, and 698 of the 700 predictions were identical to the last digit written.
Two moved, by at most 0.0015 — against that model's own RMSE of 0.075, and an assay noise floor
an order of magnitude above that again. The same file predicted twice is byte-identical, so what
little moves is the order and not the run.

It enters in two places, neither of them ZairaChem's:

- **The descriptor servers.** For 8 of those 700 molecules, `eos4u6p` returned an embedding that
  differed by up to 4.5e-08 — one float32 ulp — depending on where the molecule sat in the request
  batch. Those 8 are the rows whose per-descriptor prediction moved most, up to 0.0039.
- **The batched predict underneath.** Hand lazy-qsar the *same* matrix twice, once permuted, and
  245 of 700 predictions move by up to 6.9e-06; predict a row on its own and it differs from the
  same row inside the batch. That is floating-point reassociation in the linear algebra, the usual
  consequence of a library choosing its blocking by the shape of what it is given.

So a prediction is reproducible to about five decimal places across orderings, and exactly
reproducible for a given ordering. Anything that claims to depend on more digits than that is
reading noise. If you need two runs to agree exactly, feed them the rows in the same order.

What this is *not* is a shuffle. Labels, SMILES and predictions travel together: shifting the
predictions by a single row against their molecules takes that model's R² from 0.994 to -0.243,
which is the size of the signal a real misalignment would destroy.

## Knobs

| flag | effect |
| --- | --- |
| `-p none` (predict) | skip the projection model; drops its output_table coordinate columns |
| `--max-descriptors N` | train only the N descriptors that add most to the pool |
| `--evaluate <schemas>` | which held-out schemas to run; each costs a full refit per fold |
| `--repeats N` | folds per schema |
| `-f <ids>` | the descriptors themselves — the largest lever on a fit's cost |
