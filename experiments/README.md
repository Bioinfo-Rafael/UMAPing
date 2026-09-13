# UMAPing experiments: local/global out-of-sample evaluation

This directory extends the core `umaping` package (see the top-level
`README.md` and `docs/method.md`/`docs/Paper.md`) with:

- **Experiment A**: analysis-only diagnostics for already-trained runs
  (`umaping analyze-advanced`) -- repulsion-field smoothness/denoising, and
  per-query "did this land somewhere weird" tail statistics.
- **Experiments B/C/D**: new datasets, a common experiment
  download/prepare/train/evaluate/analyze framework, and a common baseline
  interface, to test the inductive method's behavior on real
  local-geometry, cluster-periphery, and continuous-structure benchmarks
  beyond the original COIL-20/COIL-100/pancreas three.
- **A scale benchmark** (Section 6): offline vs. online cost as the
  reference set size N grows, on synthetic data.

**No real dataset was downloaded, no real training was run, and no real
experiment was executed locally while building any of this** -- see Section
9. Every new code path was instead validated with unit tests against
synthetic/injected data and a full, tiny, synthetic end-to-end run of the
"mock" dataset -- see `tests/test_new_datasets.py`,
`tests/test_experiments_framework.py`, `tests/test_advanced_analysis.py`,
`tests/test_scale_benchmark.py`, and `tests/test_inference.py`'s new exact-
repulsion-mode test.

## 1. The four research questions

- **A (analysis-only).** For an already-trained run: is the learned
  repulsion field `B_phi` smoother than a finite-sample Monte-Carlo
  estimate of the true mean field it was distilled from, and does it
  actually approximate that true field *better* than one ordinary
  finite-sample estimate does? Separately: which queries land somewhere
  "weird" (poor local-neighborhood preservation, high fuzzy-consistency
  error, high displacement from their true neighbors' barycenter, an
  extreme class-conditional radius), and is that failure concentrated in
  the tail rather than visible in an average metric?
- **B (local geometry / catastrophic OOS errors).** Does explicitly
  recovering query-specific reference neighbors and applying analytic
  local attraction (rather than a purely parametric, neighbor-agnostic
  regression) help preserve fine-grained local structure for OOS queries,
  especially in the worst case, on Fashion-MNIST and 20 Newsgroups?
- **C (OOS repulsion / cluster-periphery effect).** Can amortizing the
  *expected* global repulsive field (rather than either the full UMAP
  repulsion or a uniformly reduced version of it) reduce the
  periphery-placement artifact documented in Islam and Fleischer's "On
  Out-of-sample Embedding in UMAP", without sacrificing local-neighborhood
  fidelity, on MNIST and the Hong et al. Emergency Department dataset?
- **D (continuous structure).** Does a fixed-reference inductive embedding
  preserve a *known* continuous structure (successive scEU-seq labeling
  time points; successive embryoid-body sampling time windows) without
  artificially fragmenting it? **This experiment does not claim that the
  method's optimization-time vector field corresponds to RNA velocity or
  true developmental time** -- it tests structure preservation only.

## 2. Datasets and public download sources

| Experiment | Dataset | Official source | Citation |
|---|---|---|---|
| B | Fashion-MNIST | `torchvision.datasets.FashionMNIST` | Xiao, Rasul, and Vollgraf (2017), arXiv:1708.07747 |
| B | 20 Newsgroups | `sklearn.datasets.fetch_20newsgroups` | Lang (1995), ICML 1995 |
| C | MNIST | `torchvision.datasets.MNIST` | LeCun, Cortes, and Burges, http://yann.lecun.com/exdb/mnist/ |
| C | Hong et al. Emergency Department | https://github.com/yaleemmlc/admissionprediction (`Results/5v_cleandf.RData`) | Hong, Haimovich, and Taylor (2018), PLOS ONE 13(7): e0201016 |
| D | Murine intestinal organoid scEU-seq | Figshare article 23737170 | Battich et al. (2020), Science 367(6483) |
| D | Embryoid body development | Figshare article 23737416 | Moon et al. (2019), Nature Biotechnology 37 |

Every loader records its own citation/source URL as `CITATION`/`SOURCE_URL`
module constants (`src/umaping/data/{fashion_mnist,twenty_newsgroups,
mnist_oos,hong_ed,organoid,embryoid_body}.py`) and downloads only from the
sources above -- see each module's docstring for exact mechanics (official
package loader vs. the Figshare API vs. a GitHub raw-content fetch).

**Schema/source caveats, stated explicitly rather than silently assumed**
(none of these could be confirmed by an actual download during
implementation):

- **Hong ED**: the exact branch name (`main` vs. `master`) is not confirmed;
  `download_hong_ed` tries both. The outcome/patient-ID columns are
  auto-detected by name-matching, not hardcoded -- confirm on first real use
  (the function fails loudly, listing the actual columns, if detection
  fails) and see `configs/hong_ed.yaml`'s note about `dataset.input_dim`
  being a placeholder until the real post-encoding feature count is known.
- **Organoid / Embryoid body**: the exact file format inside each Figshare
  article, and the exact `obs` metadata column naming the state/time group,
  are not confirmed. `data/_scrna_common.py::load_anndata_any_format`
  dispatches on whatever file extension is actually downloaded; the
  grouping column is auto-detected and the loader fails loudly (listing the
  actual `obs` columns) if none of the candidate names match.

## 3. Baseline methods and citations

See `src/umaping/experiments/baselines.py` for the full implementation of
each. Baselines already implemented for the original three datasets
(standard `umap-learn`, ours, ours + oracle neighbors, no-repulsion) are
*reused*, never reimplemented.

1. **Standard `umap-learn`** -- `umap.UMAP(...).fit(reference).transform(query)`.
2. **Reduced-repulsion UMAP OOS variant** -- fits identically to (1), then
   temporarily lowers the fitted reducer's `negative_sample_rate` (verified
   directly against the installed `umap-learn` source: `UMAP.transform()`
   reads this public attribute) before calling `.transform()`. A practical,
   best-effort reproduction of "reduce repulsion at OOS-embedding time" (the
   idea motivating recent OOS-UMAP work), not a byte-for-byte
   reimplementation of any specific paper's code.
3. **Weighted kNN interpolation** -- places each query at the UMAP-fuzzy-
   weighted barycenter of its true high-dimensional k-NN reference points'
   positions in an already-computed 2D reference layout. No optimization at
   query time at all.
4. **Parametric UMAP** -- `umap.parametric_umap.ParametricUMAP` (part of
   `umap-learn` itself, requires the optional `tensorflow` dependency).
   Gracefully reports unavailable if `tensorflow` isn't installed.
5. **NUMAP / Sep-SpectralNet** -- the method introduced in Ben-Ari, Yacobi,
   and Shaham (2025), "Generalizable Spectral Embedding with an Application
   to UMAP" (TMLR; arXiv:2501.11305). **No official pip package or verified
   source repository was located during this implementation** -- this
   baseline is marked unavailable with instructions to integrate the
   authors' own code as an out-of-repo, subprocess-based adapter (Section 7
   below) once located, rather than a guessed reimplementation.
6. **ParamRepulsor** -- similarly, no official pip package or verified
   source repository was located during this implementation (external
   network access to search for one was not exercised, per this task's own
   no-download-during-implementation constraint). Marked unavailable with
   the same integration path as (5).
7. **Ours** -- the full method (`InferenceEngine`, learned retriever +
   analytic attraction + learned repulsion).
8. **Ours + oracle neighbors** -- `InferenceEngine(neighbor_source="oracle")`.
9. **No-repulsion ablation** -- `InferenceEngine(use_repulsion=False)`.
10. **Exact/high-M repulsion diagnostic** -- `InferenceEngine(repulsion_mode="exact")`,
    new in this branch: every reference point contributes to the repulsive
    field exactly once (`dynamics.py::exact_all_reference_mean_field`,
    chunked), never sampled with replacement the way the existing
    `oracle_mc` mode is -- O(N) per query, so restricted to a query subset
    (`eval.repulsion_oracle_query_subset`).

All baselines use the same reference/query split and reference-only-fitted
input feature representation as `ours` (via the shared `PreparedDataset`),
except where a baseline's own official form requires a different
representation (none of the currently-integrated baselines do).

## 4. Exact commands: Experiments B, C, D

```bash
pip install -e ".[all,experiments]"   # adds torchvision + pyreadr on top of the base install

# One command each: download -> prepare -> train -> evaluate -> analyze ->
# analyze-advanced -> baselines.
umaping experiment --experiment fashion_mnist     --run-dir runs/fashion_mnist/main     --device auto --seed 0
umaping experiment --experiment twenty_newsgroups --run-dir runs/twenty_newsgroups/main --device auto --seed 0
umaping experiment --experiment mnist_oos         --run-dir runs/mnist_oos/main         --device auto --seed 0
umaping experiment --experiment hong_ed           --run-dir runs/hong_ed/main           --device auto --seed 0
umaping experiment --experiment organoid          --run-dir runs/organoid/main          --device auto --seed 0
umaping experiment --experiment embryoid_body     --run-dir runs/embryoid_body/main     --device auto --seed 0
```

`--download-only` and `--prepare-only` stop early (see Section 8); `--resume`
continues a partially-completed run, exactly like `umaping pipeline`.
**Before the first real `hong_ed` run**, run with `--prepare-only` first,
read the reported `input_dim`, and update `experiments/configs/hong_ed.yaml`
accordingly (see that file's own comment).

## 5. Exact command: Experiment A (analysis-only)

Requires an already-trained run (any of the six above, or the original
`coil20`/`coil100`/`pancreas`/`mock`):

```bash
umaping analyze-advanced --run-dir runs/<dataset>/main --device auto
```

Fails loudly (never silently retrains) if required checkpoints/memory/cache
artifacts are missing.

## 6. Scale benchmark

```bash
umaping scale-benchmark --output-dir runs/scale_benchmark --device auto \
  --n-reference-values 500,1000,2000,5000,10000 --n-query 200
```

Uses the synthetic mock dataset at increasing N (never a real dataset at
scale) -- see `src/umaping/experiments/scale_benchmark.py`'s module
docstring for exactly what "offline" and "online" cost each include, and
why this does not claim O(1) inference.

## 7. Optional baseline installation / integrating NUMAP and ParamRepulsor

- **Parametric UMAP**: `pip install umap-learn[parametric_umap]` (or
  `pip install tensorflow` directly).
- **NUMAP/Sep-SpectralNet, ParamRepulsor**: no official pip package was
  located during this implementation (see Section 3). To integrate either:
  1. Locate and clone the authors' official repository into a separate
     directory *outside* this one (never vendor a large third-party repo
     into `umaping`).
  2. Install its dependencies into an isolated environment (a separate
     virtualenv/conda env, so its pins never conflict with this package's).
  3. Add a thin subprocess-based adapter to
     `src/umaping/experiments/baselines.py` (matching the
     `run_parametric_umap_baseline`/`BaselineResult` contract) that shells
     out to that isolated environment's Python with the reference/query
     features written to a temp file and the resulting embedding read back
     -- never a re-implementation of the method's own code inside this
     repository.

## 8. Expected output files

For each `umaping experiment` run, under `<run-dir>/`:

```
metrics/
  retriever.json, spectral.json, field.json, embedding.json, per_query.csv   # unchanged base metrics (pipeline.py)
  advanced_analysis.json, advanced_per_query.csv                              # Experiment A
  baseline_comparison.csv, timing.json                                       # Experiments B/C/D baselines
figures/
  embedding_*.png, loss_*.png, retriever_recall.png, query_trajectories.png,
  repulsion_field_t*.png                                                     # unchanged base figures
  field_smoothness_by_t.png, field_denoising_error_by_t.png,
  query_neighbor_recall_distribution.png, query_fuzzy_error_distribution.png,
  query_tail_failure_comparison.png, query_periphery_score.png              # Experiment A
```

`--download-only` stops after downloading (nothing under `<run-dir>` is
created); `--prepare-only` stops after writing `cache/prepared_dataset.npz`
(no checkpoints yet).

For `umaping scale-benchmark`, under `<output-dir>/`:
`metrics/scale_benchmark.json`, `figures/scale_{query_latency,memory,offline_cost}_vs_n.png`.

Every run also carries the existing `config.yaml` + `metadata.json`
(package versions; git commit is not currently recorded automatically --
add it manually to a run's notes if needed for a specific report) inherited
from `pipeline.py::init_run_dir`.

## 9. Reproducibility guarantees

- Every learned preprocessing step (PCA, TF-IDF+SVD, HVG selection,
  z-scoring/one-hot encoding) is fit on the reference split only and merely
  *applied* to the query split -- checked directly by
  `tests/test_new_datasets.py` for all six new datasets (mirroring the
  existing invariant checks in `tests/test_data_splits.py` for pancreas).
- Every split uses a fixed seed; Hong ED's reproducible subset modes
  (`small`/`medium`/`full`) sample from the reference side only, and
  identically across repeated calls with the same seed (checked directly).
- Exact split indices are not currently written to a separate file (unlike
  the repulsion-oracle-diagnostic's `query_subset_indices`, which is);
  `metadata.json`'s existing package-version snapshot plus each config's
  fixed seed are what currently make a run reproducible end to end.

## 10. Confirmation: no real data was downloaded or run during implementation

**No real dataset was downloaded, and no real training/experiment was run,
against any of Fashion-MNIST, 20 Newsgroups, MNIST, Hong ED, the organoid
data, or the embryoid-body data, at any point while building this branch.**
Every new code path was instead validated against synthetic/injected data
and a full, seconds-long, synthetic end-to-end run through `umaping
experiment` (via a temporarily-registered fake experiment pointing at
`configs/mock.yaml`):

- `tests/test_new_datasets.py` -- all six loaders, via each one's
  `train_data`/`test_data`/`df`/`adata` injection parameter.
- `tests/test_experiments_framework.py` -- the registry, every baseline
  function, and a full `run_experiment` cycle (download-only,
  prepare-only, full run, resume, and the double-run-without-resume
  rejection).
- `tests/test_advanced_analysis.py`, `tests/test_scale_benchmark.py`,
  `tests/test_inference.py::test_embed_one_exact_repulsion_matches_brute_force_and_is_deterministic` --
  Experiment A and the scale benchmark, and the new `"exact"` repulsion mode.

## 11. What each new metric means

- **Field roughness (A)**: finite-difference/second-order (discrete
  Laplacian)/angular/magnitude variation of a vector field sampled on a
  grid -- how much the field's direction and magnitude change between
  neighboring points. Lower generally means smoother.
- **MC denoising test (A)**: does `B_phi` sit closer to a well-averaged
  estimate of the true mean repulsive field than one ordinary finite-sample
  Monte-Carlo draw does? Reported as vector MSE/cosine similarity/magnitude
  error, both against an R-repeat average and (when the reference set is
  small enough) the exact all-reference field.
- **Multi-k recall / NDCG (A)**: does a query's k-NN in the final 2D
  embedding (searched only against the reference embedding) recover its
  true high-dimensional k-NN, at several k?
- **Fuzzy-weighted MSE/BCE (A)**: is the query close, in 2D, to the
  reference points UMAP's own fuzzy membership says it belongs to?
- **Local displacement (A)**: distance from the query to the (weighted)
  barycenter of its true neighbors, normalized by those neighbors' own
  spread -- a diagnostic, not used alone.
- **Tail summary (A)**: mean/median/p90/p95/p99/worst-5%/worst-1% of a
  per-query metric, so a rare catastrophic misplacement isn't hidden by an
  average.
- **Periphery percentile (A)**: a query's radial distance from its own
  label's reference-embedding centroid, expressed as a percentile relative
  to same-label reference points. A diagnostic, not a ground-truth measure.
- **Repulsion accumulation score (A)**: this document's own
  operationalization of the qualitative "repulsion effect" Islam and
  Fleischer describe (points pushed toward a cluster's periphery when
  repulsive directions align) -- **not a verified reproduction of a
  specific equation from their paper**. The norm of the mean unit direction
  from a query's true neighbors toward its final position; near 0 means
  well-surrounded, near 1 means pushed to one side.
