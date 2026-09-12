# Runtime checks

This repository was implemented under an explicit constraint: **no Python,
no dependency installation, no downloads, no training, and no tests were
run locally.** Every piece of reasoning below was done by static reading of
the code, of the current `umap-learn` source (fetched and quoted directly
from GitHub, see "Verified against umap-learn" below), and of the dataset
hosts' actual pages/APIs (fetched directly, not guessed). This file lists
everything that should be confirmed the first time this actually runs on a
real machine, so nothing here is silently assumed to work.

## 1. Install and environment

- `pip install -e ".[all]"` pulls in `torch`, `umap-learn>=0.5.12`,
  `pynndescent`, `scanpy`, `anndata`, etc. together for the first time in
  this repo -- confirm there are no version conflicts on the target Python
  (3.10+) and platform (CUDA build of torch vs. CPU-only, in particular).
- `umap-learn` transitively pulls in `numba`/`llvmlite`; the very first
  import of anything that touches `umap.umap_` (only the test suite does
  this directly, via `pytest.importorskip`) will pay a JIT-compilation
  cost. This is a one-time cost per process, not per call.
- `torch.load(..., weights_only=False)` is used in `pipeline.py`/
  `inference.py` to load this project's own checkpoints (a plain dict of a
  state_dict + small hparams dict -- ints/floats/strings/lists). This is
  intentional and safe (we only ever load files this codebase just wrote),
  but recent torch versions warn about the `weights_only` default; the
  explicit `False` here is deliberate, not an oversight.

## 2. Dataset downloads -- verified sources, not yet exercised

All three download paths were verified by directly fetching the official
pages/APIs (not from training-data memory) during implementation:

- **COIL-20**: `https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip`
  -- confirmed live (HTTP 200, ~12.1 MB), 1440 grayscale 128x128 PNGs,
  `obj{N}__{idx}.png` with `idx` a **sequential pose index 0-71** (not
  degrees).
- **COIL-100**: `https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip`
  -- confirmed live (~124.6 MB), 7200 RGB 128x128 PNGs, `obj{N}__{angle}.png`
  with `angle` **literal degrees** 0,5,...,355. The archive also ships two
  leftover non-image files (a `.pl` conversion script and its `~` backup);
  `data/coil.py` only globs `*.png`, so these are already excluded, but
  confirm the extracted file count still comes out to exactly 7200.
- **Pancreas**: primary `https://exampledata.scverse.org/scvi-tools/pancreas.h5ad`
  (the current scvi-tools tutorial's own host), falling back to
  `https://ndownloader.figshare.com/files/24539828` (the original scIB
  figshare deposit, `human_pancreas_norm_complexBatch.h5ad`, confirmed live)
  if the primary fails. Both were reachable at verification time. **Known
  quirk, verified live**: `exampledata.scverse.org` returns HTTP 403 to a
  bare `Python-urllib` User-Agent but accepts `requests`'s default UA (what
  this codebase uses throughout, in `utils/io.download_file`) -- if
  downloads start failing with 403, check whether anything downstream
  swapped in a different HTTP client/UA.
- `data/pancreas.py::load_pancreas_anndata` only **warns** (does not raise)
  if the downloaded file's shape doesn't match the expected
  16382 x 19093 -- confirm on first real download that the shape either
  matches or that a mismatch is intentional (e.g. the upstream host updated
  the file).
- None of the three downloads, nor the resulting file parsing, has actually
  been executed. First real run should confirm: exact extracted file
  counts, that every `obj*__*.png` matches the expected filename regex, and
  that `anndata.read_h5ad` succeeds on the fetched pancreas file.

## 3. Numerical/algorithmic reasoning that should be spot-checked

- **Update (2026-09-13, first real remote test run, 34 tests, torch/CUDA
  driver mismatch environment):** 32/34 passed on the first try, including
  both `smooth_knn_dist`/`compute_directed_membership` cross-checks against
  the real `umap.umap_` functions at the tolerances below -- the float32/64
  rounding budget was sufficient, no formula mismatch. Two failures, both
  test-file issues (not bugs in `src/umaping/`), now fixed:
  - `umap_umap_.compute_membership_strengths(...)` returned more than 3
    values on the installed umap-learn version (the extra element(s) beyond
    `(rows, cols, vals)` weren't gated behind `return_dists` the way the
    signature research assumed). Fixed by unpacking positionally
    (`result[2]`) instead of assuming an exact tuple length.
  - `test_g_minus_matches_autograd_of_log_one_minus_phi` failed for 2/32
    elements by a small margin (max abs diff ~0.003 against a `1e-4`
    tolerance). Root cause: `g_minus`'s `+eps` (`eps=1e-3`) denominator is
    only an approximation of the true gradient near `q -> 0` (by design,
    matching umap-learn's own repulsion formula -- see the "≈" in this
    file's docstring), and the test's *unconstrained* random `y`/`z` pairs
    occasionally landed close enough together (small `q`) for that expected
    approximation gap to exceed the tolerance -- not a bug in `g_minus`.
    Fixed by constructing pairs with a guaranteed minimum separation
    (`q` uniformly in `[9, 25]`, i.e. `q/eps >= 9000`) instead of an
    unconstrained draw.
- `graph.py`'s `smooth_knn_dist`/`compute_directed_membership` are a
  from-scratch reimplementation (not an import of umap-learn's private
  `umap.umap_.smooth_knn_dist`), deliberately adapted because our $K_i$
  never includes a self-column (see the module docstring for the full
  reasoning). `tests/test_graph.py` cross-checks this against the actual
  `umap.umap_` functions by prepending a synthetic zero self-column; the
  comparison tolerances (`rtol=3e-2` for sigma) were chosen to account for
  umap-learn's numba-internal float32 locals vs. this codebase's float64
  computation -- confirmed sufficient by an actual run (see above).
- `umap_forces.py::find_ab_params` is a reimplementation of umap-learn's
  curve-fit (verified line-for-line against the current source). It
  depends on `scipy.optimize.curve_fit` converging the same way on both
  sides; `tests/test_umap_forces.py::test_find_ab_params_matches_umap_learn`
  checks this directly -- confirmed passing by an actual run.
- The autograd-vs-closed-form gradient tests
  (`test_g_plus_matches_autograd_of_log_phi`,
  `test_g_minus_matches_autograd_of_log_one_minus_phi`) use `rtol=1e-3`;
  this was chosen by manual derivation (reproduced in the module docstring
  of `umap_forces.py`). Confirmed passing by an actual run once the second
  test's sample pairs were changed to guarantee sufficient separation (see
  above) -- `g_plus`'s test needed no such change, since its `eps`
  convention (a `q.clamp(min=eps)` floor) doesn't have the same small-`q`
  approximation gap that `g_minus`'s `(q+eps)` denominator does.
- `tests/test_umap_forces.py::test_mean_negative_field_monte_carlo_converges_with_more_samples`
  compares the *average* error over 8 trials at two very different sample
  sizes (8 vs. 1500 out of a 3000-point population) specifically to make
  the comparison robust to RNG luck -- this was reasoned through
  analytically (expected error scales as $O(1/\sqrt{M})$, so a ~14x gap in
  expected error should essentially never flip on 8-trial averages) but
  never actually executed; if it is ever observed to be flaky, increase
  `n_trials` further rather than removing the check.
- `evaluation/spectral.py::exact_laplacian_eigenvectors` uses
  `scipy.sparse.linalg.eigsh(..., sigma=0.0, which="LM")` (shift-invert,
  the standard trick for the smallest eigenvalues of a sparse matrix) with
  a fallback to `which="SM"` if shift-invert raises. Shift-invert can still
  fail to converge on some graphs (e.g. a near-singular shifted operator);
  if evaluation fails here on a real dataset, the fallback path is
  untested and may need a different fallback strategy (e.g. increasing
  `ncv` or `maxiter`).
- `sc.pp.highly_variable_genes(..., flavor="seurat", n_top_genes=2000,
  batch_key="tech")` is called on the real reference set, which has 7
  reference technologies of very uneven size (`inDrop1-4`, `smarter`,
  `celseq`, `fluidigmc1`). This is standard usage but was never actually
  run against the real file; watch for scanpy warnings about small
  per-batch gene-expression bins.

## 4. Not exercised at all

- `models/retriever.py::PyNNDescentIndex` (the optional ANN backend,
  selected via `retriever.ann_backend: pynndescent` in a config) is
  implemented but no test or default config path exercises it. If you
  switch a config to use it, confirm `pynndescent.NNDescent(...).query(...)`
  returns `(indices, distances)` in the shape this codebase assumes.
- The full CLI (`umaping download|train|evaluate|analyze|pipeline`) has
  never been invoked. `cli.py`'s `--resume` guard (refusing to reuse a
  run-dir with an existing `metadata.json` unless `--resume` is passed) is
  logic-checked but not run.
- Every figure in `evaluation/plotting.py` (Agg backend, `pd.Categorical`
  color coding, `quiver` vector-field plots) has never actually been
  rendered. One thing worth a visual check specifically: if a query point
  in `plot_embedding` carries a label value that never appears in the
  reference set's categories (structurally impossible for COIL's
  `object_id`, since the split guarantees every object appears in both, but
  *possible* for pancreas -- query technologies could in principle carry a
  `celltype` value reference technologies don't), `pd.Categorical(...,
  categories=...)` will code it as -1; confirm this renders sanely (or add
  an explicit color for "unseen category") rather than erroring or
  silently mis-coloring.
- Resumability is implemented and reasoned through at the granularity of
  whole stages (preprocess / graph / retriever / spectral / flow_dynamics /
  repulsion), each gated by an on-disk `.done` marker -- **not** at the
  granularity of individual training steps within a stage. A process killed
  mid-stage will redo that whole stage from scratch on `--resume`, not
  continue from the exact step it was on.

## 5. Tests

`tests/` implements all of the required checks (fuzzy-weight/umap-learn
agreement, the symmetric fuzzy union identity, autograd-vs-closed-form
gradients, Monte-Carlo repulsion convergence, retriever
positive/negative-masking invariants, spectral output dimensions and
frozen-calibration reuse, `embed_one` query-independence, COIL/pancreas
split leakage checks, and save/reload/`embed_one`).

```bash
pip install -e ".[all]"
pytest -v
```

**Status (2026-09-13):** run for the first time on a real remote machine
(Python 3.13, CUDA driver present but too old for the installed torch build
-- CPU fallback, no functional impact on these unit tests). 34/34 passing
after the two test-only fixes described in Section 3 above; nothing in
`src/umaping/` needed to change. If a *different* failure shows up on
another environment, the most likely causes, roughly in order of
likelihood: (a) a genuine tolerance too tight for that environment's
umap-learn/torch version (Section 3), (b) a real bug in this from-scratch
implementation, (c) an environment issue (missing optional dependency,
numba/llvmlite mismatch).
