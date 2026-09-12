# UMAPing: Inductive Single-Point UMAP via Learned Neighborhood Retrieval, Spectral Function Approximation, and Mean-Field Repulsion Distillation

*Technical record, not yet a submission draft. This document describes exactly the method implemented in this repository at the time of writing (see Appendix D for a line-by-line mapping from every equation to source code). It is written so that a real paper could later be drafted from it; several sections (Abstract, Introduction, Related Work, Results, Discussion, Conclusion) are kept intentionally concise, while Method, Algorithms, Derivations, and the Appendix are exhaustive.*

**Status:** implementation complete for three datasets (COIL-20, COIL-100, pancreas scRNA-seq); unit test suite passing; end-to-end pipeline runs confirmed on real hardware for two of the three datasets at time of writing (Appendix G). No scientific (accuracy/quality) results are reported in Section 15 — see that section for why.

---

## Abstract

We consider the problem of embedding a single previously unseen point $x_*$ relative to a large, fixed reference dataset $X$, using a UMAP-style (McInnes et al., 2018) low-dimensional layout, without ever constructing a graph over more than one query point at a time and without any query-query interaction. Rather than training one neural network to directly regress $x_* \mapsto y_*$, we decompose the computation into components that mirror the non-parametric structure of UMAP itself: a learned dual-encoder retriever supplies *candidate* reference neighbor identities, from which exact UMAP fuzzy edge weights are recomputed analytically; a learned pointwise spectral encoder supplies an initialization; and a low-dimensional force field integrates this initialization forward in time, using UMAP's closed-form attractive gradient exactly and a *learned* approximation only for the one genuinely expensive, non-parametric quantity — the mean repulsive field exerted by the whole reference set. We give the exact algorithms, state precisely which parts of this pipeline are analytically exact, conditionally exact, or learned approximations, and describe the accompanying evaluation protocol, ablations, and diagnostics. This document does not report scientific accuracy results (Section 15); it documents the method and its verified correspondence to UMAP's formulas (Appendix G) so that such results can be produced and added later.

---

## 1. Introduction

UMAP (McInnes, Healy, and Melville, 2018) is normally applied transductively: the fuzzy simplicial set, its low-dimensional layout, and the stochastic-gradient-descent (SGD) optimization that produces the final embedding are all computed jointly over one fixed dataset. The reference `umap-learn` implementation also exposes an inductive `.transform()` method for new points, and parametric variants exist that train a neural network end to end to reproduce a UMAP-like embedding (Sainburg, McInnes, and Gentner, 2021). This document does not claim to be the first inductive or learned approach to UMAP-style embedding — both already exist, in different forms, and are discussed in Section 2 and Section 9.

What we describe here is a specific *decomposition* of the inductive embedding problem: instead of asking a single neural network to learn the entire map $x \to y$, we identify which parts of UMAP's own construction are already closed-form and inexpensive to evaluate for a single new point given a **fixed, frozen memory** built once from $X$, and which parts genuinely require approximation because they are defined as an expectation, a search, or an eigenproblem over the whole reference set. We amortize exactly those latter pieces with small learned models, and leave the former analytic.

Concretely, given a fixed reference set $X$ (Section 3), we build offline: a directed UMAP fuzzy membership matrix $\mathrm{Mu}$ and its symmetrization $W$ (Section 4); a learned dual-encoder retriever trained against $\mathrm{Mu}$ (Section 5.2); a learned pointwise spectral encoder trained against $W$, with a frozen reference-only post-hoc calibration (Section 5.3); a frozen reference trajectory obtained by integrating UMAP's *expected* (mean-field) dynamics over $X$ (Section 5.4); and a residual network distilling the one expectation that does not have a closed form, the mean repulsive field exerted by $X$ at a given point and time (Section 5.4, Section 6). At inference, a single point $x_*$ is processed using only this frozen memory plus $x_*$ itself (Algorithm 1).

## 2. Related Work

Four bodies of work are directly relevant and are drawn on explicitly, each for a specific component, not for the overall architecture: UMAP itself (McInnes et al., 2018) for the fuzzy graph, kernel, and force formulas we reuse or approximate; Dense Passage Retrieval (Karpukhin et al., 2020) for the dual-encoder retrieval formulation; SpectralNet (Shaham et al., 2018) and Generalizable Spectral Embedding (Ben-Ari et al., 2025) for the pointwise spectral network and its out-of-sample calibration; and Flow Matching (Lipman et al., 2023) for the form of the vector-field regression objective used to distill the repulsive field. Section 9 discusses, with direct quotations verified against the source papers, specific limitations in prior out-of-sample and spectral methods that motivate parts of this design, and Section 8 gives a full provenance table with exact equation references for every borrowed component. Parametric UMAP (Sainburg, McInnes, and Gentner, 2021) is the closest prior approach to the same *problem* (learned, inductive UMAP-style embedding) and is discussed on its own terms in Section 9 rather than framed as something this method supersedes.

## 3. Problem Formulation

Let

$$
X = \{x_1, \ldots, x_N\}, \qquad x_i \in \mathbb{R}^D
\tag{3.1}
$$

be a fixed reference dataset, with $D$ a fixed, dataset-specific input dimensionality (COIL-20/100 use a reference-only-fitted PCA projection to 256 dimensions by default; pancreas uses a reference-only-fitted 50-dimensional PCA over 2,000 reference-only-selected highly variable genes — Appendix E). Training may use all of $X$: its pairwise structure, its kNN graph, and any offline optimization over it.

At inference, exactly one previously unseen point

$$
x_* \in \mathbb{R}^D
\tag{3.2}
$$

is available at a time. We seek a map

$$
F_{\Theta, \mathcal{M}_X} : x_* \mapsto y_* \in \mathbb{R}^d,
\tag{3.3}
$$

where $\Theta$ denotes learned neural parameters (the retriever, the spectral encoder, and the repulsion network) and $\mathcal{M}_X$ denotes a *frozen memory* computed once, offline, from $X$ (the reference features themselves, the retriever's key embeddings, the spectral calibration transforms, and the reference trajectory of Section 5.4). $d$ is a small fixed embedding dimension (2 in the default configuration).

The problem is constrained as follows:

- The output $y_*$ for a given $x_*$ must not depend on any other query point. Formally, if $x_a \ne x_b$ are two query points evaluated (in either order, or independently) against the same $\Theta, \mathcal{M}_X$, then $F_{\Theta,\mathcal{M}_X}(x_a)$ must be identical regardless of whether, or when, $F_{\Theta,\mathcal{M}_X}(x_b)$ is also evaluated.
- No graph, attention mechanism, or message-passing structure may be constructed over more than one query point.
- $\mathcal{M}_X$ is frozen at inference time: no part of it (in particular, no stored reference trajectory position) may be updated as a function of $x_*$.
- Each dataset has one fixed input dimensionality $D$ and one corresponding model instance; the method does not support variable-dimensional inputs within one trained instance.

Section 7 states precisely which properties of $F_{\Theta,\mathcal{M}_X}(x_*)$ hold exactly, which hold conditionally (on a stated, checkable event), and which are learned approximations only.

## 4. Method Overview

The reference-side computation and the query-side computation are structurally different, and this asymmetry is the organizing idea of the method.

**Offline, using all of $X$:**

$$
X \;\xrightarrow{\ \text{kNN + smooth-kNN}\ }\; \mathrm{Mu}, W
\;\xrightarrow[\ \text{spectral graph}\ ]{\ \text{retrieval teacher}\ }\;
\big(\text{Retriever trained on } \mathrm{Mu}\big),\;
\big(\text{Spectral encoder trained on } W\big)
\tag{4.1}
$$

followed by an offline integration of the *expected* UMAP dynamics over $X$ to obtain a frozen reference trajectory $Y_{\mathrm{ref}}(t)$, and the distillation of a repulsion network $B_\phi(y,t)$ against that trajectory (Section 5.4).

**Online, for one $x_*$:**

$$
x_*
\;\xrightarrow{\ \text{Neighborhood Retriever}\ }\;
\widehat{\mathcal N}_X(x_*),\ \widehat\mu_{*\to j}
\qquad\text{and independently}\qquad
x_*
\;\xrightarrow{\ \text{Spectral Encoder}\ }\;
y_*(0),
\tag{4.2}
$$

then

$$
\frac{dy_*}{dt} = \underbrace{v_{\mathrm{attr}}(y_*, t)}_{\text{analytic}} + \underbrace{v_{\mathrm{rep}}(y_*, t)}_{\text{learned, mean-field}},
\tag{4.3}
$$

integrated by a fixed number of Euler steps to obtain $y_*$ (Algorithm 1).

The point emphasized throughout this document is that Eq. (4.2)–(4.3) is *not* one black-box encoder $x_* \to y_*$. It is:

$$
x_* \to \text{reference neighbor identities} \to \text{analytic UMAP edge weights},
\tag{4.4}
$$

combined with

$$
x_* \to \text{spectral initialization},
\tag{4.5}
$$

followed by

$$
\text{analytic attraction} + \text{learned expected repulsion}.
\tag{4.6}
$$

Table 1 in Section 7 and Appendix D give, respectively, the exact/approximate status and the source-code location of every arrow above.

### 4.1 Directed $\mathrm{Mu}$ versus symmetric $W$ — a distinction used throughout

UMAP's own fuzzy simplicial set construction first builds a *directed* local membership function per point and then symmetrizes it (McInnes et al., 2018; see Section 6.1 and Appendix D for the exact correspondence). This implementation keeps the directed and symmetric objects explicit and never substitutes one for the other:

- **Directed $\mathrm{Mu}$** ($\mu_{i\to j}$): used *only* to train the Neighborhood Retriever (Algorithm 2). It is never symmetrized before being used as a retrieval target.
- **Symmetric $W$**: used *only* for the Spectral Encoder (Algorithm 3) and the reference mean dynamics (Algorithm 4).

At query time, the analogous query-side directed weights $\widehat\mu_{*\to j}$ are recomputed from scratch from exact reranked distances (Algorithm 2); they are never symmetrized, and no $(N+1)$-point graph over $X \cup \{x_*\}$ is ever built (Section 7.5).

## 5. Algorithms

Four algorithms are given. Algorithm 4 covers both the reference mean-field dynamics and the repulsion-field distillation together, since in the implementation they are one coupled procedure (the distillation trains directly against trajectories the dynamics procedure produces) — they are not split into separate numbered algorithms.

### 5.1 Algorithm 1 — Training and Single-Point Inference of UMAPing

> **Input:** reference set $X$; neighbor count $k$; embedding dimension $d$; UMAP configuration (`min_dist`, `spread`, `negative_sample_rate`, ...); training configuration for the retriever, spectral encoder, and repulsion field.
> **Output:** trained retriever (query/key encoders + frozen key index); trained spectral encoder and its frozen calibration; frozen reference trajectory $Y_{\mathrm{ref}}(t)$; trained repulsion field $B_\phi$; the callable `EmbedOne(x_*)`.

```text
function FIT_UMAPING(X, config):

    Mu, W ← BUILD_REFERENCE_GRAPH(X, config.umap)          # Algorithm 2

    Retriever ← TRAIN_RETRIEVER(X, Mu, config.retriever)    # Algorithm 2

    SpectralMap ← TRAIN_SPECTRAL_ENCODER(X, W, config.spectral)   # Algorithm 3

    a, b ← FIT_UMAP_KERNEL(config.umap.spread, config.umap.min_dist)

    Y_ref(t), B_phi ← TRAIN_UMAP_MEAN_FLOW(
        W, SpectralMap, a, b, config.flow, config.repulsion)      # Algorithm 4

    M_X ← STORE(reference features, retriever keys/index,
                spectral calibration, Y_ref(t), a, b, config)

    return {Retriever, SpectralMap, B_phi}, M_X


function EMBED_ONE(x_star, models, M_X):

    ids, distances ← RETRIEVE_AND_RERANK(x_star)     # Algorithm 2
    mu_star        ← COMPUTE_QUERY_UMAP_WEIGHTS(distances)

    y       ← SpectralMap(x_star)                    # Algorithm 3
    s_star  ← SUM(mu_star)

    for e = 0, ..., E-1:
        t     ← e / E
        alpha ← alpha_0 * (1 - e / E)

        neighbor_positions ← Y_ref(t)[ids]           # frozen, read-only

        v_attr ← SUM_j  mu_star[j] * g_plus(y, neighbor_positions[j])
        v_rep  ← negative_sample_rate * s_star * B_phi(y, t)

        v ← CLIP(v_attr + v_rep)
        y ← y + alpha * v

    return y
```

No graph over query points is ever constructed inside `EMBED_ONE`; `Y_ref(t)` is read-only inside this function. Because a second call to `EMBED_ONE` with a different $x_*'$ never writes to `M_X`, the two calls' outputs cannot depend on each other, regardless of call order — this property is checked directly by unit tests (`tests/test_inference.py::test_embed_one_is_independent_of_other_queries`, and, for the stochastic repulsion-oracle diagnostic mode specifically, `test_embed_one_oracle_mc_repulsion_is_reproducible_and_query_independent`). "Batch" inference (`baselines.py::embed_all_queries`) is implemented only as a Python loop calling `EMBED_ONE` once per point and timing each call independently; it is a convenience for reporting throughput, not a vectorized joint computation.

### 5.2 Algorithm 2 — UMAP-Guided Neighborhood Retriever

> **Input:** reference set $X$; neighbor count $k$; the original preprocessed-space Euclidean metric; retrieval configuration (encoder widths, retrieval dimension, temperature $\tau$, candidate pool size, negative-sampling configuration).
> **Output:** directed $\mathrm{Mu}$ and symmetric $W$; trained query/key encoders $f_\theta, g_\psi$; a frozen index over reference key embeddings.

```text
function BUILD_REFERENCE_GRAPH(X, k):

    for i = 1, ..., N:
        K_i, delta_i ← ExactKNN(x_i, X \ {x_i}, k)      # self excluded by construction
        rho_i        ← local-connectivity distance(delta_i)
        sigma_i      ← solve  SUM_{j in K_i} exp(-max(0, delta_ij - rho_i) / sigma_i) = log2(k)

        for j in K_i:
            Mu[i, j] ← exp(-max(0, d(x_i, x_j) - rho_i) / sigma_i)

    W ← Mu + Mu^T - Mu ⊙ Mu^T

    return Mu, W


function TRAIN_RETRIEVER(X, Mu):

    initialize  q_i = f_theta(x_i),  k_j = g_psi(x_j)

    repeat:
        sample query indices I  (with replacement, uniform over X)

        for each i in I:
            j_plus ~  Mu[i, :] / SUM_j Mu[i, j]         # sampled positive

        Q ← L2Normalize(f_theta(X[I]))
        K ← L2Normalize(g_psi(candidate points))         # sampled positives ∪ random negatives

        score(i, j) ← q_i^T k_j / tau

        mask out, per query i, any candidate whose underlying reference
        index equals i, and (by default) any candidate that is a true
        Mu-neighbor of i other than i's own sampled positive

        minimize  L_ret  (Eq. 5.7)  by stochastic gradient descent

    keys ← L2Normalize(g_psi(X))          # computed once, stored
    return {f_theta, g_psi}, keys


function RETRIEVE_AND_RERANK(x_star):

    q_star ← L2Normalize(f_theta(x_star))
    C_star ← top-M candidates by inner product against the frozen key index

    compute exact Euclidean distances d(x_star, x_j) for j in C_star
    K_star ← exact top-k within C_star, by those distances

    rho_star, sigma_star ← solve the same smooth-kNN equation as above,
                            using only these k distances

    mu_star[j] ← exp(-max(0, d(x_star, x_j) - rho_star) / sigma_star),  j in K_star

    return K_star, mu_star
```

**Smooth-kNN construction, cited precisely.** $\rho_i$ (the local-connectivity distance) and the binary search for $\sigma_i$ against a $\log_2 k$ target are exactly McInnes, Healy, and Melville's (2018) construction (p. 14, unnumbered; restated in scalar form as Equation (15), Appendix C, p. 57, of arXiv:1802.03426v3) — with one caveat, found by direct comparison of the paper's own two presentations of this formula: the primary definition (p. 14) includes the $\max(0,\cdot)$ clamp shown above, but the paper's own compact Appendix C restatement (Eq. 15) and its Algorithm 3 pseudocode (p. 19) both omit it, an internal inconsistency in the source paper itself, not an artifact of this document. This implementation follows the primary (clamped) definition, matching the current `umap-learn` software (Appendix G). The one deliberate departure from the paper's own $K_i$ construction: $K_i$ here excludes $x_i$ (or, for a query, the query point cannot be excluded since it is not itself a member of $X$) by construction of the neighbor search itself, so there is never a self/zero-distance column to special-case — this differs from `umap-learn`'s own *internal* implementation, which assumes column 0 of its working distance matrix is such a self-distance and skips it explicitly (verified directly against the current `umap-learn` source; RUNTIME_CHECKS.md, Appendix G), a convention difference documented in `graph.py`'s own module docstring and cross-checked numerically by `tests/test_graph.py`.

**Retrieval model, exactly.** With $f_\theta$ the query encoder and $g_\psi$ the (separately parameterized) key encoder,

$$
q_i = \frac{f_\theta(x_i)}{\lVert f_\theta(x_i)\rVert_2}, \qquad
k_j = \frac{g_\psi(x_j)}{\lVert g_\psi(x_j)\rVert_2}, \qquad
s_{ij} = \frac{q_i^\top k_j}{\tau}.
\tag{5.1}
$$

This follows the general form of the dual-encoder dot-product score of Dense Passage Retrieval, $\mathrm{sim}(q,p)=E_Q(q)^\top E_P(p)$ (Karpukhin et al., 2020, Eq. 1, p. 6770/p. 2). **Precisely what is, and is not, DPR's own convention, verified directly against the source**: DPR's own score is a *raw, unnormalized* dot product with no temperature scaling — the paper explicitly evaluates cosine similarity (which requires normalized vectors) as one alternative among several and reports that "L2 performs comparable to dot product, and both of them are superior to cosine" before choosing the plain inner product (Karpukhin et al., 2020, Section 5.2/Appendix, p. 6774); its softmax loss (Eq. 2, below) uses the similarity score directly, with no dividing temperature constant anywhere in the paper. **The L2-normalization and the temperature $\tau$ in Eq. (5.1) above are therefore modifications made in this implementation, not features already present in DPR's own Eq. (1)–(2).**

**Positive sampling.** For a reference point $i$ used as a training query, a positive key is sampled proportionally to $\mathrm{Mu}$'s $i$-th row:

$$
P(j^+ = j \mid i) = \frac{\mu_{i \to j}}{\sum_\ell \mu_{i \to \ell}}, \qquad j \in K_i.
\tag{5.2}
$$

**Loss.** With $\mathcal C_i$ the set of in-batch candidate keys for query $i$ (the sampled positives of every query in the minibatch, plus optional additional random negatives, with the masking described above),

$$
\mathcal L_{\mathrm{ret}} = -\log \frac{\exp(s_{i,j^+})}{\sum_{\ell \in \mathcal C_i} \exp(s_{i\ell})}.
\tag{5.3}
$$

This is the DPR-style contrastive / in-batch-negative negative-log-likelihood loss (Karpukhin et al., 2020, Eq. 2, p. 6771/p. 3, with in-batch negatives via $S=QP^\top$ introduced in the same section). **What is borrowed and what is not, precisely:** the *loss form* — a softmax negative-log-likelihood over one positive and a set of negatives drawn from the same minibatch — is DPR's. DPR's positives are given (question, gold passage) pairs; there is no analogue of $\mathrm{Mu}$ in DPR. The modification made here is entirely in how the positive $j^+$ is *sampled*: proportionally to UMAP's directed fuzzy membership row (Eq. 5.2), not from a fixed label. The loss in Eq. (5.3) itself does not contain $\mu_{i\to j}$ anywhere; nothing in a single gradient step "sees" the fuzzy weights except through which $j^+$ happened to be sampled. It is the *repeated* resampling of $j^+ \sim \mathrm{Mu}[i,:]$ across many training steps that makes the expected gradient behave as if the objective were a fuzzy-weighted multi-positive one — this is a consequence of the sampling scheme, not a claim about any single step's loss value.

**Candidate-recall lemma.** Let $K_X(x_*)$ denote the true exact top-$k$ Euclidean neighbors of $x_*$ within $X$ (in the fixed preprocessed feature space), and let $C_*$ be the candidate set returned by the learned retriever (size $M = \max(4k, 64)$ by default; Appendix C). If

$$
K_X(x_*) \subseteq C_*,
\tag{5.4}
$$

then, because `RETRIEVE_AND_RERANK` selects the exact top-$k$ *within* $C_*$ by the same Euclidean metric used to define $K_X(x_*)$, and the $k$ points of $K_X(x_*)$ are by definition the $k$ closest points to $x_*$ in the whole of $X$ (hence also the $k$ closest within any subset of $X$ that contains all of them), it follows that

$$
\widehat K_* = K_X(x_*).
\tag{5.5}
$$

Since $\rho_*, \sigma_*, \widehat\mu_{*\to j}$ are computed by a deterministic function of exactly these $k$ (identity, distance) pairs (the same smooth-kNN solve used at reference-graph-construction time, Eq. in Algorithm 2), Eq. (5.5) implies

$$
\widehat\rho_* = \rho_*^{\mathrm{true}}, \qquad \widehat\sigma_* = \sigma_*^{\mathrm{true}}, \qquad \widehat\mu_{*\to j} = \mu_{*\to j}^{\mathrm{true}} \ \ \forall j \in \widehat K_*.
\tag{5.6}
$$

A full proof is given in Appendix B.6. **The practical consequence is the central design property of Algorithm 2**: the learned retriever's only job is to make Eq. (5.4) hold with high probability (candidate recall); it never has to learn to reproduce a fuzzy weight value, because whenever recall succeeds, the weight is recomputed exactly. Two important caveats: (i) Eq. (5.4)–(5.6) are conditional, not unconditional — if recall fails, $\widehat K_*$ is generally a *different*, still well-formed set (Section 7.2); (ii) this reconstructs only the *query-side directed* membership $\widehat\mu_{*\to j}$. It does not reconstruct $\mu_{j \to *}$ for reference points $j$, and it is therefore not equivalent to rebuilding a fully symmetric graph over $X \cup \{x_*\}$ (Section 7.5).

### 5.3 Algorithm 3 — Learned Spectral Initialization

> **Input:** reference set $X$; symmetric graph $W$; embedding dimension $d$.
> **Output:** a pointwise spectral map $S_\omega(x)$ usable on a single new point with no further graph access.

With degree $D_{ii} = \sum_j W_{ij}$, the symmetric normalized Laplacian is

$$
L_{\mathrm{sym}} = I - D^{-1/2} W D^{-1/2}.
\tag{5.7}
$$

The network's raw output dimension is $r = d + 1$ ($r=3$ for the default $d=2$).

```text
function TRAIN_SPECTRAL_ENCODER(X, W):

    D     ← degree(W)
    L_sym ← I - D^(-1/2) W D^(-1/2)

    initialize  h_omega : R^D -> R^(d+1)

    repeat:
        sample a minibatch of undirected W-edges (i, j, W_ij)   # i<j, each stored once
        gather the unique node indices touched by this batch
        z_unique ← h_omega(X[unique indices])                    # one forward pass per unique node

        z_i, z_j ← z_unique gathered back to the batch's edge endpoints

        L_dir  ← MEAN_edges  W_ij * || z_i / sqrt(D_i) - z_j / sqrt(D_j) ||^2
        L_orth ← || Z_unique^T Z_unique / U - I ||_F^2            # U = #unique nodes in this batch

        minimize  L_dir + lambda_orth * L_orth

    R ← h_omega(X)                        # batched forward pass over all of X
    C ← R^T R / N
    T ← C^(-1/2)                          # exact reference-only whitening
    Z ← R T                               # (1/N) Z^T Z = I, exactly

    H ← Z^T L_sym Z                       # small (d+1)x(d+1) matrix
    eigenvalues, Q ← Eigh(H)               # ascending
    discard the lowest (trivial) mode
    P ← the next d eigenvectors

    Y0 ← Z P
    center, scale ← reference-only calibration(Y0)   # mean-subtract, then scale to a fixed target spread

    return  x |-> (h_omega(x) T P - center) * scale
```

**Graph-energy identity used by the training objective.** For any $Z \in \mathbb R^{N \times r}$,

$$
\operatorname{Tr}\!\left(Z^\top L_{\mathrm{sym}} Z\right)
= \frac12 \sum_{i,j} W_{ij} \left\lVert \frac{z_i}{\sqrt{D_{ii}}} - \frac{z_j}{\sqrt{D_{jj}}} \right\rVert_2^2.
\tag{5.8}
$$

A full derivation is given in Appendix B.2. The implementation's minibatch Dirichlet term is a stochastic (edge-subsampled) estimator of the right-hand side of Eq. (5.8), computed on the network's *raw* (not yet whitened) outputs. **Precise correspondence to SpectralNet** (Shaham et al., 2018): the right-hand side of Eq. (5.8) has the form of that paper's *normalized*, degree-scaled pairwise loss, $\frac{1}{m^2}\sum_{i,j}W_{i,j}\lVert y_i/d_i - y_j/d_j\rVert^2$ (their Eq. 5, Section 3.1, p. 5) — not their base, unnormalized pairwise loss $\frac{1}{m^2}\sum_{i,j}W_{i,j}\lVert y_i-y_j\rVert^2$ (their Eq. 3, same section), which lacks the degree normalization used throughout this document's construction.

**Orthogonality term, exactly as implemented.** With $Z_{\mathrm{unique}}$ the raw network outputs for the $U$ unique nodes touched by a training minibatch (note: $U$, the number of nodes touched by a random *edge* sample, not the full reference count $N$),

$$
\mathcal L_{\mathrm{orth}} = \left\lVert \frac{1}{U} Z_{\mathrm{unique}}^\top Z_{\mathrm{unique}} - I \right\rVert_F^2,
\qquad
\mathcal L_{\mathrm{spectral}} = \widehat{\mathcal L}_{\mathrm{Dirichlet}} + \lambda_{\mathrm{orth}} \, \mathcal L_{\mathrm{orth}}.
\tag{5.9}
$$

This is implemented as an additive soft penalty on the loss. **This is a deliberate departure from SpectralNet's own mechanism, verified directly against the source rather than assumed:** Shaham et al. (2018) enforce their analogous minibatch constraint, $\frac1m Y^\top Y = I$ (their Eq. 4, Section 3.1, p. 4), with a *hard* architectural constraint, not a soft penalty — "a special-purpose output layer" (their Abstract, p. 1) that orthogonalizes the minibatch output via a linear map computed from the QR/Cholesky decomposition of the batch's Gram matrix, with training alternating between orthogonalization steps (which set that layer's weights in closed form) and ordinary gradient steps (their Section 3.1 and Algorithm 1, p. 4/p. 6). SpectralNet's own paper explicitly attributes a soft-penalty implementation of orthogonality to a *different* prior method (Yi et al., 2016), contrasting it with their own hard-constraint layer: "we enforce orthogonalization stochastically through a constraint layer, while they attempt to learn orthogonalized functional maps by adding an orthogonalization term to the loss function, which involves non-trivial balancing between two loss components" (Shaham et al., 2018, Section 2, p. 3). **The mechanism used in Eq. (5.9) is the soft-penalty style SpectralNet attributes to Yi et al. (2016), not SpectralNet's own hard-constraint mechanism** — a simpler choice made for this implementation, relying on the exact, closed-form, reference-only whitening of Eq. (5.10) after training to obtain exact orthonormality rather than enforcing it architecturally throughout training.

**Post-hoc, reference-only calibration (exact, not learned).** Training only encourages $\mathcal L_{\mathrm{orth}} \approx 0$; it does not guarantee it. After training, the raw outputs are made *exactly* orthonormal (up to floating-point precision) using reference data only: with $R = h_\omega(X) \in \mathbb R^{N \times r}$ and $C = \frac1N R^\top R = U_C \Lambda_C U_C^\top$,

$$
T = U_C \Lambda_C^{-1/2}, \qquad Z = R T, \qquad \frac1N Z^\top Z = I \ \ (\text{exactly}).
\tag{5.10}
$$

The small "projected operator" (this document's own descriptive term; see the note below)

$$
H = Z^\top L_{\mathrm{sym}} Z \in \mathbb R^{r \times r}
\tag{5.11}
$$

is eigendecomposed; its lowest (trivial) mode is discarded and the next $d$ modes give a projection $P \in \mathbb R^{r \times d}$.

*Relation to Ben-Ari, Yacobi, and Shaham (2025).* This step draws on the eigenvector-separation idea of Generalizable Spectral Embedding, applied here to the UMAP-specific graph rather than to a generic affinity graph. That paper proves that a SpectralNet-style network trained under an orthogonality constraint cannot by itself resolve the rotation/reflection ambiguity of its output basis ("every minimizer ... is of the form $VQ$, where $V$ is the first $k$ eigenvectors matrix of $L$ and $Q$ is an arbitrary ... orthogonal matrix," their Lemma 1, Section 4.1), and resolves it by forming a small matrix from the (still rotation-ambiguous) trained basis and the target operator, $(VQ)^\top L(VQ) = Q^\top\Lambda Q$, then diagonalizing it (their Section 4.2) — exactly the same construction as Eq. (5.11) here, with $Z$ playing the role of their $VQ$ and $L_{\mathrm{sym}}$ the role of their $L$. That paper itself calls this an "SE approximation" / eigendecomposition-based post-processing procedure, not a "projected operator" — that phrase does not appear anywhere in their text and is only this document's own label for $H$. Two further precisions: this construction is not a reimplementation of their specific method (named "Sep-SpectralNet"/"NUMAP" in their paper, which trains a *second* network against UMAP's own contrastive loss on top of the disambiguated spectral output); Algorithm 4 instead integrates an explicit force-based dynamics on top of the spectral initialization. The frozen pointwise map is

$$
S_\omega(x) = \gamma \big[ h_\omega(x)\, T\, P - c \big],
\tag{5.12}
$$

with $c$ (a $d$-vector) and $\gamma$ (a scalar) a reference-only center and scale calibrated so the coordinates sit at a scale appropriate for the force equations of Section 6 (Appendix C gives the exact calibration rule). $T$, $P$, $c$, $\gamma$ are all fit once, from reference data only, and then applied identically to any query point: $x_* \to h_\omega(x_*) \to$ stored linear transforms $\to y_*(0)$, with no reference-graph access at query time.

**What this does and does not establish.** Under idealized conditions — sufficient network capacity, exact optimization of Eq. (5.9), exact orthogonality, and a non-degenerate spectral gap around the retained modes — the Rayleigh–Ritz variational characterization of eigenvalues implies

$$
\operatorname{span}(Z) = \operatorname{span}(U_{1:d+1}),
\tag{5.13}
$$

where $U_{1:d+1}$ are the true bottom $d{+}1$ eigenvectors of $L_{\mathrm{sym}}$. This is a subspace target, not a per-point equality, and it is an *idealized* statement, not a guarantee about the trained network. Section 7.3 states this precisely, and evaluation (`evaluation/spectral.py`) reports the empirical subspace/principal-angle gap against an exact `scipy.sparse.linalg.eigsh` solution, used for evaluation only, never as a training signal. Critically, **we do not claim**

$$
S_\omega(x_*) = \operatorname{SpectralEmbedding}\big(W(X \cup \{x_*\})\big)_*,
\tag{5.14}
$$

because inserting $x_*$ into the graph changes $W$ itself (Section 7.5).

### 5.4 Algorithm 4 — UMAP Mean-Field Dynamics and Repulsion Distillation

> **Input:** symmetric graph $W$; the spectral map $S_\omega$ of Algorithm 3; UMAP kernel parameters $a, b$; negative-sample rate $m$; number of flow steps $E$.
> **Output:** a frozen reference trajectory $Y_{\mathrm{ref}}(t)$, $t \in \{0, \tfrac1E, \ldots, 1\}$; a trained repulsion network $B_\phi(y,t)$; the single-query update rule used inside `EMBED_ONE`.

```text
function TRAIN_UMAP_MEAN_FLOW(W, SpectralMap, a, b):

    Y ← SpectralMap(X)              # Y_ref(0)
    s_i ← SUM_j W_ij                # row mass ("degree")
    store Y_ref(0)

    for e = 0, ..., E-1:
        t     ← e / E
        alpha ← alpha_0 * (1 - e / E)

        A_i ← SUM_j  W_ij * g_plus(y_i, y_j)         # scatter-add over sparse W edges

        sample a fresh, shared set of reference indices C (uniform, size M_neg)
        B_i ← MEAN_{c in C}  g_minus(y_i, y_c)        # Monte-Carlo mean-field, all i share C

        F_i ← A_i + m * s_i * B_i
        F_i ← CLIP(F_i)
        y_i ← y_i + alpha * F_i

        store Y_ref(t_{e+1}) checkpoint

    freeze Y_ref(t)   # never modified again


    initialize B_phi

    repeat repulsion-field training:
        sample a reference index i and t ~ Uniform(0, 1)
        y ← Y_ref_i(t) + Gaussian jitter                # off-trajectory training points

        sample a FRESH set of negative reference indices c_1, ..., c_M   (independent of the above)
        B_target ← (1/M) SUM_l  g_minus(y, Y_ref_{c_l}(t))

        minimize  || B_phi(y, t) - stopgrad(B_target) ||^2

    return Y_ref(t), B_phi
```

```text
function QUERY_FLOW(x_star):                      # = the inner loop of EMBED_ONE, Algorithm 1

    ids, mu_star ← RETRIEVE_AND_RERANK / COMPUTE_QUERY_UMAP_WEIGHTS      # Algorithm 2
    y      ← SpectralMap(x_star)                                        # Algorithm 3
    s_star ← SUM_j mu_star[j]

    for e = 0, ..., E-1:
        t     ← e / E
        alpha ← alpha_0 * (1 - e / E)

        y_neighbors ← Y_ref(t)[ids]                    # frozen, read-only lookup

        v_attr ← SUM_j  mu_star[j] * g_plus(y, y_neighbors[j])
        v_rep  ← m * s_star * B_phi(y, t)

        y ← y + alpha * CLIP(v_attr + v_rep)

    return y
```

**A key structural point, stated explicitly because it is easy to elide:** the reference trajectory in the first half of Algorithm 4 is built using a *fresh Monte-Carlo estimate* of the mean negative field at every step (`B_i` above) — it never calls $B_\phi$. $B_\phi$ is trained only *afterward*, against samples drawn from the now-frozen trajectory (plus Gaussian jitter, so it also sees off-trajectory positions), and is used only (a) inside `QUERY_FLOW` at inference, by default, and (b) as an evaluation/diagnostic target (`evaluation/field.py`) and inside the repulsion-oracle diagnostic baseline (Section 13), which replaces $B_\phi$ with a large Monte-Carlo estimate at inference to isolate how much error is attributable to the learned approximation itself, as opposed to the mean-field approximation more generally.

The derivation of $A_i$, $B_i$, $B_X$, and the resulting $F_i$ from UMAP's own negative-sampling scheme is given in Section 6.2. The residual-network architecture used for $B_\phi$ is given in Section 11 and Appendix C; the regression loss is Eq. (6.9), whose form follows the squared vector-field-regression objective of Flow Matching (Lipman, Chen, Ben-Hamu, Nickel, and Le, 2023) applied to a different target field and without that work's generative probability-path construction (Section 8).

## 6. UMAP Force Equations

### 6.1 Low-dimensional kernel and its analytic gradients

UMAP's low-dimensional membership-strength kernel is

$$
\Phi(y,z) = \frac{1}{1 + a\lVert y-z\rVert^{2b}} = \frac{1}{1+aq^b}, \qquad q = \lVert y-z\rVert_2^2,
\tag{6.1}
$$

with $(a,b)$ fit by a nonlinear curve fit against a target curve (McInnes, Healy, and Melville, 2018, Definition 11, p. 20; restated as Eq. (17), Appendix C, p. 57, of arXiv:1802.03426v3). One precise correction to the reference implementation's own convention, verified directly against the paper's text: the archival paper's Definition 11 fits this curve using **`min_dist` only** — the word "spread" does not appear anywhere in that paper as a named parameter. `spread` is a `umap-learn` **software**-only extension of the same fitting procedure (it widens/rescales the target curve before the identical curve fit is applied), not something documented in the arXiv paper itself. This repository's `find_ab_params` reimplements `umap-learn`'s software convention (both `min_dist` and `spread`), verified by direct comparison against the current `umap-learn` source to reproduce it exactly (`tests/test_umap_forces.py::test_find_ab_params_matches_umap_learn`) — not the paper's narrower, `min_dist`-only, published version. The analytic attractive direction is the gradient of $\log \Phi$:

$$
g_+(y,z) = \nabla_y \log \Phi(y,z) = -\frac{2ab\,q^{b-1}}{1+aq^b}(y-z).
\tag{6.2}
$$

The analytic repulsive direction used throughout this codebase is

$$
g_-(y,z) \approx \nabla_y \log\big(1-\Phi(y,z)\big) = \frac{2b}{(q+\varepsilon)(1+aq^b)}(y-z), \qquad \varepsilon = 10^{-3}.
\tag{6.3}
$$

The exact derivative of $\log(1-\Phi)$ has a $1/q$ singularity as $q \to 0$; Eq. (6.3) regularizes the denominator with $\varepsilon$, exactly mirroring the constant used in the current `umap-learn` C/Numba implementation of its own negative-sampling gradient step (verified against the current source; Appendix D, Appendix G). This is stated as an *approximate* equality deliberately: Eq. (6.3) equals the true gradient of $\log(1-\Phi)$ only in the limit $q \gg \varepsilon$; a numerical autograd cross-check (`tests/test_umap_forces.py::test_g_minus_matches_autograd_of_log_one_minus_phi`) is therefore constructed over point pairs with a guaranteed minimum separation, not over unconstrained random pairs, precisely because unconstrained pairs occasionally land close enough together ($q \sim \varepsilon$) that the two would legitimately differ by more than a tight numerical tolerance — this is expected behavior of Eq. (6.3), not a bug (see the code comment at the point where this test was fixed, and Appendix G).

Both Eq. (6.2) and Eq. (6.3) are, in this implementation, clipped component-wise to $[-4,4]$ **before** being summed/averaged over any set of edges or samples — mirroring `umap-learn`'s own per-edge gradient clipping constant (verified against the current source). In the reference-dynamics and query-inference loops (Algorithm 4), the *aggregate* force/velocity (after summing the clipped attractive contribution and the clipped-then-averaged repulsive contribution) is clipped a second time, componentwise, to the same $[-4,4]$ range. This second clip is an implementation choice specific to the mean-field aggregation used here (it has no direct analogue in `umap-learn`'s per-edge SGD, which never sums many already-clipped terms together in this way) and is applied identically in the reference-trajectory construction and in single-query inference (`dynamics.py::simulate_reference_dynamics` and `inference.py::InferenceEngine.embed_one`, respectively).

### 6.2 The mean repulsive field, derived

The point of Algorithm 4 is not to learn the entire UMAP vector field; attraction is never learned. It is to preserve the attractive term analytically and regress only the one term defined as an expectation over the whole reference set.

With row mass $s_i = \sum_j W_{ij}$, the attractive field at reference point $i$, time $t$, is

$$
A_i(t) = \sum_j W_{ij}\, g_+\big(y_i(t), y_j(t)\big),
\tag{6.4}
$$

computed by scatter-add over the sparse edges of $W$ — never as a dense $N\times N$ product.

Define the reference mean negative field as the expectation, over a uniformly sampled reference point $c$, of the repulsive direction toward it:

$$
B_X(y,t) = \mathbb E_{c \sim \mathrm{Uniform}(X)}\big[g_-(y, y_c(t))\big].
\tag{6.5}
$$

**Why this is the right quantity to approximate.** UMAP's actual SGD procedure samples a positive edge and then draws $m$ (the negative-sample rate) random negative reference points per positive sample (McInnes et al., 2018; and see Section 9's discussion of Damrich and Hamprecht (2021) on the precise relationship between this sampling procedure and any single nominal loss function). For reference point $i$ with total incident edge mass $s_i = \sum_j W_{ij}$, the expected number of negative-sampling events touching $i$ per pass is proportional to $s_i$, each contributing, in expectation over the uniformly drawn negative, exactly $B_X(y_i(t), t)$. Linearity of expectation over the (independent) positive-edge and negative-sample draws then gives the reference mean-field update used in this implementation:

$$
F_i(t) = A_i(t) + m\, s_i\, B_X(y_i(t), t).
\tag{6.6}
$$

A step-by-step derivation is given in Appendix B.5. The Monte-Carlo estimator actually computed at every reference-dynamics step is

$$
\widehat B_X(y,t) = \frac1M \sum_{\ell=1}^M g_-\big(y, y_{c_\ell}(t)\big), \qquad c_\ell \sim \mathrm{Uniform}\{1,\ldots,N\},
\tag{6.7}
$$

with a fresh set of $M$ ($=$ `flow.dynamics_negative_samples`, default 64) indices drawn once per step and shared across all $i$ (an implementation choice trading a small amount of correlation across $i$ within one step for an $O(NM)$, rather than $O(N^2)$, cost per step; Appendix B.5 discusses this explicitly).

**What $B_\phi$ actually learns.** Only the population quantity $B_X(y,t)$ of Eq. (6.5) — a function of a 2D (or $d$-dimensional) point and a scalar time, never of degree, never of the negative-sample rate:

$$
B_\phi(y,t) \approx B_X(y,t).
\tag{6.8}
$$

The training target is generated by drawing fresh, *independent* negative samples (distinct from whatever the reference-trajectory construction used at the same $t$) at jittered off-trajectory points, with a stop-gradient on the target:

$$
\mathcal L_{\mathrm{rep}} = \mathbb E\Big[\big\lVert B_\phi(y,t) - \operatorname{stopgrad}\big(\widehat B_X(y,t)\big)\big\rVert_2^2\Big].
\tag{6.9}
$$

Eq. (6.9) borrows the squared vector-field-regression *form* of the Flow Matching training objective — $\mathcal L_{\mathrm{FM}}(\theta) = \mathbb E_{t,p_t(x)}\lVert v_t(x)-u_t(x)\rVert^2$ (Lipman, Chen, Ben-Hamu, Nickel, and Le, 2023, Eq. 5, Section 3, p. 3 of arXiv:2210.02747v2, made tractable via their conditional form, Eq. 9) — a network regressed, by mean-squared error, against a stochastically sampled target vector at a sampled $(y,t)$ pair, with the target itself given a stop-gradient. As the paper's own text puts it, "the FM loss regresses the vector field $u_t$ with a neural network $v_t$" (same page). It does not borrow Flow Matching's generative probability-path construction (their Section 2: probability paths, vector fields, and the flow $\phi_t$ that pushes a base distribution to the data distribution over time) — there is no data-generating stochastic process being matched here; $B_X$ is a deterministic, if analytically-inconvenient, function of $(y,t)$ defined directly by Eq. (6.5).

The multiplicative factor $m \, s_*$ (negative-sample rate times row mass, or its query-side analogue $m\, s_* $ with $s_* = \sum_j \widehat\mu_{*\to j}$) is applied **exactly once**, at the point where the repulsive term is combined with the attractive term (Eq. 6.6 for the reference trajectory, and the corresponding line of `QUERY_FLOW` for a query) — it is never folded into $B_\phi$'s own training target (Eq. 6.9 regresses the bare mean field, with no such factor), and never applied a second time at inference. This is checked in the static review underlying this document (Appendix G, item 24).

### 6.3 Repulsion network architecture (implementation choice, not a mathematically required form)

$B_\phi$ takes $(y, t)$ with $y \in \mathbb R^d$ ($d=2$ by default) and a sinusoidal ("Fourier") embedding of the scalar time $t$,

$$
\gamma(t) = \big[\sin(\omega_1 t), \ldots, \sin(\omega_H t), \cos(\omega_1 t), \ldots, \cos(\omega_H t)\big],
\tag{6.10}
$$

with $H$ = `repulsion.time_embed_dim` / 2 log-spaced frequencies (default `time_embed_dim`=32). An input projection and activation produce $h_0 = \sigma(W_{\mathrm{in}}[y;\gamma(t)] + b_{\mathrm{in}})$, followed by $L$ (default 4) residual blocks

$$
h_{\ell+1} = \sigma\big[h_\ell + W_{\ell,2}\, \sigma(W_{\ell,1} h_\ell + b_{\ell,1}) + b_{\ell,2}\big],
\tag{6.11}
$$

and an output projection, zero-initialized at construction (so the untrained network starts at a benign all-zero repulsion prior):

$$
B_\phi(y,t) = W_{\mathrm{out}} h_L + b_{\mathrm{out}}.
\tag{6.12}
$$

Defaults: hidden width 256, 4 residual blocks, GELU activation throughout, time embedding dimension 32, output dimension $d$. This architecture (residual MLP + Fourier time features) is an implementation choice; only the *regression objective* it is trained with (Eq. 6.9) is attributed to prior work (Section 8).

## 7. Exact and Approximate Correspondence to UMAP

This section states, as precisely as possible, which parts of the pipeline are exact identities, which hold conditionally on a checkable event, which are approximations with a defined target, and which are not equivalent to anything UMAP computes.

### 7.1 Exact algebraic identities

**(a) Fuzzy union.** $W = \mathrm{Mu} + \mathrm{Mu}^\top - \mathrm{Mu}\odot\mathrm{Mu}^\top$ is implemented (`graph.py::symmetrize_fuzzy_union`) as exactly the closed-form combination rule given by McInnes, Healy, and Melville (2018) for symmetrizing directed local fuzzy simplicial sets via the probabilistic t-conorm — stated as a matrix identity, $B=A+A^\top-A\odot A^\top$, in Section 3.1 (pp. 15–16), and restated in scalar form as Equation (16), Appendix C (p. 57), of arXiv:1802.03426v3 — at the default mixing ratio (`set_op_mix_ratio=1.0`; the paper's Eq. (16) is exactly this default case, not the general mixed form `umap-learn`'s software additionally supports). Verified directly against the current `umap-learn` source (Appendix D, Appendix G).

**(b) Low-dimensional kernel.** $\Phi(y,z) = (1+a\lVert y-z\rVert^{2b})^{-1}$ (Eq. 6.1) is Definition 11 (p. 20), restated as Equation (17), Appendix C (p. 57), of McInnes, Healy, and Melville (2018). The curve fit producing $(a,b)$ is implemented as a line-for-line reimplementation of `umap-learn`'s own *software* procedure (not an import of it, to avoid a private-API dependency; verified numerically identical by direct comparison — `tests/test_umap_forces.py::test_find_ab_params_matches_umap_learn`) — which fits against a target curve parameterized by both `min_dist` and `spread`, the latter absent from the archival paper's own Definition 11 (Section 6.1 states this distinction precisely).

**(c) Analytic attraction.** Eq. (6.2) is the exact derivative of $\log\Phi$ with respect to $y$; this is a closed-form fact, not an approximation, and is never learned.

### 7.2 Conditional equality

If $K_X(x_*) \subseteq C_*$ (Eq. 5.4, the candidate-recall event), then reranking reproduces the true query-side fixed-reference kNN exactly, and therefore $\widehat\mu_{*\to j} = \mu_{*\to j}$ under this implementation's no-self-column smooth-kNN convention (Eq. 5.5–5.6, proved in Appendix B.6). This equality is conditional on an event whose *probability* is exactly what the retriever is trained to make high (Section 5.2); it is not an unconditional guarantee, and its failure mode is a genuinely different (not merely "noisier") candidate set, not a perturbation of the formula.

### 7.3 Spectral equivalence, as a subspace target under idealized assumptions

Under the idealized assumptions stated in Section 5.3 (sufficient capacity, exact optimization, exact orthogonality, non-degenerate spectral gap), $\operatorname{span}(Z) = \operatorname{span}(U_{1:d+1})$ (Eq. 5.13) — a *subspace* statement about the trained network's outputs over the reference set, evaluated empirically (never exactly, and never as a training signal) via principal angles against an exact sparse eigensolver (`evaluation/spectral.py`). This is emphatically not a claim that a query's spectral coordinate equals what a transductive spectral embedding of $X \cup \{x_*\}$ would assign it (Eq. 5.14 is explicitly not claimed) — the graph itself would differ.

### 7.4 Expected-update equivalence

At a fixed state $\{y_i(t)\}$, the mean-field reference update (Eq. 6.6) is the expectation, over UMAP's own independent positive-edge and negative-sample draws, of the net displacement UMAP's stochastic SGD would apply to point $i$ at that state (Appendix B.5). This is a statement about *expectations*, holding by linearity of expectation over the sampling scheme — it is not a claim that any single stochastic UMAP trajectory coincides with the deterministic mean-field trajectory computed here.

### 7.5 What is not equivalent

We do not claim

$$
F_{\Theta,\mathcal M_X}(x_*) = \operatorname{UMAP}(X \cup \{x_*\})_*.
\tag{7.1}
$$

Reasons: (1) inserting $x_*$ can change reverse-kNN relationships among reference points, so some reference points' own $K_i$ (and hence rows of $\mathrm{Mu}$ and $W$) could change; (2) new edges between $x_*$ and reference points would need to enter $W$, changing its degree sequence; (3) the normalized Laplacian $L_{\mathrm{sym}}$ would change; (4) its eigenvectors would generally change; (5) if the *reference points themselves* were re-optimized jointly with $x_*$ (as a transductive rerun would do), they would move; UMAPing instead freezes the reference trajectory once, offline, and never updates it at query time; (6) even holding the reference trajectory fixed, the mean-field Euler dynamics of Algorithm 4 are a deterministic expectation, not a replay of any one stochastic asynchronous SGD realization UMAP's own optimizer would produce.

**Table 7.1 — component-by-component relation to UMAP.**

| Component | Relation to UMAP |
|---|---|
| Directed $\mathrm{Mu}$ / symmetric $W$ on $X$ | exact (Section 7.1a) |
| Query-side $\widehat\mu_{*\to j}$, if candidate recall is complete | conditionally exact (Section 7.2) |
| Spectral subspace on the reference graph | same Rayleigh–Ritz target; realized by a trained, evaluated (not exactly verified) network (Section 7.3) |
| Analytic attraction $g_+$ | exact, closed-form, at any given state (Section 7.1c) |
| Reference mean negative-sampling field $F_i$ | conditional expectation over UMAP's own sampling scheme (Section 7.4) |
| $B_\phi$ | learned approximation of the population quantity $B_X$ (Section 6.2, 6.3) |
| Reference mean trajectory $Y_{\mathrm{ref}}(t)$ | deterministic mean-field approximation of stochastic UMAP SGD, frozen after construction |
| A transductive rerun on $X \cup \{x_*\}$ | not equivalent (Section 7.5) |

## 8. Model / Loss Provenance

All equation numbers and quotations below were independently verified against the current text (specific arXiv version, or published proceedings version, noted in the References) of each source paper before being included here; none is stated from memory alone. Rows marked "implementation choice" carry no literature attribution and should not be read as attributed to any cited work.

| # | Component | Our implementation | Literature origin | Exact borrowed part | Modification made here |
|---|---|---|---|---|---|
| 1 | UMAP fuzzy graph | `graph.py` | McInnes, Healy, and Melville (2018), arXiv:1802.03426v3 | directed smooth-kNN membership (p. 14, unnumbered; restated as Eq. 15, Appendix C); symmetric fuzzy union (Section 3.1, pp. 15–16, unnumbered; restated as Eq. 16, Appendix C) | none in the underlying formulas (reimplemented, verified numerically identical); this document's $K_i$ excludes self by construction of the neighbor search itself, unlike `umap-learn`'s own internal function, which assumes and skips a self-distance column (Section 5.2, "Smooth-kNN construction, cited precisely"; RUNTIME_CHECKS.md) |
| 2 | Retriever score + loss | `models/retriever.py`, `training/retriever.py` | Karpukhin et al. (2020), DPR, EMNLP 2020 | dual-encoder dot-product score (their Eq. 1); in-batch-negative NLL loss (their Eq. 2) | positive sampled $\propto \mathrm{Mu}$ row (Eq. 5.2, this document); **L2-normalization and temperature $\tau$ are additions, confirmed absent from DPR's own Eq. (1)–(2)** (DPR uses a raw, unnormalized, temperature-free dot product — Section 5.2); known-neighbor negative masking is also this implementation's addition |
| 3 | Spectral network training | `models/spectral.py`, `training/spectral.py` | Shaham et al. (2018), SpectralNet, arXiv:1801.01587v6 | the *normalized*, degree-scaled pairwise spectral loss structure (their Eq. 5, not their unnormalized base loss, Eq. 3), applied here to the UMAP graph's normalized Laplacian (Eq. 5.7, this document) | **orthogonality enforced as an additive soft penalty (Eq. 5.9, this document) — explicitly not SpectralNet's own mechanism**, which is a hard architectural constraint (a QR/Cholesky-based orthogonalization output layer, their Eq. 4 and Section 3.1); SpectralNet's own paper attributes a soft-penalty style to a different prior work (Yi et al., 2016), not to itself (Section 5.3) |
| 4 | Spectral mode/eigenvector separation | `training/spectral.py::compute_calibration` | Ben-Ari, Yacobi, and Shaham (2025), Generalizable Spectral Embedding, arXiv:2501.11305v2 | the eigenvector-separation idea: form a small matrix from a trained, rotation-ambiguous output basis and the target operator, then diagonalize it to recover a disambiguated basis (their Lemma 1 and Section 4.2) | applied to $L_{\mathrm{sym}}$ built from the UMAP graph specifically; not a reimplementation of their own named method (Sep-SpectralNet/NUMAP, which additionally trains a second contrastive network); "projected operator" (Eq. 5.11) is this document's own term, not theirs (Section 5.3) |
| 5 | UMAP low-dim forces | `umap_forces.py` | McInnes, Healy, and Melville (2018), arXiv:1802.03426v3 | kernel $\Phi$ (Definition 11, p. 20; Eq. 17, Appendix C); analytic $g_+$ (exact derivative); approximate $g_-$ (Section 6.1) | the $(a,b)$-fitting *software* procedure additionally uses `spread`, which does not appear in the archival paper's own Definition 11 (Section 6.1, Section 7.1b) — reimplemented from `umap-learn`'s software, not the paper alone; otherwise none (verified numerically identical) |
| 6 | Mean-field repulsion derivation | `dynamics.py`, `training/flow.py` | derived in this work from UMAP's own negative-sampling scheme (McInnes, Healy, and Melville, 2018, Section 4.1/4.2, citing Mikolov et al., 2013 for the negative-sampling technique itself); see Damrich and Hamprecht (2021) regarding the precise relationship between that sampling scheme and any single nominal loss function | — | — |
| 7 | Vector-field regression objective | `training/flow.py::train_repulsion_field` | Lipman, Chen, Ben-Hamu, Nickel, and Le (2023), Flow Matching, arXiv:2210.02747v2 | the squared-error vector-field regression form of their Eq. 5, $\mathcal L_{\mathrm{FM}}(\theta)=\mathbb E\lVert v_t(x)-u_t(x)\rVert^2$ (made tractable in their paper as the conditional form, their Eq. 9) | regresses a different (deterministic, population-mean) target field, $B_X$, not a conditional-probability-path velocity field; the probability-path/flow construction of their Section 2 is not used here at all (Section 6.2) |
| 8 | Residual Fourier-time MLP | `models/repulsion.py` | implementation choice | — | not attributed to any cited work |

## 9. Limitations of Prior Work and What UMAPing Changes

Every quotation below was independently verified directly against the cited paper's own PDF text (not a summary or memory) before inclusion; exact section/page locations and the arXiv/proceedings version inspected are given alongside each one.

**Generalizable Spectral Embedding with an Application to UMAP** (Ben-Ari, Yacobi, and Shaham, 2025; TMLR 07/2025; arXiv:2501.11305v2). Ben-Ari et al. prove that a SpectralNet-style network trained to minimize a Rayleigh-quotient objective under an orthogonality constraint cannot, by itself, produce a disambiguated spectral embedding: "every minimizer of $R_L$ under the orthogonality constraint[] is of the form $VQ$, where $V$ is the first $k$ eigenvectors matrix of $L$ and $Q$ is an arbitrary squar[]ed orthogonal matrix" (Lemma 1, Section 4.1, p. 6), and, applied specifically to SpectralNet: "SpectralNet's method, using a deep neural network for RQ minimization (while enforcing orthogonality), does not lead to the SE. However, it only leads to the space spanned by the constant vector and the leading $k-1$ eigenvectors of $L$, with different rotations and reflections for each run" (Section 4.1, p. 6). They also document, with direct quotations, the out-of-sample and scalability difficulties of prior spectral out-of-sample methods: "their out-of-sample extension is far from trivial. Usually, it is done by out-of-sample extension (OOSE) methods such as Nyström ... or Geometric Harmonics ... However, these methods provide only local extension (i.e., near existing training points), and are both computationally and memory restrictive, as they rely on computing the distances between every new test point and all training points" (Section 2, pp. 2–3); of a different prior generalizable-spectral-embedding method: "the training procedure of the network is computationally expensive, therefore restricting its usage for large datasets" (Section 2, p. 3). Algorithm 3 addresses the out-of-sample aspect the same way their own method does — a fixed, frozen, linear read-out of a trained network's output, $x_* \to h_\omega(x_*)\,T\,P$ (Eq. 5.12), requiring no further graph access — using the eigenvector-separation mechanism described precisely in Section 5.3 and Section 8 (row 4); it is not a reimplementation of their specific named method (Sep-SpectralNet/NUMAP), which additionally trains a second network against UMAP's own contrastive loss on top of the disambiguated spectral output.

**On UMAP's True Loss Function** (Damrich and Hamprecht, NeurIPS 2021; arXiv:2103.14608v2). Damrich and Hamprecht show, in closed form, that "[UMAP's e]ffective loss function ... differs from the published one" (Abstract, p. 1), because "UMAP's use of negative sampling ... the resulting effective loss function differs significantly from UMAP's purported loss function" (Section 1, p. 1). This is exactly why Section 6.2 derives the reference mean-field update (Eq. 6.6) directly from UMAP's actual negative-sampling *procedure* — an expectation over that sampling scheme (Appendix B.5) — rather than from any nominal published loss function. This document makes no independent claim about what UMAP's true effective loss is; it follows Damrich and Hamprecht's finding only as the reason to derive Eq. (6.6) from the sampling procedure rather than from a nominal loss.

**On Out-of-sample Embedding in UMAP** (Islam and Fleischer, 2026; arXiv:2606.04451v1). Islam and Fleischer document a specific, named failure mode of out-of-sample UMAP embedding: "UMAP has trouble adding out-of-sample points to a pre-existing mapping. In particular, UMAP often places new points on the periphery of the found clusters, rather than in their interiors with their correlated neighbors" (Abstract, p. 1), which they term the "repulsion effect": "[w]hen most of the repulsive forces are in the same direction ... (e.g., near a boundary), the resultant repulsive force pushes the point towards the periphery of the original embedding ... We dub this phenomenon the 'repulsion effect'" (Section 2.3, p. 5). **We state explicitly: UMAPing does not theoretically resolve this effect.** $B_\phi$ is trained to approximate $B_X$ (Eq. 6.5, 6.8) faithfully; if the true mean field $B_X$ itself carries the directional bias Islam and Fleischer describe near a cluster boundary, a well-trained $B_\phi \approx B_X$ will reproduce that same bias, not correct it. Their own solution is different in kind — they "overcome this 'repulsion effect' by optimizing pairwise interactions within the original k-nearest-neighbor graph" (Abstract, p. 1), i.e. by changing the optimization itself — whereas this document's contribution (Section 10) is restricted to an explicit decomposition of the inference computation and diagnostics/ablations (Section 13) that let such an effect be measured and attributed to a specific component, not a mechanism that corrects it.

**Graph Flow Matching** (Siddiqui, Eliasof, and Haber, 2026; AAAI-26; also arXiv:2505.24434v3). This paper's own framing must be stated precisely, since the natural first guess about it (independent, neighbor-ignoring point-wise velocity prediction) turns out to describe only what it identifies as a limitation of *prior* work, not its own method: "[e]xisting flow matching networks typically predict each point's velocity independently, considering only its location and time along its flow trajectory, and ignoring neighboring points" (Abstract) — a limitation their own method, Graph Flow Matching, is designed to fix by decomposing the learned velocity into "a reaction term — any standard flow matching network — and a diffusion term that aggregates neighbor information via a graph neural module" (Abstract), i.e. $v_\theta(x,t)=v_{\mathrm{react}}(x,t)+v_{\mathrm{diff}}(x,t;\mathcal N(x,t))$ (their Eq. 7, Section 3.1). **Graph Flow Matching's own contribution is therefore explicitly neighbor-aware, not independent/pointwise — the opposite of what a superficial reading of its name might suggest, and this document does not claim otherwise.** The accurate comparison is with the *prior, non-graph* flow-matching networks their paper itself criticizes: $B_\phi(y,t)$ (Eq. 6.8–6.9) is exactly such a pointwise-only network, matching their own description of the "reaction term" $v_{\mathrm{react}}$ alone ("[t]his component models pointwise transport based on global training dynamics," their Section 3.1). UMAPing does not ask $B_\phi$ to also learn a graph/neighbor-conditioned correction analogous to their diffusion term; instead, the neighbor-aware role is filled entirely by the explicit, analytic attraction term (Eq. 9.1 below), which by construction depends on the retrieved neighbors' positions and is never delegated to any learned network:

$$
v_{\mathrm{attr}} = \sum_j \widehat\mu_{*\to j}\, g_+(y_*, y_j(t)).
\tag{9.1}
$$

**Parametric UMAP** (Sainburg, McInnes, and Gentner; v1 title "Parametric UMAP: Learning Embeddings with Deep Neural Networks for Representation and Semi-Supervised Learning," 2020; retitled in later revisions; arXiv:2009.12981). We do not present this as a limitation to be overcome; it already solves the same *problem* (a learned, inductive UMAP-style embedding) with a different design. In the authors' own words, "Parametric UMAP can be defined simply by applying the UMAP cost function to a deep neural network over mini batches using negative sampling" (v1, Section 1.2, pp. 4–5) — one neural network trained end to end, with UMAP's own attractive/repulsive negative-sampling loss, to reproduce a UMAP-consistent embedding as a single learned function $x \to y$. Ben-Ari et al. (2025) note a specific consequence of this design relevant to Section 5.3 above: "P. UMAP replaces step (3) with the training of a neural network. Importantly, P. UMAP overlooks step (2), the SE [spectral embedding] initialization. Consequently, P. UMAP may struggle to preserve global structure" (Ben-Ari et al., 2025, Section 3.3, p. 5) — a claim we report as theirs, not our own independent finding, since no comparison against Parametric UMAP is run in this document (Section 15, Appendix H). The methodological difference from UMAPing is the factorization: $x_* \to \text{neighbor identities} \to \mu \to \text{analytic forces}$, plus a separately-trained spectral initialization and a separately-distilled repulsion field, rather than one function approximating the whole map. Section 10 discusses what this factorization enables (component-level error attribution, exact post-retrieval weights, trajectory-level interpretability) as capabilities, not as an argument that the factorized design is more accurate — no such comparison is made in this document.

**SpectralNet** (Shaham, Stanton, Li, Nadler, Basri, and Kluger, 2018; ICLR 2018; arXiv:1801.01587v6). SpectralNet is used here as a foundation for Algorithm 3's training objective, not as a target of criticism. In the authors' own words, "SpectralNet ... learns a map that embeds input data points into the eigenspace of their associated graph Laplacian matrix ... [and] the map learned by SpectralNet naturally generalizes the spectral embedding to unseen data points" (Abstract, p. 1). The normalized pairwise loss structure it introduced (their Eq. 5) is reused (Section 8, row 3), applied here to a different (UMAP-specific) graph Laplacian, with a different orthogonality mechanism: SpectralNet's own is architectural, "implemented using a special-purpose output layer" (Abstract, p. 1) via QR/Cholesky decomposition (Section 3.1, p. 4), not the additive soft penalty used in this implementation (Eq. 5.9; Section 5.3, Section 8 row 3).

## 10. Additional Capabilities and Novelty Candidates

These are described as capabilities enabled by the decomposition, not as proven advantages over any specific prior method, and none of them is claimed to be presented here for the first time.

**A. Query order/set independence.** Because $F_{\Theta,\mathcal M_X}(x_*)$ reads only $x_*$ and the frozen memory $\mathcal M_X$, for any other query $x_b$, $F_{\Theta,\mathcal M_X}(x_a)$ does not depend on whether, or when, $x_b$ is also evaluated. This is checked directly by `tests/test_inference.py::test_embed_one_is_independent_of_other_queries` and, for the stochastic repulsion-oracle-diagnostic mode, `test_embed_one_oracle_mc_repulsion_is_reproducible_and_query_independent`.

**B. Component-level error localization.** End-to-end error can be decomposed by substituting one component at a time, using the baselines of Section 13: oracle (exact) neighbor retrieval vs. the learned retriever; the learned $B_\phi$ vs. a large Monte-Carlo estimate of $B_X$ at inference (the repulsion-oracle diagnostic); the spectral initialization alone (no flow integration) vs. the full method; a no-repulsion ablation. Failure can therefore in principle be attributed to a specific stage (retrieval recall, spectral generalization, the mean-field approximation itself, or the repulsion network's fidelity to it) rather than observed only as one opaque map's aggregate error.

**C. Exact edge weights after retrieval.** The retriever predicts candidate *identities* only; edge weights are always recomputed analytically from exact distances (Section 5.2, Eq. 5.4–5.6). The design is "learned retrieval + analytic UMAP membership," not "a network predicts $W_{*j}$ directly."

**D. Trajectory-level and per-neighbor decomposition.** At every integration step, $v_{\mathrm{attr}}$, $v_{\mathrm{rep}}$, and (since $v_{\mathrm{attr}}$ is a sum over retrieved neighbors) each per-neighbor contribution $v_j^{\mathrm{attr}} = \widehat\mu_{*\to j}\, g_+(y_*, y_j(t))$ can be logged. This permits examining, after the fact, which retrieved neighbors and which force component were responsible for a given step's displacement. We do not call this causal interpretability — it is a decomposition of a deterministic computation the method already performs, not an inference about any causal structure in the underlying data-generating process.

**E. Edge-weight sensitivity.** Holding the reference trajectory fixed, one can alter a specific $\widehat\mu_{*\to j}$ and re-integrate `QUERY_FLOW` for the same $x_*$, observing the resulting change in $y_*$. This is an algorithmic sensitivity/counterfactual analysis of the inference procedure itself, not a causal claim about the underlying (e.g. biological or image) system that generated $X$.

**F. Deterministic mean repulsion at inference.** In the default ("learned") mode, `QUERY_FLOW` never draws fresh random negative samples at inference — it evaluates $B_\phi$, a fixed function. This removes inference-time Monte-Carlo negative sampling from the learned inference path (contrast with the repulsion-oracle diagnostic, which reintroduces it deliberately, for comparison). We phrase this as removing a source of run-to-run sampling variance in the default path, not as a guarantee of better embedding quality — no such comparison is reported in Section 15.

## 11. Model Architecture Details

All values below are the current defaults in `configs/coil20.yaml`, `configs/coil100.yaml`, and `configs/pancreas.yaml` (the three configs share identical model/training hyperparameters; only `dataset` and, for pancreas, log frequency differ). **These are experimental defaults, not mathematically required values**; Appendix H lists sensitivity analyses that have not yet been run.

**Retriever** (`models/retriever.py`, Section 5.2): input dimension = reference-only PCA output (256 for COIL-20/100; 50 for pancreas, over 2,000 reference-only-selected highly variable genes); two independently-parameterized towers, each an MLP `input → 512 → 256 → 128` with GELU activations between layers; L2-normalized outputs; temperature $\tau=0.1$; candidate pool size $M = \max(4k, 64)$; in-batch batch size 256 plus 64 additional random negatives; 2,000 training steps, Adam, lr $10^{-3}$.

**Spectral encoder** (`models/spectral.py`, Section 5.3): MLP `input → 512 → 256 → 3` (raw output dimension $r = d+1 = 3$ for the default $d=2$), GELU; edge minibatch size 4,096; orthogonality weight 1.0; 2,000 training steps, Adam, lr $10^{-3}$; post-hoc calibration target scale 10.0.

**Reference mean flow** (`dynamics.py`, `training/flow.py`, Section 5.4): 200 Euler steps; $\alpha_0=1$, linearly decayed to 0; `negative_sample_rate` $m=5$; 64 shared negative samples per step for the reference-trajectory construction; force/velocity clip $\pm 4$ (Section 6.1); every step checkpointed by default (`checkpoint_stride=1`).

**Repulsion network** (`models/repulsion.py`, Section 6.3): input $y \in \mathbb R^2$ plus a 32-dimensional Fourier time embedding; hidden width 256; 4 residual blocks; GELU; output dimension 2; output layer zero-initialized; 64 fresh negative samples per teacher example; jitter std 0.1; 4,000 training steps, batch size 512, Adam, lr $10^{-3}$.

Full per-layer tables are given in Appendix C.

## 12. Experimental Datasets

**COIL-20** (Nayar, Nene, and Murase, 1996; Columbia CAVE processed release): 1,440 grayscale $128\times128$ images, 20 objects, 72 rotational poses each. Poses are ranked by rotational order per object, and every 4th pose (by rank) is held out as the query set — 54 reference / 18 query poses per object (75%/25%), with every object class present on both sides by construction. Pixels are scaled to $[0,1]$, flattened, and projected by a PCA fit on the reference split only (default 256 components; raw flattened pixels are a config option).

**COIL-100**: 7,200 RGB $128\times128$ images, 100 objects, 72 poses each; identical pose-rank-based split and identical reference-only-PCA preprocessing convention as COIL-20 (default 256 components).

**Pancreas scRNA-seq** (the pancreas reference-mapping dataset used in the scvi-tools/scArches/scIB tutorials): raw counts, `obs['tech']`, `obs['celltype']`. Query technologies: `smartseq2`, `celseq2`; reference: every other technology present in the file. Preprocessing: `normalize_total(1e4)` then `log1p` (both purely per-cell, hence leakage-free regardless of fitting order); 2,000 highly variable genes selected on the reference split only (`batch_key="tech"`); a 50-component PCA fit on the reference split only and applied unchanged to the query split.

In all three datasets, no query information is used in HVG selection, PCA fitting, the reference kNN graph, retriever training, spectral training, the reference mean-field trajectory, or repulsion-field training — every one of these is fit on the reference split alone (checked directly by `tests/test_data_splits.py`, including a test that perturbs the query cells' raw expression data and confirms the reference-fitted HVG selection and PCA transform are completely unaffected).

Exact preprocessing steps, and the observed reference/query counts from an actual run, are given in Appendix E.

## 13. Baselines

1. **Standard `umap-learn`**: `umap.UMAP(...).fit(X_{\mathrm{reference}})`, then `.transform(X_{\mathrm{query}})` — the reference package's own inductive path, used as-is (`baselines.py::run_standard_umap`).
2. **Spectral-only**: $y_* = S_\omega(x_*)$ (Algorithm 3 alone; no flow integration).
3. **Ours + oracle neighbors**: `RETRIEVE_AND_RERANK` is replaced by an exact brute-force top-$k$ search against the full reference set (`neighbor_source="oracle"`); everything downstream (weights, spectral initialization, forces) is otherwise identical to the full method.
4. **Full method**: learned retriever + analytic attraction + learned repulsion ($B_\phi$), as in Algorithm 1.
5. **No-repulsion ablation**: $v_{\mathrm{rep}} \equiv 0$; analytic attraction only.
6. **Repulsion-oracle diagnostic**: $B_\phi$ is replaced, at inference, by a large Monte-Carlo estimate $\widehat B_X$ (Eq. 6.7) computed on the fly against the frozen reference trajectory, on a configurable subset of queries (`eval.repulsion_oracle_query_subset`, default 50) — used to isolate the error contributed by the learned approximation of $B_X$ specifically.
7. **(Optional) Transductive UMAP on $X \cup X_{\mathrm{query}}$** (`baselines.py::run_transductive_umap`): fits `umap-learn` jointly on reference and query. This is explicitly *not* the inductive target and is included, when run, only as a qualitative visualization reference point — it uses query data during graph construction, which every other method here never does.

## 14. Metrics

**Retrieval** (`evaluation/retrieval.py`): $\mathrm{Recall@}k = |\widehat K_* \cap K_*| / k$; candidate $\mathrm{Recall@}M$ (analogous, against the pre-rerank candidate pool); fuzzy-weighted recall,

$$
\mathrm{WeightedRecall} = \frac{\sum_{j \in \widehat K_* \cap K_*} \mu^{\mathrm{true}}_{*\to j}}{\sum_{j \in K_*} \mu^{\mathrm{true}}_{*\to j}};
\tag{14.1}
$$

$\mathrm{NDCG@}k$ using $\mu^{\mathrm{true}}_{*\to j}$ as graded relevance; mean per-query retrieval latency.

**Spectral** (`evaluation/spectral.py`): graph Rayleigh energy $\operatorname{Tr}(Z^\top L_{\mathrm{sym}} Z)$; orthogonality error $\lVert Z^\top Z/N - I\rVert_F$; the eigenvalues of the projected operator $H$; the eigenvalues of an exact sparse solve of $L_{\mathrm{sym}}$; and the principal angles between the two subspaces (`scipy.linalg.subspace_angles`), for evaluation only.

**Repulsion field** (`evaluation/field.py`): vector MSE and RMSE of $B_\phi(y,t)$ against a held-out, independently-sampled Monte-Carlo teacher; mean cosine similarity; mean magnitude error.

**Final embedding** (`evaluation/embedding.py`). The primary metric respects the fixed-reference, pointwise setting: for each query, its true high-dimensional $k$-NN in $X$,

$$
K^{\mathrm{high}}_* = k\mathrm{NN}(x_*, X),
\tag{14.2}
$$

is compared against its $k$-NN in the final embedding, searched only against the reference embedding $Y_X$,

$$
K^{\mathrm{low}}_* = k\mathrm{NN}(y_*, Y_X), \qquad
\mathrm{Q2RRecall@}k = \frac{|K^{\mathrm{high}}_* \cap K^{\mathrm{low}}_*|}{k}.
\tag{14.3}
$$

Also reported: reference-only trustworthiness (computed on the reference embedding alone, deliberately not extended to query points, since doing so would require ranking query-query distances, which the inductive method itself never computes); a $k$-NN label-classification accuracy fit on the reference embedding and evaluated on the query embedding (object ID for COIL, cell type for pancreas); and query latency. Any coordinate-level comparison between two independently-fit 2D embeddings (e.g., against the standard-`umap-learn` baseline) is Procrustes-aligned first (`evaluation/embedding.py::procrustes_align`) — raw coordinates from independently-fit, arbitrarily-rotated embeddings are never compared directly.

## 15. Results

No scientific (accuracy/quality) results are reported here. At the time of writing: the unit test suite (35 tests; Appendix G) passes; the full pipeline (preprocessing through analysis/figure generation) has been run to completion, end to end, on real hardware for COIL-20 and COIL-100; a caching bug specific to the pancreas dataset's string-valued labels was found on an actual pancreas run, fixed, and independently reproduced/confirmed fixed locally, but a full, successful, end-to-end pancreas run has not yet been independently confirmed in the record this document is based on. **These are engineering/correctness signals, not scientific results** — "35/35 unit tests passed" and "the pipeline completes without error" are claims about the software running correctly, not claims about embedding quality, and must not be conflated with the latter.

| Dataset | Method | Q2R Recall@$k$ | Label Acc. | Retrieval Recall@$k$ | Weighted Recall | Field Cosine Sim. | Latency |
|---|---|---|---|---|---|---|---|
| COIL-20 | standard umap-learn | TBD | TBD | — | — | — | TBD |
| COIL-20 | spectral-only | TBD | TBD | — | — | — | TBD |
| COIL-20 | oracle neighbors | TBD | TBD | TBD | TBD | TBD | TBD |
| COIL-20 | full method | TBD | TBD | TBD | TBD | TBD | TBD |
| COIL-20 | no-repulsion | TBD | TBD | TBD | TBD | TBD | TBD |
| COIL-20 | repulsion-oracle diagnostic | TBD | TBD | TBD | TBD | TBD | TBD |
| COIL-100 | (same rows) | TBD | TBD | TBD | TBD | TBD | TBD |
| Pancreas | (same rows) | TBD | TBD | TBD | TBD | TBD | TBD |

## 16. Discussion

1. This is a fixed-reference inductive method: $X$ is fixed at training time, and only single, independent query points are handled at inference.
2. It is not exact $\operatorname{UMAP}(X \cup \{x_*\})$ (Section 7.5); no claim to the contrary is made anywhere in this document.
3. The reference graph and the reference trajectory are frozen after offline construction and are never modified at query time.
4. The reference mean dynamics are a deterministic expectation, not a replay of any specific stochastic asynchronous UMAP SGD path (Section 7.4).
5. The retriever still requires a frozen representation/index of $X$ (the stored key embeddings); nothing here removes the need to keep $X$'s features and derived index in memory.
6. The default candidate-search backend (`models/retriever.py::ExactChunkedIndex`) is a deterministic, chunked *exact* inner-product search — $O(N)$ per query in the current default backend. An approximate (`pynndescent`-backed) backend is implemented behind the same interface but is optional and untested in the default configuration (Appendix G).
7. Spectral generalization to strongly out-of-distribution queries (relative to the reference set) is not guaranteed by anything in Section 5.3's derivation, which assumes idealized training conditions.
8. Because the mean-field integration is a discrete Euler recursion, error in the learned repulsion approximation can accumulate across steps; this has not yet been measured (Appendix H).
9. Storing every reference-trajectory checkpoint costs $O(E \cdot N \cdot d)$ memory (a `checkpoint_stride` config option exists to store a subset instead, with interpolation used to reconstruct intermediate positions).
10. Input dimensionality is fixed per dataset/model instance; the method does not support variable-dimensional inputs.
11. PCA (and, for pancreas, HVG selection) is part of the experimental preprocessing pipeline used in this repository's three datasets, not part of the UMAPing method itself, which is agnostic to how its fixed-dimensional input features are produced.
12. The method does not theoretically resolve any systematic out-of-sample repulsion bias that may exist in UMAP-style embeddings generally (Section 9); if the true expected field $B_X$ carries such a bias, a faithful $B_\phi \approx B_X$ reproduces it rather than correcting it.

## 17. Conclusion

We have described, in full mathematical and algorithmic detail, a fixed-reference inductive UMAP-style embedding method that amortizes specific non-parametric components of UMAP — candidate neighbor identification, spectral initialization, and the mean repulsive field — with small learned models, while leaving UMAP's fuzzy-graph construction and attractive-force gradient exactly analytic wherever a state is given. We have stated explicitly which properties of the resulting map are exact, which are conditional on a checkable retrieval-recall event, which are mean-field approximations of a well-defined population quantity, and which do not hold at all (in particular, this is not equivalent to a transductive rerun of UMAP on $X \cup \{x_*\}$). No scientific accuracy results are reported; Section 15 and Appendix H lay out what remains to produce them.

---

## Appendix A — Full Algorithms

Algorithms 1–4 are given in full in Sections 5.1–5.4; they are not abbreviated in the main text, so they are not reprinted here. Cross-references: Algorithm 1 (Section 5.1), Algorithm 2 (Section 5.2), Algorithm 3 (Section 5.3), Algorithm 4 (Section 5.4).

## Appendix B — Full Mathematical Derivations

### B.1 Fuzzy union is a probabilistic OR

For $p,q\in[0,1]$ interpreted as independent membership probabilities, the probability that at least one of two independent events holds is $p+q-pq$ (inclusion–exclusion for independent events). Applying this elementwise to $\mu_{i\to j}$ and $\mu_{j\to i}$ gives $W_{ij}=\mu_{i\to j}+\mu_{j\to i}-\mu_{i\to j}\mu_{j\to i}$, i.e. $W=\mathrm{Mu}+\mathrm{Mu}^\top-\mathrm{Mu}\odot\mathrm{Mu}^\top$ (Eq. in Section 4.1). No independence assumption about the *underlying data* is implied or needed for this to be the definition adopted; it is used here purely as the closed-form symmetrization rule, exactly as `umap-learn` implements it at `set_op_mix_ratio=1.0` (Section 8, Appendix D).

### B.2 The Dirichlet-energy / trace identity (Eq. 5.8)

Let $Z \in \mathbb R^{N\times r}$ have rows $z_i$, and let $D=\operatorname{diag}(D_{11},\ldots,D_{NN})$, $D_{ii}=\sum_jW_{ij}$. Expand the right-hand side of Eq. (5.8):

$$
\frac12\sum_{ij}W_{ij}\Big\lVert\frac{z_i}{\sqrt{D_{ii}}}-\frac{z_j}{\sqrt{D_{jj}}}\Big\rVert^2
=\frac12\sum_{ij}W_{ij}\Big[\frac{z_i^\top z_i}{D_{ii}}-\frac{2z_i^\top z_j}{\sqrt{D_{ii}D_{jj}}}+\frac{z_j^\top z_j}{D_{jj}}\Big].
$$

Using $\sum_j W_{ij}=D_{ii}$, the first bracketed term summed over $j$ gives $\sum_i z_i^\top z_i=\operatorname{Tr}(Z^\top Z)$; by symmetry the third term (summed over $i$ for fixed $j$, then over $j$) gives the same. So the first and third terms together contribute $\operatorname{Tr}(Z^\top Z)$ (after the leading $\tfrac12$ and the factor-of-two from adding two equal contributions cancel). The middle term is $\sum_{ij}\frac{W_{ij}}{\sqrt{D_{ii}D_{jj}}}z_i^\top z_j=\operatorname{Tr}\big(Z^\top D^{-1/2}WD^{-1/2}Z\big)$. Hence the whole expression equals

$$
\operatorname{Tr}(Z^\top Z)-\operatorname{Tr}\big(Z^\top D^{-1/2}WD^{-1/2}Z\big)=\operatorname{Tr}\big(Z^\top(I-D^{-1/2}WD^{-1/2})Z\big)=\operatorname{Tr}(Z^\top L_{\mathrm{sym}}Z),
$$

which is Eq. (5.8). $\blacksquare$

### B.3 Derivative of $\log\Phi$ (Eq. 6.2)

With $q=\lVert y-z\rVert^2$ and $\Phi=(1+aq^b)^{-1}$: $\log\Phi=-\log(1+aq^b)$, so $\nabla_y\log\Phi=-\frac{ab\,q^{b-1}}{1+aq^b}\nabla_y q=-\frac{ab\,q^{b-1}}{1+aq^b}\cdot 2(y-z)=-\frac{2ab\,q^{b-1}}{1+aq^b}(y-z)$, using $\nabla_y q = 2(y-z)$. This matches Eq. (6.2) exactly.

### B.4 Derivative of $\log(1-\Phi)$ (Eq. 6.3, exact form before regularization)

$1-\Phi=\frac{aq^b}{1+aq^b}$, so $\log(1-\Phi)=\log a+b\log q-\log(1+aq^b)$. Then $\nabla_y\log(1-\Phi)=\Big[\frac{b}{q}-\frac{ab\,q^{b-1}}{1+aq^b}\Big]\nabla_y q =2(y-z)\Big[\frac{b}{q}-\frac{ab\,q^{b-1}}{1+aq^b}\Big]$. Combining the bracket over a common denominator: $\frac{b(1+aq^b)-abq^b}{q(1+aq^b)}=\frac{b}{q(1+aq^b)}$. So the exact derivative is $\frac{2b}{q(1+aq^b)}(y-z)$, which has a $1/q$ singularity at $q=0$; Eq. (6.3) is this expression with $q$ replaced by $q+\varepsilon$ in the denominator only, which is the regularization actually implemented (Section 6.1).

### B.5 The mean-field reference update (Eq. 6.6), in detail

Consider one pass of UMAP's stochastic optimizer over the graph. For each stored (directed) edge $(i,j)$ with weight $W_{ij}$, an attractive update is applied with a frequency governed by $W_{ij}$, contributing (in the continuous/expected-frequency idealization used here) a term $W_{ij}\,g_+(y_i,y_j)$ to point $i$'s expected displacement; summing over all $j$ gives $A_i$ (Eq. 6.4). Independently, for every positive sample touching $i$, UMAP's negative-sampling scheme draws $m$ (the configured `negative_sample_rate`) reference points uniformly at random and applies a repulsive update against each. The number of positive-sample events touching $i$ per pass is governed by $i$'s total incident edge mass $s_i=\sum_jW_{ij}$; each of the $m\cdot(\text{event count})$ resulting negative draws contributes, in expectation over its uniformly-drawn target, exactly $\mathbb E_{c\sim\mathrm{Uniform}(X)}[g_-(y_i,y_c)]=B_X(y_i,t)$ (Eq. 6.5). Collecting the proportionality constants (event count $\propto s_i$, each contributing $m$ draws in expectation $B_X$) gives the mean-field repulsive contribution $m\,s_i\,B_X(y_i(t),t)$, and summing the attractive and repulsive contributions gives Eq. (6.6). This derivation is an expectation over the *sampling scheme*, holding by linearity of expectation over the independent draws involved; it makes no claim about any single realized stochastic trajectory (Section 7.4).

The Monte-Carlo estimator (Eq. 6.7) shares one sampled negative set $\{c_1,\ldots,c_M\}$ across all $i$ within a given step, rather than drawing an independent set per $i$ (which would cost $O(N \cdot M)$ independent draws and, more importantly, would not reduce the asymptotic cost of the *evaluation* itself, since $g_-$ still has to be evaluated for every $(i,c_\ell)$ pair either way). Sharing the negative set across $i$ makes the per-step cost $O(N\cdot M)$ in *evaluations* rather than $O(N^2)$, at the cost of introducing a shared-sample correlation across $i$ within one step; this correlation is not removed by this implementation, only amortized across the many steps of the trajectory (each with an independently resampled shared set).

### B.6 Proof of the candidate-recall lemma (Eq. 5.4–5.6)

Let $C_*\subseteq X$ be the candidate set with $K_X(x_*)\subseteq C_*$. By definition, $K_X(x_*)$ consists of the $k$ points of $X$ with the $k$ smallest values of $d(x_*,\cdot)$; equivalently, for every $p\in K_X(x_*)$ and every $q\in X\setminus K_X(x_*)$, $d(x_*,p)\le d(x_*,q)$ (with ties broken consistently by the same rule on both sides of any comparison). Restricting attention to the subset $C_*$: since $K_X(x_*)\subseteq C_*$, and every point of $C_*\setminus K_X(x_*)$ is a subset of $X\setminus K_X(x_*)$, the same inequality holds for every $p\in K_X(x_*)$ against every $q\in C_*\setminus K_X(x_*)$. Hence $K_X(x_*)$ is exactly the set of $k$ points of $C_*$ with the $k$ smallest distances to $x_*$. `RETRIEVE_AND_RERANK` computes exact distances $d(x_*,x_j)$ for $j\in C_*$ (the same metric used to define $K_X(x_*)$) and selects the $k$ smallest, i.e. $\widehat K_*=K_X(x_*)$ (Eq. 5.5). Finally, $\rho_*,\sigma_*$ are obtained by a deterministic root-solve (the same smooth-kNN equation used at reference-graph-construction time) applied to the multiset of $k$ distances $\{d(x_*,x_j):j\in\widehat K_*\}$, and $\widehat\mu_{*\to j}=\exp(-\max(0,d(x_*,x_j)-\rho_*)/\sigma_*)$ is a deterministic function of $\rho_*,\sigma_*$, and that same distance. Since $\widehat K_*=K_X(x_*)$ implies the multiset of distances used is identical to the one that would be used to compute the "true" quantities $\rho_*^{\mathrm{true}},\sigma_*^{\mathrm{true}},\mu^{\mathrm{true}}_{*\to j}$ directly from $K_X(x_*)$, and both are the same deterministic function applied to the same input, the two must be equal (Eq. 5.6). $\blacksquare$

## Appendix C — Architecture Tables

**Retriever** (`models/retriever.py::DualEncoder`)

| | Query tower $f_\theta$ | Key tower $g_\psi$ |
|---|---|---|
| Input dim | 256 (COIL) / 50 (pancreas) | same |
| Hidden layers | 512 → 256 | 512 → 256 |
| Activation | GELU | GELU |
| Output dim | 128 | 128 |
| Output normalization | L2 | L2 |
| Score | $q^\top k / \tau$, $\tau=0.1$ | |
| Optimizer / lr / steps | Adam / $10^{-3}$ / 2,000 | |

**Spectral encoder** (`models/spectral.py::SpectralEncoderNet`)

| Input dim | Hidden layers | Activation | Raw output dim $r$ | Retained dim $d$ | Optimizer / lr / steps |
|---|---|---|---|---|---|
| 256 / 50 | 512 → 256 | GELU | 3 | 2 | Adam / $10^{-3}$ / 2,000 |

Post-hoc (not trained): whitening $T\in\mathbb R^{3\times3}$; projection $P\in\mathbb R^{3\times2}$ (non-trivial eigenvectors of $H=Z^\top L_{\mathrm{sym}}Z$); reference-only center $c\in\mathbb R^2$ and scale $\gamma\in\mathbb R$, calibrated to target spread 10.0.

**Repulsion network** (`models/repulsion.py::RepulsionField`)

| Input | Time embedding | Hidden width | Residual blocks | Activation | Output dim | Output init | Optimizer / lr / steps |
|---|---|---|---|---|---|---|---|
| $y\in\mathbb R^2$, $\gamma(t)\in\mathbb R^{32}$ | 16 sin + 16 cos, log-spaced frequencies | 256 | 4 | GELU | 2 | zero | Adam / $10^{-3}$ / 4,000 |

**Reference mean flow** (`dynamics.py`, `training/flow.py`)

| Steps $E$ | $\alpha_0$ | schedule | negative-sample rate $m$ | dynamics negatives $M$ | pairwise/aggregate clip |
|---|---|---|---|---|---|
| 200 | 1.0 | $\alpha_e=\alpha_0(1-e/E)$ | 5 | 64 | $\pm4$ (both levels; Section 6.1) |

## Appendix D — Implementation Mapping

| Mathematical object | Code |
|---|---|
| $\mathrm{Mu}$ (directed) | `src/umaping/graph.py::compute_directed_membership` |
| $W$ (symmetric) | `src/umaping/graph.py::symmetrize_fuzzy_union` |
| $\rho_i,\sigma_i$ | `src/umaping/graph.py::smooth_knn_dist` |
| query-side $\widehat\rho_*,\widehat\sigma_*,\widehat\mu_{*\to j}$ | `src/umaping/graph.py::query_fuzzy_weights` |
| exact chunked kNN (used for both graph construction and reranking) | `src/umaping/graph.py::chunked_exact_knn` |
| retriever model | `src/umaping/models/retriever.py::DualEncoder` |
| candidate index (exact / optional ANN) | `src/umaping/models/retriever.py::ExactChunkedIndex`, `PyNNDescentIndex` |
| retriever training loss/masking | `src/umaping/training/retriever.py` (`train_retriever`, `build_negative_mask`, `_sample_positive_per_row`) |
| spectral network | `src/umaping/models/spectral.py::SpectralEncoderNet` |
| spectral training loss | `src/umaping/training/spectral.py::train_spectral` |
| spectral calibration ($T,P,c,\gamma$) | `src/umaping/training/spectral.py::compute_calibration`, `src/umaping/models/spectral.py::SpectralCalibration` |
| $\Phi, g_+, g_-$, $(a,b)$ fit | `src/umaping/umap_forces.py` |
| reference trajectory construction | `src/umaping/dynamics.py::simulate_reference_dynamics`, `src/umaping/training/flow.py::build_reference_trajectory` |
| frozen reference trajectory storage/lookup | `src/umaping/dynamics.py::ReferenceTrajectory`, `TorchTrajectoryView` |
| repulsion model | `src/umaping/models/repulsion.py::RepulsionField` |
| repulsion training | `src/umaping/training/flow.py::train_repulsion_field` |
| single-point inference | `src/umaping/inference.py::InferenceEngine.embed_one` |
| baselines/ablations | `src/umaping/baselines.py` |
| stage orchestration/checkpointing | `src/umaping/pipeline.py` |
| retrieval / spectral / field / embedding evaluation | `src/umaping/evaluation/retrieval.py`, `spectral.py`, `field.py`, `embedding.py` |

## Appendix E — Experimental Splits

**COIL-20 / COIL-100.** Filenames are parsed as `obj{N}__{X}.png`; for COIL-20, `X` is a sequential pose index 0–71, while for COIL-100, `X` is the literal rotation angle in degrees (0, 5, …, 355) — the split logic ranks poses by parsed value *within each object* and keys the every-4th-pose holdout off that rank, which is correct for both conventions (`data/coil.py::compute_pose_ranks`). Reference: 1,080 images (COIL-20) / 5,400 images (COIL-100) — both counts observed directly in an actual pipeline run on the target remote environment (log line `Building reference kNN graph (N=..., n_neighbors=15)`), matching the expected $54\times20$ and $54\times100$. Query: 360 (COIL-20) / 1,800 (COIL-100), by the complementary count.

**Pancreas.** Steps, in order, all fit on the reference split only where fitting is involved: (1) raw counts from `layers['counts']`; (2) `sc.pp.normalize_total(target_sum=1e4)`; (3) `sc.pp.log1p()` — (2)–(3) are per-cell transforms and are applied identically to every cell before the split, since they use no cross-cell statistics and hence cannot leak query information into the reference; (4) `sc.pp.highly_variable_genes(n_top_genes=2000, batch_key="tech", flavor="seurat")` computed on the reference cells only; (5) the resulting 2,000-gene set is applied to both splits; (6) a 50-component PCA (`sklearn.decomposition.PCA`) fit on the reference cells' HVG matrix only; (7) the query split is transformed (never refit) through that same PCA. An actual pipeline run on the target remote environment observed 11,703 reference cells (log line `Building reference kNN graph (N=11703, n_neighbors=15)`), implying roughly 4,679 query cells (`smartseq2` + `celseq2` combined), out of a total dataset size documented as approximately 16,382 cells by the upstream tutorial (RUNTIME_CHECKS.md); this document reports the observed reference count directly rather than the approximate total, since it was directly measured rather than assumed.

## Appendix F — Exact vs. Approximate Correspondence Table

This reprints and slightly extends Table 7.1 with the equation number establishing each row.

| Component | Status | Established by |
|---|---|---|
| $W = \mathrm{Mu}+\mathrm{Mu}^\top-\mathrm{Mu}\odot\mathrm{Mu}^\top$ | exact identity | Section 4.1; Appendix B.1 |
| $\Phi(y,z)=(1+aq^b)^{-1}$, $(a,b)$ curve fit | exact reimplementation, numerically verified | Eq. 6.1; Section 7.1b |
| $g_+ = \nabla_y\log\Phi$ | exact, closed-form | Eq. 6.2; Appendix B.3 |
| $g_- \approx \nabla_y\log(1-\Phi)$ | approximate (regularized at $q\approx0$) | Eq. 6.3; Appendix B.4 |
| $\widehat\mu_{*\to j}=\mu_{*\to j}$ | conditional on $K_X(x_*)\subseteq C_*$ | Eq. 5.4–5.6; Appendix B.6 |
| $\operatorname{span}(Z)=\operatorname{span}(U_{1:d+1})$ | idealized/asymptotic subspace target | Eq. 5.13; Section 7.3 |
| $F_i=A_i+ms_iB_X$ | expectation over UMAP's own sampling scheme | Eq. 6.6; Appendix B.5 |
| $B_\phi\approx B_X$ | learned function approximation | Eq. 6.8–6.9 |
| $Y_{\mathrm{ref}}(t)$ vs. stochastic UMAP SGD | deterministic mean-field approximation | Section 7.4 |
| $F_{\Theta,\mathcal M_X}(x_*)$ vs. $\operatorname{UMAP}(X\cup\{x_*\})_*$ | not equivalent | Eq. 7.1; Section 7.5 |

## Appendix G — Test / Numerical Validation

This appendix documents only what has actually been checked, per `RUNTIME_CHECKS.md`, and is written to keep engineering validation separate from scientific results (Section 15).

**Unit tests (35 total, all passing at time of writing, both on the target remote environment and independently reproduced on a second local machine):** smooth-kNN semantics cross-checked directly against the current `umap-learn` source (accounting explicitly for the self-column convention difference documented in Section 4.1/`graph.py`); the symmetric fuzzy union identity; `find_ab_params` verified numerically identical to `umap-learn`'s own curve fit; the closed-form $g_+$ and $g_-$ verified against autograd derivatives of $\log\Phi$ and $\log(1-\Phi)$ (the $g_-$ check uses point pairs with a guaranteed minimum separation, for the reason given in Section 6.1); the Monte-Carlo mean-repulsion estimator verified, on a tiny synthetic reference set, both against an exhaustive brute-force average and for convergence (decreasing error with more samples, averaged over trials to avoid RNG-driven flakiness); the retriever's positive/negative masking invariants (a query point is never usable as its own negative, and a true-but-unsampled $\mathrm{Mu}$-neighbor is excluded from the negative pool by default, without ever masking the sampled positive itself); the spectral encoder's output dimensions and its frozen-calibration reuse (a fresh `SpectralEmbedder` built only from a saved model and calibration reproduces `embed_one`'s output, with no access to the training graph); `embed_one`'s independence from other query points and from processing order, including for the stochastic repulsion-oracle diagnostic mode specifically (added after a real bug of exactly this kind was found and fixed, see below); the COIL angle-rank split's disjointness and full object coverage on both sides; the pancreas HVG/PCA fitting's invariance to arbitrary changes in the query cells' data; and that a saved model can be reloaded from disk and used for single-point inference.

**A synthetic ("mock") end-to-end smoke test** (`configs/mock.yaml`, `data/mock.py`) runs every pipeline stage — preprocessing through evaluation and figure generation — on a small synthetic Gaussian-blob-mixture dataset in a few seconds, with no download and no real training; it exists to catch integration-level bugs (in particular, bugs that only manifest once the full evaluate/analyze path executes) without requiring a full run against real data, and was used to independently reproduce two bugs found on real remote runs (below) after they were fixed.

**Bugs found by actual execution (not by static review) and fixed, in the order found:** (1) a repository `.gitignore` pattern that was unanchored and therefore also matched, and silently excluded from version control, the `src/umaping/data/` subpackage itself; (2) a stateful random-number generator stored on `InferenceEngine` whose state advanced across `embed_one` calls, making the `oracle_mc` repulsion-diagnostic mode's output depend on which other queries had been processed earlier on the same engine instance — fixed by constructing a fresh, locally-seeded generator at the start of every `embed_one` call; (3) the repulsion-field training and evaluation loops not threading the configured `flow.grad_clip` into their teacher-target computation, so changing that setting would silently train/evaluate against a mismatched field; (4) a shape mismatch in the repulsion-oracle-diagnostic baseline's evaluation, which only embeds a configurable subset of queries but was compared against ground truth for the full query set; (5) an object-dtype numpy array (produced by a pandas string column via `.astype(str).to_numpy()`) that `numpy.savez` pickles but `numpy.load(..., allow_pickle=False)` then cannot read back, encountered on an actual pancreas run and fixed by casting such arrays to a fixed-width string dtype before saving. Each of these was found via actual execution — either on the target remote GPU machine or, for (1) and later regression coverage of (2), via independent static/adversarial code review and local reproduction — not by reasoning about the code alone, which is itself informative about the limits of the static-only review this implementation initially received.

**Not yet exercised at all:** the optional `pynndescent`-backed approximate retrieval backend; multi-seed runs; the optional transductive-UMAP baseline; and, as of this writing, a fully successful end-to-end run of the pancreas pipeline on the target remote environment (the underlying caching bug was fixed and reproduced locally, but the remote re-run's outcome had not yet been reported back at the time this document was written).

## Appendix H — Open Questions Before Submission

- Results across multiple random seeds, for all three datasets and all six baselines/ablations of Section 13.
- The actual scientific results themselves (Section 15 is entirely TBD).
- Wall-clock/throughput scaling with reference set size $N$, in particular for the current $O(N)$-per-query exact retrieval backend.
- Sensitivity of end-to-end quality to the candidate pool size $M$.
- Sensitivity to the number of flow steps $E$ (Algorithm 4) and to the number of Monte-Carlo negative samples used both in reference-trajectory construction and in repulsion-field teacher generation.
- Sensitivity to the pairwise and aggregate gradient-clipping thresholds (Section 6.1).
- Sensitivity to the reference-trajectory `checkpoint_stride` (interpolation error introduced by storing fewer checkpoints).
- Whether/how the optional `pynndescent` ANN backend changes candidate recall and latency relative to the default exact backend.
- A direct empirical comparison against Parametric UMAP (Sainburg, McInnes, and Gentner, 2021) under matched preprocessing.
- Whether a comparison against any other specific named out-of-sample UMAP baseline referenced in the literature review (Section 9) is appropriate and feasible.
- Stronger, deliberately out-of-distribution query evaluation (beyond the held-out-pose/held-out-technology splits used here, which hold out specific poses/technologies but not necessarily distributionally distant ones).
- A complete novelty/prior-art search before any claim of being first at anything — none is made in this document, and none should be added without such a search.

---

## References

All entries below were independently fetched and read (full text, not abstract-only) as part of writing this document; the arXiv/proceedings version actually inspected is given for each, since equation numbers can shift between versions.

- McInnes, L., Healy, J., and Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.* arXiv:1802.03426v3 (inspected 18 Sep 2020 revision).
- Sainburg, T., McInnes, L., and Gentner, T. Q. (2020/2021). *Parametric UMAP: Learning Embeddings with Deep Neural Networks for Representation and Semi-Supervised Learning* (v1 title; retitled *Parametric UMAP Embeddings for Representation and Semi-Supervised Learning* in later revisions). arXiv:2009.12981 (v1 and v4 inspected).
- Shaham, U., Stanton, K., Li, H., Nadler, B., Basri, R., and Kluger, Y. (2018). *SpectralNet: Spectral Clustering Using Deep Neural Networks.* ICLR 2018. arXiv:1801.01587v6.
- Ben-Ari, R., Yacobi, N., and Shaham, U. (2025). *Generalizable Spectral Embedding with an Application to UMAP.* Transactions on Machine Learning Research, 07/2025. arXiv:2501.11305v2.
- Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., and Yih, W. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020, pp. 6769–6781. https://aclanthology.org/2020.emnlp-main.550/ (arXiv:2004.04906v3).
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. (2023). *Flow Matching for Generative Modeling.* arXiv:2210.02747v2.
- Damrich, S., and Hamprecht, F. A. (2021). *On UMAP's True Loss Function.* NeurIPS 2021. arXiv:2103.14608v2.
- Islam, M. T., and Fleischer, J. W. (2026). *On Out-of-sample Embedding in UMAP.* arXiv:2606.04451v1.
- Siddiqui, M. S. R., Eliasof, M., and Haber, E. (2026). *Graph Flow Matching: Enhancing Image Generation with Neighbor-Aware Flow Fields.* AAAI-26 (Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, No. 30; DOI 10.1609/aaai.v40i30.39741). Also arXiv:2505.24434v3.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., and Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS 2013. (Cited by McInnes, Healy, and Melville, 2018, as the source of the negative-sampling technique their own optimizer uses; not independently re-verified beyond that citation.)
