"""Analysis pass for the Symmetry-II Steerling experiment.

Loads the Jacobians cached by ``experiment_steerling_symmetryII.py`` and:

* computes the QR and SVD-truncated subspace-alignment metrics,
* reports the effective rank read off the singular spectra,
* saves a log-y plot of the spectra to inspect the rank cliff.

No GPU needed and no model load — this file only touches the cached
``.pt`` artifacts. Run after ``experiment_steerling_symmetryII.py`` has
populated the cache directory.
"""

import os

import torch


# Keep these in sync with experiment_steerling_symmetryII.py so the cache
# filenames match.
OUT_DIR = "steerling_experiment"


def _set_publication_style() -> None:
    """Apply a LaTeX-like serif style with publication-grade font sizes.

    Idempotent — safe to call from every plot function.  Uses matplotlib's
    built-in Computer Modern mathtext rather than ``text.usetex=True`` so
    no external LaTeX install is required.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
    })


# Relative tolerance for "is this singular value real or numerical noise?".
# bf16 noise floor is ~1e-3 of S.max(); fp32 is ~1e-7.  Use a value above
# the bf16 noise floor when Jacobians came from a bf16 forward pass.
EFFECTIVE_RANK_RTOL = 1e-4
FRACTION_ENERGY = 0.999 # Capture 99.9% of the energy


def normalized_gradient_alignment_metric(g_c, g_y):
    """Normalized Frobenius distance between two gradient subspaces (QR).

    Each input has shape ``(Batch, InputDim, EmbDim)`` and is treated as
    ``InputDim`` row vectors spanning a subspace of ``R^EmbDim``.  The
    two inputs must share ``Batch`` and ``EmbDim``; ``InputDim`` may
    differ (``k_c`` vs ``k_y``).
    """
    # 1. Orthonormal bases (QR on the (EmbDim, InputDim) view).
    # Cast to fp32: torch.linalg.qr has no bf16 CPU kernel, and QR is
    # more numerically stable in fp32. Cost is negligible at (4096, k).
    q_c, _ = torch.linalg.qr(g_c.float().transpose(1, 2), mode='reduced')  # (B, D, k_c)
    q_y, _ = torch.linalg.qr(g_y.float().transpose(1, 2), mode='reduced')  # (B, D, k_y)

    k_c = q_c.shape[2]
    k_y = q_y.shape[2]

    # 2. Projection matrices P = Q @ Q.T  →  (B, D, D).
    p_c = torch.bmm(q_c, q_c.transpose(1, 2))
    p_y = torch.bmm(q_y, q_y.transpose(1, 2))

    raw_loss = torch.linalg.matrix_norm(p_c - p_y, ord='fro')

    # 3. Normalize by the maximum possible distance between two rank-k subspaces.
    max_val = torch.sqrt(torch.tensor(k_c + k_y, dtype=raw_loss.dtype, device=raw_loss.device))
    return raw_loss / max_val


def normalized_gradient_alignment_metric_svd(
    g_c, g_y,
    *,
    method: str = "rtol",
    rtol: float | None = None,
    fraction: float = 0.99,
):
    """SVD-truncated variant of the gradient-subspace alignment metric.

    The QR-based metric collapses on square rank-deficient Jacobians
    (returns ``p = I`` regardless of true rank).  This variant truncates
    columns of ``U`` and selects a rank ``r`` per input via one of two
    strategies, controlled by ``method``:

    * ``method="rtol"`` (default).  Keep singular values greater than
      ``rtol * s.max()``.  Reasonable when the spectrum has a clear
      cliff; brittle on smoothly-decaying spectra (you have to invent a
      cutoff).
    * ``method="energy"``.  Keep the smallest ``r`` such that the top-r
      singular values capture ``fraction`` of the squared Frobenius
      norm.  More principled for smooth spectra — ``0.99`` is a sensible
      default.

    The resulting projections ``P = Q Q.T`` and the denominator
    ``sqrt(r_c + r_y)`` **both** use the chosen rank, so the metric is
    self-consistent.

    Args:
        g_c, g_y: Tensors of shape ``(Batch=1, InputDim, EmbDim)`` with
            matching ``EmbDim``.
        method: ``"rtol"`` or ``"energy"``.  Truncation strategy.
        rtol: Relative tolerance, used only when ``method="rtol"``.
            ``None`` falls back to NumPy/MATLAB's
            ``max(D, InputDim) * eps(dtype)`` convention.
        fraction: Cumulative-energy target in ``(0, 1]``, used only
            when ``method="energy"``.

    Returns:
        Tuple ``(metric, s_c, s_y)`` with metric a scalar tensor and
        ``s_c``/``s_y`` the (full, untruncated) sorted-descending
        singular values — handy for plotting the spectrum.
    """
    if g_c.shape[0] != 1 or g_y.shape[0] != 1:
        raise NotImplementedError(
            "SVD-truncation path assumes Batch=1 (ragged ranks per batch "
            "would need a list output)."
        )
    if method not in ("rtol", "energy"):
        raise ValueError(f"method must be 'rtol' or 'energy', got {method!r}.")

    def _proj(g):
        g_2d = g.float().reshape(-1, g.shape[-1])                  # (k, D)
        u, s, _ = torch.linalg.svd(g_2d.T, full_matrices=False)    # u: (D, min(D, k))
        if s.numel() == 0:
            d = g_2d.shape[-1]
            zero = torch.zeros(d, d, dtype=g_2d.dtype, device=g_2d.device)
            return zero, 0, s
        if method == "energy":
            r = _effective_rank_energy(s, fraction)
        else:  # "rtol"
            tol = rtol if rtol is not None else (
                max(g_2d.shape) * torch.finfo(g_2d.dtype).eps
            )
            cutoff = float(s.max()) * tol
            r = int((s > cutoff).sum().item())
        q = u[:, :r]                                                # (D, r)
        return q @ q.T, r, s                                        # (D, D), rank, full s

    p_c, k_c, s_c = _proj(g_c)
    p_y, k_y, s_y = _proj(g_y)

    raw_loss = torch.linalg.matrix_norm(p_c - p_y, ord='fro')
    if k_c + k_y == 0:
        return raw_loss, s_c, s_y
    max_val = torch.sqrt(torch.tensor(
        float(k_c + k_y), dtype=raw_loss.dtype, device=raw_loss.device,
    ))
    return raw_loss / max_val, s_c, s_y


def normalized_gradient_alignment_metric_containment(
    g_h, g_c,
    *,
    method: str = "rtol",
    rtol: float | None = None,
    fraction: float = 0.99,
):
    """Asymmetric containment metric: how much of ``g_h`` is captured by ``g_c``?

    Unlike the symmetric Frobenius-of-projection-difference metric, this
    one-sided variant doesn't penalize ``g_c`` for spanning *more*
    directions than ``g_h``.  It directly answers "does the concept
    Jacobian span ``∂h_bar/∂input``'s subspace?", which is the
    Symmetry-II faithfulness question.

    Definition::

        m² = 1 - ||Q_cᵀ Q_h||²_F / rank(Q_h)

    where ``Q_h``, ``Q_c`` are orthonormal bases for the truncated row
    spans of ``g_h`` and ``g_c`` respectively.  The numerator
    ``||Q_cᵀ Q_h||²_F = trace(P_c P_h) = sum_i cos²(θ_i)`` is the sum of
    squared cosines of the principal angles between the two subspaces.

    * ``m = 0``: row span of ``g_h`` is fully contained in row span of
      ``g_c``.
    * ``m = 1``: the two row spans are orthogonal.

    Adding more directions to ``g_c`` can only *increase* the overlap,
    so the metric is monotonically non-increasing as ``g_c`` grows —
    no "right-tail rise" like the symmetric metric.

    Args mirror :func:`normalized_gradient_alignment_metric_svd`.  The
    truncation strategy (``method``, ``rtol``, ``fraction``) is applied
    to **both** Jacobians symmetrically, just as in the SVD metric.

    Returns:
        Tuple ``(metric, s_h, s_c)`` — scalar metric and the full
        (untruncated) sorted-descending singular values of ``g_h`` and
        ``g_c`` respectively.  Note the order matches the input order
        ``(g_h, g_c)``, *not* the SVD metric's ``(g_c, g_y)``.
    """
    if g_h.shape[0] != 1 or g_c.shape[0] != 1:
        raise NotImplementedError(
            "Containment metric assumes Batch=1."
        )
    if method not in ("rtol", "energy"):
        raise ValueError(f"method must be 'rtol' or 'energy', got {method!r}.")

    def _basis(g):
        g_2d = g.float().reshape(-1, g.shape[-1])
        u, s, _ = torch.linalg.svd(g_2d.T, full_matrices=False)
        if s.numel() == 0:
            d = g_2d.shape[-1]
            return torch.zeros(d, 0, dtype=g_2d.dtype, device=g_2d.device), 0, s
        if method == "energy":
            r = _effective_rank_energy(s, fraction)
        else:
            tol = rtol if rtol is not None else (
                max(g_2d.shape) * torch.finfo(g_2d.dtype).eps
            )
            cutoff = float(s.max()) * tol
            r = int((s > cutoff).sum().item())
        return u[:, :r], r, s

    q_h, r_h, s_h = _basis(g_h)
    q_c, _, s_c = _basis(g_c)

    if r_h == 0:
        # Degenerate: g_h has no signal → trivially contained.
        return torch.zeros((), dtype=q_h.dtype, device=q_h.device), s_h, s_c

    overlap_sq = (q_c.T @ q_h).square().sum()                  # ||Q_cᵀ Q_h||²_F
    m_sq = (1.0 - overlap_sq / float(r_h)).clamp(min=0.0)      # guard against tiny <0
    return m_sq.sqrt(), s_h, s_c


def plot_jacobian_spectrum(
    s: torch.Tensor,
    *,
    out_path: str,
    label: str | None = None,
    expected_drop: int | None = None,
    log_x: bool = False,
) -> None:
    """Plot sorted-descending singular values of one Jacobian on a log-y axis.

    A sharp drop in magnitude marks the boundary between "real" directions
    and numerical noise — i.e. the effective rank of the Jacobian. Pass
    ``expected_drop`` to overlay a vertical guideline (e.g. ``TOPK`` for
    the masked-concept-bottleneck case).

    Args:
        s: 1-D tensor of singular values (sorted descending).
        out_path: Full path for the saved figure.
        label: Optional legend label for the spectrum.
        expected_drop: Optional rank to mark with a vertical dashed line.
        log_x: If ``True``, also put the x-axis (index) on log scale.
            Useful when the spectrum has many indices and you want to
            see the head decay clearly — a log-log plot turns power-law
            decay into a straight line.  Default ``False`` (linear x).
    """
    import matplotlib.pyplot as plt
    _set_publication_style()

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    vals = s.detach().cpu().float().numpy()
    plot = ax.loglog if log_x else ax.semilogy
    plot(range(1, len(vals) + 1), vals, marker="o", markersize=3, label=label)
    if expected_drop is not None:
        ax.axvline(
            x=expected_drop + 0.5,
            color="red", linestyle="--", alpha=0.6,
            label=f"rank = {expected_drop}",
        )
    ax.set_xlabel(r"singular value index $i$")
    ax.set_ylabel(r"$\sigma_i$")
    ax.grid(True, alpha=0.3, which="both")
    if label is not None or expected_drop is not None:
        ax.legend(loc="best")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_concept_activations(
    activations: torch.Tensor,
    *,
    out_path: str,
    topk_marker: int | None = None,
    log_y: bool = True,
    log_x: bool = False,
) -> None:
    """Plot known-concept activations sorted descending (highest → lowest).

    Useful to see *how peaked* the model's concept distribution is for the
    prompt: a sharp head followed by a long flat tail near zero suggests
    a small "active set"; a slowly decaying curve suggests many concepts
    contributing weakly.

    Args:
        activations: Tensor of concept activations. Any shape; flattened.
            Typically the ``out.probs`` returned by an inference query
            over all known concepts, shape ``(B, n_known)``.
        out_path: Full path for the saved figure.
        topk_marker: Optional rank to mark with a vertical dashed line
            (e.g. ``TOPK``).
        log_y: Use a log-scale y-axis (default). Set to ``False`` for a
            linear axis, useful when activations are already in ``[0, 1]``
            and you want to read absolute magnitudes.
        log_x: Use a log-scale x-axis (default ``False``). Helpful when
            ``n_known`` is large (~33k) — a log-x plot lets you read the
            head, body, and tail of the distribution in one view.
    """
    import matplotlib.pyplot as plt
    _set_publication_style()

    vals = (
        activations.detach().cpu().float().reshape(-1)
        .sort(descending=True).values.numpy()
    )

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    if log_x and log_y:
        plot = ax.loglog
    elif log_x:
        plot = ax.semilogx
    elif log_y:
        plot = ax.semilogy
    else:
        plot = ax.plot
    plot(range(1, len(vals) + 1), vals, linewidth=1.4)
    if topk_marker is not None:
        ax.axvline(
            x=topk_marker + 0.5,
            color="red", linestyle="--", alpha=0.6,
            label=fr"top-{topk_marker}",
        )
        ax.legend(loc="best")
    ax.set_xlabel(r"concept index $i$ (sorted by activation)")
    ax.set_ylabel(r"activation $\sigma(\nabla_z c_i)$")
    ax.grid(True, alpha=0.3, which="both" if (log_y or log_x) else "major")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_jacobian_spectra_comparison(
    spectra: list[tuple[torch.Tensor, str, float | None]],
    *,
    out_path: str,
    expected_drop: int | None = None,
    log_x: bool = False,
) -> None:
    """Plot multiple Jacobian spectra on the same axes for side-by-side comparison.

    Each input entry is ``(singular_values, label, metric)``.  If ``metric``
    is not ``None``, it is appended to the label so the legend doubles as
    the summary readout.

    Args:
        spectra: List of ``(s, label, metric)`` tuples. ``s`` is a 1-D
            tensor of sorted-descending singular values. ``metric`` may
            be ``None`` (no annotation) or a float.
        out_path: Full path for the saved figure.
        expected_drop: Optional rank to mark with a vertical dashed line.
        log_x: If ``True``, use log-log axes (default linear x, log y).
    """
    import matplotlib.pyplot as plt
    _set_publication_style()

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    plot = ax.loglog if log_x else ax.semilogy
    for s, label, metric in spectra:
        vals = s.detach().cpu().float().numpy()
        full_label = label if metric is None else fr"{label}  $m{{=}}{metric:.3f}$"
        plot(range(1, len(vals) + 1), vals, marker="o", markersize=2.5, label=full_label)
    if expected_drop is not None:
        ax.axvline(
            x=expected_drop + 0.5,
            color="red", linestyle="--", alpha=0.6,
            label=fr"rank = {expected_drop}",
        )
    ax.set_xlabel(r"singular value index $i$")
    ax.set_ylabel(r"$\sigma_i$")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_topk(
    tracks: list[tuple[str, list[tuple[int, torch.Tensor, torch.Tensor]]]],
    *,
    out_path: str,
    method: str = "rtol",
    rtol: float | None = None,
    fraction: float = 0.99,
    log_x: bool = False,
    mark_rank: int | None = None,
    mark_label: str | None = None,
    mark_ranks: list[tuple[str, int]] | None = None,
    show_mark_ranks: bool = True,
    mark_cmap: str = "Reds",
    metric_fn=None,
    title: str = "",
) -> None:
    """Plot the SVD subspace-alignment metric as a function of ``TOPK``.

    Each entry in ``runs`` is ``(topk, h_bar_g, concepts_g)`` — the
    Jacobians produced by an experiment run configured with that
    specific ``TOPK``.  For each entry the function computes the metric
    once and plots the resulting ``(topk, metric)`` pair.

    Interpretation: if the concept bottleneck is faithful, the metric
    should stay near zero across all ``TOPK`` values (the masked ``h_bar``
    subspace is always perfectly recovered by the top-``TOPK`` concept
    gradients).  Departures from zero signal that some directions of
    ``∂h_bar/∂input`` are *not* spanned by the concept gradients — either
    because of numerical noise (small) or because the bottleneck isn't
    truly rank-``TOPK`` (large).

    Args:
        runs: List of ``(topk, h_bar_g, concepts_g)`` tuples.  Each
            ``h_bar_g`` / ``concepts_g`` is already reshaped to
            ``(Batch=1, InputDim, EmbDim)`` and was computed with the
            paired ``topk`` masking configuration.
        out_path: Full path for the saved figure.
        label: Optional legend label for the curve.
        rtol: Forwarded to
            :func:`normalized_gradient_alignment_metric_svd`.
        log_x: If ``True``, put the x-axis (TOPK) on log scale — useful
            when sweeping ``TOPK`` over many orders of magnitude
            (e.g. 1, 2, 4, 8, 16, 32, …).
    """
    import matplotlib.pyplot as plt
    _set_publication_style()

    if metric_fn is None:
        metric_fn = normalized_gradient_alignment_metric_svd

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    plot_fn = ax.semilogx if log_x else ax.plot

    for track_label, runs in tracks:
        runs_sorted = sorted(runs, key=lambda r: r[0])
        topks: list[int] = []
        metrics: list[float] = []
        for topk, h_bar_g, concepts_g in runs_sorted:
            m, _, _ = metric_fn(
                h_bar_g, concepts_g,
                method=method, rtol=rtol, fraction=fraction,
            )
            topks.append(topk)
            metrics.append(m.item())
        plot_fn(topks, metrics, marker="o", markersize=5, linewidth=1.8, label=track_label)

    if show_mark_ranks and mark_ranks:
        # Gradient-color vertical lines for a set of (label, rank) pairs.
        # Drawn *before* `mark_rank` so the latter sits on top.
        cmap = plt.get_cmap(mark_cmap)
        n = len(mark_ranks)
        for i, (lbl, rnk) in enumerate(mark_ranks):
            t = 0.3 + 0.55 * (i / max(n - 1, 1))
            ax.axvline(
                x=rnk, color=cmap(t), linestyle="--",
                alpha=0.55, linewidth=1.0, label=lbl,
            )
    if mark_rank is not None:
        ax.axvline(
            x=mark_rank,
            color="gray", linestyle="--", alpha=0.75, linewidth=1.4,
            zorder=4,
            label=mark_label if mark_label is not None else fr"rank = {mark_rank}",
        )
    ax.set_xlabel(r"concept bottleneck size")
    ax.set_ylabel(r"$\sqrt{\mathrm{Symmetry\ II\ constraint}}$")
    if title:
        ax.set_title(title)
    ax.set_ylim(bottom=0.0)
    # Anchor x-axis at 0 (or 1 on log scale, since log(0) = -∞).
    ax.set_xlim(left=1 if log_x else 0)
    ax.grid(True, alpha=0.3, which="both" if log_x else "major")
    if tracks or mark_rank is not None or (show_mark_ranks and mark_ranks):
        ax.legend(loc="best")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _effective_rank(s: torch.Tensor, rtol: float) -> int:
    if s.numel() == 0:
        return 0
    return int((s > s.max() * rtol).sum().item())


def _effective_rank_energy(s: torch.Tensor, fraction: float = 0.99) -> int:
    """Smallest k such that the top-k singular values capture ``fraction``
    of the squared Frobenius norm.

    More principled than an ``rtol`` cutoff when the spectrum decays
    smoothly (no clean cliff to find). Equivalent to the "99% energy"
    convention used in classical PCA dimensionality selection.

    Args:
        s: 1-D tensor of singular values (sorted descending).
        fraction: Energy fraction in ``(0, 1]``.

    Returns:
        Effective rank in ``[1, len(s)]`` (or ``0`` if ``s`` is empty).
    """
    if s.numel() == 0:
        return 0
    s32 = s.float()
    energy = (s32 ** 2).cumsum(0)
    target = float(fraction) * float(energy[-1])
    # First index whose cumulative energy reaches the target.
    k = int((energy < target).sum().item()) + 1
    return min(k, int(s.numel()))


def _singular_values(g: torch.Tensor) -> torch.Tensor:
    """Return sorted-descending singular values of ``g.transpose(-2, -1)``.

    Input shape: ``(Batch, InputDim, EmbDim)``.  Output is 1-D of length
    ``min(EmbDim, InputDim * Batch)`` — the same s_c / s_y that
    :func:`normalized_gradient_alignment_metric_svd` returns internally.
    """
    g_2d = g.float().reshape(-1, g.shape[-1])             # (k, D)
    _, s, _ = torch.linalg.svd(g_2d.T, full_matrices=False)
    return s


def main():

    # --- 1. Sanity check
    # load the Joacobian of h_bar with masked topk (16) logits.
    h_bar_g_topk = torch.load(os.path.join(OUT_DIR, f"h_bar_gradients_False_True_16.pt"))
    print(f"Loaded masked h_bar Jacobian: {tuple(h_bar_g_topk.shape)}")
    # Reshape -> (Batch, InputDim, EmbDim).
    h_bar_g_topk = h_bar_g_topk.reshape(h_bar_g_topk.shape[0], -1, h_bar_g_topk.shape[-1])
    # plot and save the jacobian spectra plot.
    out_png = os.path.join(OUT_DIR, f"spectra_h_bar_False_True_16.png")
    s_h_bar_topk = _singular_values(h_bar_g_topk)
    plot_jacobian_spectrum(s_h_bar_topk, out_path=out_png, label="∇h_bar (masked)", expected_drop=16, log_x=True)
    print(f"Saved masked-spectrum plot: {out_png}")

    # --- 1.1 plot of known concept activation values
    activations = torch.load(os.path.join(OUT_DIR, f"known_concept_activations.pt"))
    out_png = os.path.join(OUT_DIR, f"activations_concepts.png")
    plot_concept_activations(activations, out_path=out_png, topk_marker=16)
    print(f"Saved concept-activation plot: {out_png}")

    # --- 1.2 metric with masking
    concepts_g_topk = torch.load(os.path.join(OUT_DIR, f"concept_gradients_False_True_16.pt"))
    print(f"Loaded masked concept Jacobian: {tuple(concepts_g_topk.shape)}")
    concepts_g_topk = concepts_g_topk.reshape(concepts_g_topk.shape[0], -1, concepts_g_topk.shape[-1])
    svd_metric_masked, _, _ = normalized_gradient_alignment_metric_svd(h_bar_g_topk, concepts_g_topk, method="energy", fraction=FRACTION_ENERGY)
    print(f"Symmetry-II metric (SVD) with masking in forward pass (topk = 16) (should be close to 0.): {svd_metric_masked.item():.6f}")

    # --- 2. Plot spectra without masking
    # load the Joacobian of h_bar without masking topk logits.
    h_bar_g = torch.load(os.path.join(OUT_DIR, f"h_bar_gradients_False_False_16.pt"))
    print(f"Loaded unmasked h_bar Jacobian: {tuple(h_bar_g.shape)}")
    # Reshape -> (Batch, InputDim, EmbDim).
    h_bar_g = h_bar_g.reshape(h_bar_g.shape[0], -1, h_bar_g.shape[-1])
    # plot and save the jacobian spectra plot.
    out_png = os.path.join(OUT_DIR, f"spectra_h_bar_False_False_16.png")
    s_h_bar_unmasked = _singular_values(h_bar_g)
    plot_jacobian_spectrum(s_h_bar_unmasked, out_path=out_png, label="∇h_bar (unmasked)", log_x=True)
    print(f"Saved unmasked-spectrum plot: {out_png}")

    # --- 3. metric without masking
    concepts_g = torch.load(os.path.join(OUT_DIR, f"concept_gradients_False_False_16.pt"))
    print(f"Loaded unmasked concept Jacobian: {tuple(concepts_g.shape)}")
    concepts_g = concepts_g.reshape(concepts_g.shape[0], -1, concepts_g.shape[-1])
    svd_metric_unmasked, _, _ = normalized_gradient_alignment_metric_svd(h_bar_g, concepts_g, method="energy", fraction=FRACTION_ENERGY)
    print(f"Symmetry-II metric (SVD) without masking in forward pass (topk = 16): {svd_metric_unmasked.item():.6f}")
    svd_metric_valid, _, _ = normalized_gradient_alignment_metric_svd(h_bar_g, concepts_g_topk, method="energy", fraction=FRACTION_ENERGY)
    print(f"Symmetry-II metric (SVD) validation using masked concept Jacobian (should be same as above): {svd_metric_valid.item():.6f}")

    # --- 4. Metric vs. TOPK across separate experiment runs.
    # Two tracks share the same concept-gradient sweep but use different
    # h_bar Jacobians:
    #   - unmasked: full h_bar (no head masking) → effective rank ~r_star.
    #   - masked:   h_bar with the LM head masked to top-16 concepts → effective rank ≤ 16.
    # The masked track should drop at TOPK = 16 (concept subspace
    # finally large enough to contain the rank-16 masked-h_bar subspace).
    runs_unmasked = []
    runs_masked = []
    for topk in [2, 4, 8, 15, 16, 17, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 33732]:
        concepts_path = os.path.join(OUT_DIR, f"concept_gradients_False_False_{topk}.pt")
        if not os.path.exists(concepts_path):
            print(f"Skipping TOPK={topk}: missing {concepts_path}.")
            continue
        c = torch.load(concepts_path)
        c = c.reshape(c.shape[0], -1, c.shape[-1])
        runs_unmasked.append((topk, h_bar_g, c))
        runs_masked.append((topk, h_bar_g_topk, c))

    # Effective rank of ∂h_bar/∂input across several energy fractions.
    rank_fractions = [0.9, 0.99, 0.999, 0.9999, 0.99999]
    rank_marks: list[tuple[str, int]] = []
    for f in rank_fractions:
        r = _effective_rank_energy(s_h_bar_unmasked, f)
        print(f"Effective rank of ∇h_bar (unmasked, energy={f}): {r}")
        rank_marks.append((fr"$E{{=}}{f}$ ($r{{=}}{r}$)", r))

    r_star = _effective_rank_energy(s_h_bar_unmasked, FRACTION_ENERGY)

    if runs_unmasked:
        # Containment metric — asymmetric, monotonically non-increasing.
        # Masked-h_bar track should drop sharply at TOPK = 16.
        out_png = os.path.join(
            OUT_DIR, f"metric_vs_topk_containment_{FRACTION_ENERGY}.png"
        )
        plot_metric_vs_topk(
            [
                (r"unmasked $\nabla_z \bar h$", runs_unmasked),
                (r"masked $\nabla_z \bar h$  ($K{=}16$)", runs_masked),
            ],
            out_path=out_png,
            method="energy", fraction=FRACTION_ENERGY,
            log_x=True,
            mark_rank=r_star,
            mark_label=r"effective rank($\nabla_z \bar h$)",
            mark_ranks=rank_marks,
            show_mark_ranks=False,
            mark_cmap="Reds",
            metric_fn=normalized_gradient_alignment_metric_containment,
        )
        print(f"Saved containment-metric-vs-topk plot: {out_png}")
    else:
        print("Skipped containment-metric-vs-topk plot: no qualifying caches found.")


if __name__ == "__main__":
    main()
