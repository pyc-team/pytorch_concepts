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


# Relative tolerance for "is this singular value real or numerical noise?".
# bf16 noise floor is ~1e-3 of S.max(); fp32 is ~1e-7.  Use a value above
# the bf16 noise floor when Jacobians came from a bf16 forward pass.
EFFECTIVE_RANK_RTOL = 1e-4


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


def normalized_gradient_alignment_metric_svd(g_c, g_y, *, rtol: float | None = None):
    """SVD-truncated variant of the gradient-subspace alignment metric.

    The QR-based metric collapses on square rank-deficient Jacobians
    (returns ``p = I`` regardless of true rank).  This variant truncates
    columns of ``U`` whose singular value is below ``rtol * s.max()`` so
    ``Q`` has the *numerical* rank of ``g.transpose(1, 2)``.

    Args:
        g_c, g_y: Tensors of shape ``(Batch=1, InputDim, EmbDim)`` with
            matching ``EmbDim``.  ``InputDim`` and effective rank may
            differ.
        rtol: Relative tolerance for singular-value truncation. ``None``
            falls back to NumPy/MATLAB's
            ``max(D, InputDim) * eps(dtype)`` convention.

    Returns:
        Tuple ``(metric, s_c, s_y)`` with metric a scalar tensor and
        ``s_c``/``s_y`` the sorted-descending singular values.
    """
    if g_c.shape[0] != 1 or g_y.shape[0] != 1:
        raise NotImplementedError(
            "SVD-truncation path assumes Batch=1 (ragged ranks per batch "
            "would need a list output)."
        )

    def _proj(g):
        g_2d = g.float().reshape(-1, g.shape[-1])                  # (k, D)
        u, s, _ = torch.linalg.svd(g_2d.T, full_matrices=False)    # u: (D, min(D, k))
        if s.numel() == 0:
            d = g_2d.shape[-1]
            zero = torch.zeros(d, d, dtype=g_2d.dtype, device=g_2d.device)
            return zero, 0, s
        tol = rtol if rtol is not None else (
            max(g_2d.shape) * torch.finfo(g_2d.dtype).eps
        )
        cutoff = float(s.max()) * tol
        keep = s > cutoff
        q = u[:, keep]                                              # (D, rank)
        return q @ q.T, int(keep.sum()), s

    p_c, k_c, s_c = _proj(g_c)
    p_y, k_y, s_y = _proj(g_y)

    raw_loss = torch.linalg.matrix_norm(p_c - p_y, ord='fro')
    if k_c + k_y == 0:
        return raw_loss, s_c, s_y
    max_val = torch.sqrt(torch.tensor(
        float(k_c + k_y), dtype=raw_loss.dtype, device=raw_loss.device,
    ))
    return raw_loss / max_val, s_c, s_y


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

    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = s.detach().cpu().float().numpy()
    plot = ax.loglog if log_x else ax.semilogy
    plot(range(1, len(vals) + 1), vals, marker="o", markersize=3, label=label)
    if expected_drop is not None:
        ax.axvline(
            x=expected_drop + 0.5,
            color="red", linestyle="--", alpha=0.6,
            label=f"expected drop after rank {expected_drop}",
        )
    ax.set_xlabel(
        "singular value index (sorted descending)"
        + (" — log scale" if log_x else "")
    )
    ax.set_ylabel("singular value (log scale)")
    ax.set_title("Jacobian spectrum")
    ax.grid(True, alpha=0.3, which="both")
    if label is not None or expected_drop is not None:
        ax.legend()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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

    vals = (
        activations.detach().cpu().float().reshape(-1)
        .sort(descending=True).values.numpy()
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if log_x and log_y:
        plot = ax.loglog
    elif log_x:
        plot = ax.semilogx
    elif log_y:
        plot = ax.semilogy
    else:
        plot = ax.plot
    plot(range(1, len(vals) + 1), vals, linewidth=1)
    if topk_marker is not None:
        ax.axvline(
            x=topk_marker + 0.5,
            color="red", linestyle="--", alpha=0.6,
            label=f"top-{topk_marker}",
        )
        ax.legend()
    ax.set_xlabel(
        "concept index (sorted descending by activation)"
        + (" — log scale" if log_x else "")
    )
    ax.set_ylabel("activation" + (" (log scale)" if log_y else ""))
    ax.set_title(f"Known concept activations ({len(vals)} concepts)")
    ax.grid(True, alpha=0.3, which="both" if (log_y or log_x) else "major")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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

    fig, ax = plt.subplots(figsize=(7.5, 5))
    plot = ax.loglog if log_x else ax.semilogy
    for s, label, metric in spectra:
        vals = s.detach().cpu().float().numpy()
        full_label = label if metric is None else f"{label} (metric={metric:.4f})"
        plot(range(1, len(vals) + 1), vals, marker="o", markersize=2, label=full_label)
    if expected_drop is not None:
        ax.axvline(
            x=expected_drop + 0.5,
            color="red", linestyle="--", alpha=0.6,
            label=f"expected drop after rank {expected_drop}",
        )
    ax.set_xlabel(
        "singular value index (sorted descending)"
        + (" — log scale" if log_x else "")
    )
    ax.set_ylabel("singular value (log scale)")
    ax.set_title("Jacobian spectra — comparison")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _effective_rank(s: torch.Tensor, rtol: float) -> int:
    if s.numel() == 0:
        return 0
    return int((s > s.max() * rtol).sum().item())


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
    plot_jacobian_spectrum(s_h_bar_topk, out_path=out_png, label="∇h_bar (masked)", expected_drop=16)
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
    svd_metric_masked, _, _ = normalized_gradient_alignment_metric_svd(h_bar_g_topk, concepts_g_topk)
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
    plot_jacobian_spectrum(s_h_bar_unmasked, out_path=out_png, label="∇h_bar (unmasked)")
    print(f"Saved unmasked-spectrum plot: {out_png}")

    # --- 3. metric without masking
    concepts_g = torch.load(os.path.join(OUT_DIR, f"concept_gradients_False_False_16.pt"))
    print(f"Loaded unmasked concept Jacobian: {tuple(concepts_g.shape)}")
    concepts_g = concepts_g.reshape(concepts_g.shape[0], -1, concepts_g.shape[-1])
    svd_metric_unmasked, _, _ = normalized_gradient_alignment_metric_svd(h_bar_g, concepts_g)
    print(f"Symmetry-II metric (SVD) without masking in forward pass (topk = 16): {svd_metric_unmasked.item():.6f}")

    # --- 4. Summary: superpose masked and unmasked spectra with metrics
    out_png = os.path.join(OUT_DIR, f"spectra_summary_16.png")
    plot_jacobian_spectra_comparison(
        [
            (s_h_bar_topk,      "∇h_bar (masked)",   svd_metric_masked.item()),
            (s_h_bar_unmasked,  "∇h_bar (unmasked)", svd_metric_unmasked.item()),
        ],
        out_path=out_png,
        expected_drop=16,
    )
    print(f"Saved summary spectra plot: {out_png}")


if __name__ == "__main__":
    main()
