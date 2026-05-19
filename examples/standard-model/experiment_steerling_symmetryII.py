"""Symmetry-II gradient-alignment experiment for Steerling.

Measures how aligned the gradient subspace of the ``Yes``/``No`` token
logits is with the gradient subspace of the reconstructed latent
``h_bar``, both taken with respect to the last-token backbone hidden
state.  A perfectly faithful concept bottleneck would give zero
normalized Frobenius distance between the two projection matrices.

The PGM is built once by :class:`SteerlingMidLevelModel`; this script
only sits on top of ``model.inference`` and pulls Jacobians out via
``torch.func``.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacrev, jvp, vjp
from tqdm.auto import tqdm

from torch_concepts.steerling import SteerlingMidLevelModel


class TopKMaskedHead(nn.Module):
    """Wrap a concept head so that non-top-k logits are pushed to ~-inf.

    The wrapper returns logits where positions in ``topk_indices`` keep
    their original value and all other positions are replaced with a
    large negative number (``mask_logit``). After the inference engine's
    sigmoid activation, non-top-k concepts saturate to ~0 and contribute
    nothing to downstream CPDs (``k_hat``, ``h_bar``, ``new_token``).
    The mask is constant, so gradient still flows cleanly through the
    top-k logits to ``input``; non-top-k gradients vanish through the
    sigmoid saturation tail (expected, since they are "off").
    """

    def __init__(self, base: nn.Module, topk_indices: torch.Tensor, mask_logit: float = -1e4):
        super().__init__()
        self.base = base
        self.register_buffer("topk_indices", topk_indices.to(torch.long))
        self.mask_logit = mask_logit

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        logits = self.base(latent)
        mask = torch.full_like(logits, self.mask_logit)
        mask[..., self.topk_indices] = 0.0          # 0 for top-k, -inf elsewhere
        return logits + mask                        # top-k unchanged; rest pushed down

    # Forward attribute access to the underlying head so callers that look
    # at e.g. `.factorize`, `.use_attention`, `.out_concepts` keep working.
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)


def normalized_gradient_alignment_metric(g_c, g_y):
    """Normalized Frobenius distance between two gradient subspaces.

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

    The QR-based metric is only faithful when ``InputDim < EmbDim`` and
    the row vectors are linearly independent.  When the Jacobian is
    "tall and rank-deficient" — e.g. ``h_bar`` has 4096 rows in
    ``R^4096`` but lives in a rank-16 subspace because only 16 concepts
    are active — reduced QR returns an orthonormal-completed basis with
    ``min(D, InputDim)`` columns and the projection collapses to ``I``.

    This variant uses singular values to drop columns of ``U`` whose
    ``s`` is below ``rtol * s.max()``, so ``Q`` has the *numerical* rank
    of ``g.transpose(1, 2)``.  The projection ``Q @ Q.T`` is then the
    true column-space projector.

    Args:
        g_c, g_y: Tensors of shape ``(Batch, InputDim, EmbDim)`` with
            matching ``Batch`` (assumed 1 below) and ``EmbDim``.
            ``InputDim`` and effective rank may differ.
        rtol: Relative tolerance for singular-value truncation. ``None``
            falls back to NumPy/MATLAB's
            ``max(D, InputDim) * eps(dtype)`` convention.
    """
    if g_c.shape[0] != 1 or g_y.shape[0] != 1:
        raise NotImplementedError(
            "SVD-truncation path assumes Batch=1 (ragged ranks per batch "
            "would need a list output)."
        )

    def _proj(g):
        # g: (1, InputDim, EmbDim) → flatten to (k, D), then SVD on (D, k).
        g_2d = g.float().reshape(-1, g.shape[-1])                  # (k, D)
        u, s, _ = torch.linalg.svd(g_2d.T, full_matrices=False)    # u: (D, min(D, k))
        if s.numel() == 0:
            d = g_2d.shape[-1]
            return torch.zeros(d, d, dtype=g_2d.dtype, device=g_2d.device), 0
        tol = rtol if rtol is not None else (
            max(g_2d.shape) * torch.finfo(g_2d.dtype).eps
        )
        cutoff = float(s.max()) * tol
        keep = s > cutoff
        q = u[:, keep]                                              # (D, rank)
        return q @ q.T, int(keep.sum())

    p_c, k_c = _proj(g_c)
    p_y, k_y = _proj(g_y)

    raw_loss = torch.linalg.matrix_norm(p_c - p_y, ord='fro')
    if k_c + k_y == 0:
        return raw_loss  # both Jacobians vanish → 0/0; report 0
    max_val = torch.sqrt(torch.tensor(
        float(k_c + k_y), dtype=raw_loss.dtype, device=raw_loss.device,
    ))
    return raw_loss / max_val


def _compute_jacobian(
    inference, 
    query, 
    evidence, 
    *, 
    chunk_size=None, 
    target_indices=None, 
    progress=False
):
    """Reverse-mode Jacobian of ``query`` outputs w.r.t. ``evidence["input"]``.

    Differentiates only w.r.t. ``evidence["input"]``; all other entries in
    ``evidence`` (K, U, ...) are captured as constants so they don't enter
    the autograd graph.

    Args:
        inference: PyC inference engine (e.g. ``model.inference``).
        query: List of variable names to differentiate.
        evidence: Evidence dict; ``"input"`` is the tensor we differentiate
            with respect to.
        chunk_size: Forwarded to ``torch.func.jacrev``.  Number of output
            rows whose backward is run per chunk.  ``None`` runs all rows
            at once (fastest, highest peak memory).  ``1`` runs one
            backward per output dim (slowest, lowest peak memory).  Use a
            small integer (e.g. ``32``) to trade speed for memory.
        target_indices: Optional 1-D index tensor.  When set, the output
            is sliced ``logits[..., target_indices]`` *before*
            differentiation — essential when the raw output is huge
            (e.g. full vocab) but only a few rows matter (e.g. Yes/No
            target tokens).  Reverse-mode cost scales with the post-slice
            output dimension.
        progress: If ``True``, run the row-by-row backward loop in Python
            with a ``tqdm`` progress bar (mathematically equivalent to
            ``jacrev(..., chunk_size=1)`` but visible).  Ignores
            ``chunk_size`` — always one output row per backward.
    """
    input_tensor = evidence["input"]
    static = {key: value for key, value in evidence.items() if key != "input"}

    def inference_fn(input_):
        full_evidence = {"input": input_, **static}
        out = inference.query(query=query, evidence=full_evidence, return_logits=True)
        logits = out.logits
        if target_indices is not None:
            logits = logits[..., target_indices]
        return logits

    if progress:
        output, vjp_fn = vjp(inference_fn, input_tensor)
        output_shape = output.shape
        n_out = output.numel()
        rows = []
        for i in tqdm(range(n_out), desc=f"jacrev[{','.join(query)}]"):
            cot = torch.zeros(n_out, dtype=output.dtype, device=output.device)
            cot[i] = 1.0
            (grad,) = vjp_fn(cot.reshape(output_shape))
            rows.append(grad.detach())
        return torch.stack(rows, dim=0).reshape(*output_shape, *input_tensor.shape)

    return jacrev(inference_fn, chunk_size=chunk_size)(input_tensor)


# def _compute_jacobian_sliced(inference, query, evidence, target_indices):
#     """Reverse-mode Jacobian for a small slice of outputs.

#     ``jacrev`` differentiates only with respect to ``evidence["input"]``.
#     Everything else in ``evidence`` (K, U, ...) is captured as a constant.
#     """
#     input_tensor = evidence["input"]
#     static = {key: value for key, value in evidence.items() if key != "input"}

#     def inference_fn(e_tensor):
#         full_evidence = {"input": e_tensor, **static}
#         outputs = inference.query(query=query, evidence=full_evidence, return_logits=True)
#         return outputs.logits[..., target_indices]

#     return jacrev(inference_fn, chunk_size=1)(input_tensor)


# def _compute_jacobian_jvp(inference, query, evidence):
#     """Forward-mode Jacobian column-by-column via JVP.

#     One forward pass per input dimension; no compute graph is retained
#     between columns, which keeps peak memory low even when the output
#     dimension is large (e.g. ``h_bar`` has 4096 dims).
#     """
#     from tqdm import tqdm

#     input_tensor = evidence["input"]
#     static = {key: value for key, value in evidence.items() if key != "input"}

#     def f(x):
#         full_evidence = {"input": x, **static}
#         outputs = inference.query(query=query, evidence=full_evidence, return_logits=True)
#         return outputs.logits

#     cols = []
#     last_jv = None
#     for i in tqdm(range(input_tensor.numel()), desc="JVP Jacobian", unit="col"):
#         tangent = torch.zeros_like(input_tensor)
#         tangent.reshape(-1)[i] = 1.0
#         _, jv = jvp(f, (input_tensor,), (tangent,))
#         cols.append(jv.detach().cpu())
#         last_jv = jv
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     jacobian = torch.stack(cols, dim=-1)
#     return jacobian.reshape(*last_jv.shape, *input_tensor.shape)


def main():
    out_dir = "steerling_experiment"
    os.makedirs(out_dir, exist_ok=True)
    
    prompt = (
        "Know that Socrates is older than Plato and Plato is older than Aristotle. "
        "Question: Is Socrates older than Plato?"
    )
    n_new_tokens = 1
    use_unknown = False  # Bool. Whether to use the "unknown" token in the concept bottleneck.
    mask_topk = 16  # None | int. Whether to mask all but the top-k most attended known concept in the bottleneck.

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Mid-level model: backbone + concept heads + PGM + inference ─
    _prev_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = SteerlingMidLevelModel(
            use_unknown=use_unknown,
            use_epsilon_correction=False
        )
    finally:
        torch.set_default_dtype(_prev_default_dtype)
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    print(model)

    # ── 2. Prepare last-token hidden as the differentiation point ──────
    input_ids, _, _ = model.prepare_input(prompt, n_new_tokens=n_new_tokens)
    input_ids = input_ids.to(device)
    print(f"\nPrompt: {prompt!r}")
    print(f"Tokens: {tuple(input_ids.shape)}")

    # prepare the evidence dict for inference; we only need the "input" key for gradients
    # only take the last token's hidden state
    evidence = model._evidence(input_ids)
    evidence["input"] = evidence["input"][:, -1, :].detach()  # (1, D)
    evidence["input"].requires_grad_(True)
    print(f"Last-token hidden: {tuple(evidence["input"].shape)}")

    topk_indices = None
    if mask_topk is not None:
        # 1. Pick top-k known concepts at the current input (no autograd).
        with torch.no_grad():
            out = model.inference.query(model.known_names, evidence=evidence, return_logits=True)
            _, topk_indices = torch.topk(out.logits[0], k=mask_topk)
        topk_names = [model.known_names[i] for i in topk_indices.cpu().tolist()]
        print(f"Top-{mask_topk} known concepts: {topk_names}")

        # 2. Swap the concept head for a top-k-masked wrapper.
        # Evidence injection on individual concepts of a shared CPD is
        # silently ignored by the inference engine, so we mask *inside*
        # the parametrization instead. Non-top-k logits get pushed to
        # -1e4 so sigmoid saturates to ~0 and they contribute nothing to
        # k_hat / h_bar / new_token. Top-k logits are untouched and still
        # carry gradient through to `input`.
        masked_head = TopKMaskedHead(model.known_concept_head, topk_indices).to(
            device=device, dtype=evidence["input"].dtype
        )
        model.known_concept_head = masked_head
        k_cpd = model.pgm.get_module_of_concept(model.known_names[0])
        k_cpd.parametrization = masked_head

        # sanity checks
        print(f"Is the same module? {model.pgm.get_module_of_concept(model.known_names[0]) is k_cpd}")
        all_idx = torch.arange(len(model.known_names), device=device)
        non_topk_indices  = all_idx[~torch.isin(all_idx, topk_indices)]   # 1-D
        out = model.inference.query(model.known_names, evidence=evidence, return_logits=True).logits
        print(f"Top-k logit (should be original value): {out[0, topk_indices]}")
        print(f"Non-top-k logit (should be ~{masked_head.mask_logit}): {out[0, non_topk_indices[0:20]]}")

    # ── 3. Yes / No target tokens ──────────────────────────────────────
    # ── 4. Jacobian of Yes/No logits w.r.t. input hidden ───────────────
    # tokenizer = model.tokenizer
    # yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    # no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    # target_indices = torch.tensor([yes_id, no_id], device=device)
    # print(f"\nToken IDs: Yes={yes_id}, No={no_id}")

    # output = model.inference.query(["new_token"], evidence=evidence, return_logits=True, mask_topk=mask_topk)
    # relevant = output.logits[0, target_indices]
    # rel_probs = F.softmax(relevant, dim=0)
    # print(f"P(Yes | prompt) ≈ {rel_probs[0].item():.2%}")
    # print(f"P(No  | prompt) ≈ {rel_probs[1].item():.2%}")

    # cache_in = os.path.join(out_dir, f"input_gradients_{use_unknown}_{mask_topk}.pt")
    # if os.path.exists(cache_in):
    #     print(f"\nLoading cached input gradients from {cache_in}")
    #     input_gradients = torch.load(cache_in)
    # else:
    #     print("\nComputing ∂(Yes/No logits) / ∂(input)…")
    #     input_gradients = _compute_jacobian(
    #         model.inference, 
    #         ["new_token"], 
    #         evidence, 
    #         target_indices=target_indices
    #     )
    #     torch.save(input_gradients.cpu(), cache_in)
    # print(f"Input Jacobian shape: {tuple(input_gradients.shape)}")

    # ── 5. Jacobian of concepts w.r.t. input hidden ───────────────
    cache_in = os.path.join(out_dir, f"concept_gradients_{use_unknown}_{mask_topk}.pt")
    if os.path.exists(cache_in):
        print(f"\nLoading cached input gradients from {cache_in}")
        concept_gradients = torch.load(cache_in)
    else:
        print("\nComputing ∂(c) / ∂(input)…")
        concept_gradients = _compute_jacobian(
            model.inference, 
            model.known_names, 
            evidence, 
            chunk_size=1,
            progress=True,
            target_indices=topk_indices if mask_topk is not None else None
        )
        torch.save(concept_gradients.cpu(), cache_in)
    print(f"Concept Jacobian shape: {tuple(concept_gradients.shape)}")

    # ── 6. Jacobian of h_bar w.r.t. input hidden ───────────────────────
    cache_hbar = os.path.join(out_dir, f"h_bar_gradients_{use_unknown}_{mask_topk}.pt")
    if os.path.exists(cache_hbar):
        print(f"\nLoading cached h_bar gradients from {cache_hbar}")
        h_bar_gradients = torch.load(cache_hbar)
    else:
        print("\nComputing ∂(h_bar) / ∂(input)…")
        # h_bar has 4096 output dims — chunk_size=1 caps peak memory at one
        # backward per output row at the cost of more sequential passes.
        h_bar_gradients = _compute_jacobian(
            model.inference,
            ["h_bar"],
            evidence,
            chunk_size=1,
            progress=True
        )
        torch.save(h_bar_gradients.cpu(), cache_hbar)
    print(f"h_bar Jacobian shape: {tuple(h_bar_gradients.shape)}")

    # ── 6. Symmetry-II metric ──────────────────────────────────────────
    # Collapse the Jacobian's intermediate singleton dims so the metric
    # sees (Batch, InputDim, EmbDim).  _compute_jacobian returns
    # (*output_shape, *input_shape), which for last-token evidence is
    # (1, k, 1, EmbDim) — fold the leading dims into InputDim.
    h_bar_g = h_bar_gradients.reshape(h_bar_gradients.shape[0], -1, h_bar_gradients.shape[-1])
    concept_g = concept_gradients.reshape(concept_gradients.shape[0], -1, concept_gradients.shape[-1])

    # metric = normalized_gradient_alignment_metric(h_bar_g, concept_g)
    metric = normalized_gradient_alignment_metric_svd(h_bar_g, concept_g)
    
    print(f"\nSymmetry-II metric: {metric.item():.8f}")

    with open(os.path.join(out_dir, "results.csv"), "a") as f:
        f.write(f"h_bar,{metric.item()}\n")


if __name__ == "__main__":
    main()
