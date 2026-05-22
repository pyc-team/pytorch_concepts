"""Symmetry-II Jacobian-computation pass for Steerling.

Builds the mid-level model, optionally masks all but the top-k known
concepts, and computes / caches the Jacobians needed by the symmetry-II
analysis:

* ``concept_gradients_{use_unknown}_{mask_topk}.pt`` — ``∂concepts/∂input``
  for the top-k known concepts (shape ``(1, k, 1, D)``).
* ``h_bar_gradients_{use_unknown}_{mask_topk}.pt``   — ``∂h_bar/∂input``
  (shape ``(1, D, 1, D)``).

The analysis (subspace alignment metrics + spectrum plot) lives in
:mod:`analyze_steerling_symmetryII` and reads these caches.  Keep
``USE_UNKNOWN``/``MASK_TOPK`` in sync across the two files so the cache
filenames match.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacrev, jvp, vjp
from tqdm.auto import tqdm

from torch_concepts.steerling import SteerlingMidLevelModel

# Keep these in sync with analyze_steerling_symmetryII.py so the cache
# filenames match.
OUT_DIR = "steerling_experiment"
USE_UNKNOWN = False # Bool. Whether to use the "unknown" token in the concept bottleneck.
# TOPK = 16           # Int. Number of top concepts to keep when masking the concept head.
MASK_TOPK = False   # Bool. Whether to mask all but the top-k most attended known concept in the bottleneck. 


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
        for i in tqdm(range(n_out), desc=f"jacrev"):
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


def main(TOPK: int):
    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    
    prompt = (
        "Know that Socrates is older than Plato and Plato is older than Aristotle. "
        "Question: Is Socrates older than Plato?"
    )
    n_new_tokens = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Mid-level model: backbone + concept heads + PGM + inference ─
    _prev_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = SteerlingMidLevelModel(
            use_unknown=USE_UNKNOWN,
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

    # 1. Single query of all known concepts (no autograd).
    #    Returns both raw logits (for top-k selection) and probs (post-sigmoid
    #    activations).  Saved for downstream analysis (concept ranking, mask
    #    design, etc.) — independent of MASK_TOPK so the file isn't suffixed
    #    by it.
    with torch.no_grad():
        out = model.inference.query(model.known_names, evidence=evidence, return_logits=True)
    cache_act = os.path.join(OUT_DIR, f"known_concept_activations.pt")
    if not os.path.exists(cache_act):
        torch.save(out.probs.cpu(), cache_act)
        print(f"Saved known concept activations ({tuple(out.probs.shape)}): {cache_act}")

    # 2. Pick top-k from the same query.
    _, topk_indices = torch.topk(out.logits[0], k=TOPK)
    topk_names = [model.known_names[i] for i in topk_indices.cpu().tolist()]
    print(f"Top-{TOPK} known concepts: {topk_names}")

    if MASK_TOPK:
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
    cache_in = os.path.join(OUT_DIR, f"concept_gradients_{USE_UNKNOWN}_{MASK_TOPK}_{TOPK}.pt")
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
            target_indices=topk_indices
        )
        torch.save(concept_gradients.cpu(), cache_in)
    print(f"Concept Jacobian shape: {tuple(concept_gradients.shape)}")

    # # ── 6. Jacobian of h_bar w.r.t. input hidden ───────────────────────
    # cache_hbar = os.path.join(OUT_DIR, f"h_bar_gradients_{USE_UNKNOWN}_{MASK_TOPK}_{TOPK}.pt")
    # if os.path.exists(cache_hbar):
    #     print(f"\nLoading cached h_bar gradients from {cache_hbar}")
    #     h_bar_gradients = torch.load(cache_hbar)
    # else:
    #     print("\nComputing ∂(h_bar) / ∂(input)…")
    #     # h_bar has 4096 output dims — chunk_size=1 caps peak memory at one
    #     # backward per output row at the cost of more sequential passes.
    #     h_bar_gradients = _compute_jacobian(
    #         model.inference,
    #         ["h_bar"],
    #         evidence,
    #         chunk_size=1,
    #         progress=True
    #     )
    #     torch.save(h_bar_gradients.cpu(), cache_hbar)
    # print(f"h_bar Jacobian shape: {tuple(h_bar_gradients.shape)}")

    print(
        f"\nCached Jacobians in {OUT_DIR}/. "
        "Run analyze_steerling_symmetryII.py to compute metrics and plot spectra."
    )


if __name__ == "__main__":
    topk_values = [2, 4, 8, 15, 16, 17, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 33732]
    for value in topk_values:
        print(f"\n\n=== Running with TOPK={value} ===")
        main(TOPK=value)
