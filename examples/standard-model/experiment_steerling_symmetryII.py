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
import torch.nn.functional as F
from torch.func import jacrev, jvp

from torch_concepts.steerling import SteerlingMidLevelModel


def normalized_gradient_alignment_metric(g_c, g_y):
    """Normalized Frobenius distance between two gradient subspaces.

    Each input has shape ``(Batch=1, InputDim, EmbDim)`` and is treated
    as ``InputDim`` vectors spanning a subspace of ``R^EmbDim``.
    """
    # 1. Orthonormal bases (QR on the (EmbDim, InputDim) view).
    q_c, _ = torch.linalg.qr(g_c.reshape([1, 1, -1]).transpose(1, 2), mode='reduced')
    q_y, _ = torch.linalg.qr(g_y.reshape([1, 1, -1]).transpose(1, 2), mode='reduced')

    k_c = q_c.shape[2]
    k_y = q_y.shape[2]

    # 2. Projection matrices P = Q @ Q.T.
    p_c = torch.bmm(q_c, q_c.transpose(1, 2))
    p_y = torch.bmm(q_y, q_y.transpose(1, 2))

    raw_loss = torch.linalg.matrix_norm(p_c - p_y, ord='fro')

    # 3. Normalize by the maximum possible distance between two rank-k subspaces.
    max_val = torch.sqrt(torch.tensor(k_c + k_y, dtype=raw_loss.dtype, device=raw_loss.device))
    return raw_loss / max_val


def _compute_jacobian_sliced(inference, query, evidence, target_indices):
    """Reverse-mode Jacobian for a small slice of outputs.

    ``jacrev`` differentiates only with respect to ``evidence["input"]``.
    Everything else in ``evidence`` (K, U, ...) is captured as a constant.
    """
    input_tensor = evidence["input"]
    static = {key: value for key, value in evidence.items() if key != "input"}

    def inference_fn(e_tensor):
        full_evidence = {"input": e_tensor, **static}
        outputs = inference.query(query=query, evidence=full_evidence, return_logits=True)
        return outputs.logits[..., target_indices]

    return jacrev(inference_fn, chunk_size=1)(input_tensor)


def _compute_jacobian_jvp(inference, query, evidence):
    """Forward-mode Jacobian column-by-column via JVP.

    One forward pass per input dimension; no compute graph is retained
    between columns, which keeps peak memory low even when the output
    dimension is large (e.g. ``h_bar`` has 4096 dims).
    """
    from tqdm import tqdm

    input_tensor = evidence["input"]
    static = {key: value for key, value in evidence.items() if key != "input"}

    def f(x):
        full_evidence = {"input": x, **static}
        outputs = inference.query(query=query, evidence=full_evidence, return_logits=True)
        return outputs.logits

    cols = []
    last_jv = None
    for i in tqdm(range(input_tensor.numel()), desc="JVP Jacobian", unit="col"):
        tangent = torch.zeros_like(input_tensor)
        tangent.reshape(-1)[i] = 1.0
        _, jv = jvp(f, (input_tensor,), (tangent,))
        cols.append(jv.detach().cpu())
        last_jv = jv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    jacobian = torch.stack(cols, dim=-1)
    return jacobian.reshape(*last_jv.shape, *input_tensor.shape)


def main():
    out_dir = "steerling_experiment"
    os.makedirs(out_dir, exist_ok=True)

    prompt = (
        "Know that Socrates is older than Plato and Plato is older than Aristotle. "
        "Question: Is Socrates older than Plato?"
    )
    n_new_tokens = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Mid-level model: backbone + concept heads + PGM + inference ─
    model = SteerlingMidLevelModel(
        use_unknown=True, 
        compact=False, 
        use_epsilon_correction=False
    )
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    print(model)

    # ── 2. Prepare last-token hidden as the differentiation point ──────
    input_ids, _, _ = model.prepare_input(prompt, n_new_tokens=n_new_tokens)
    input_ids = input_ids.to(device)
    print(f"\nPrompt: {prompt!r}")
    print(f"Tokens: {tuple(input_ids.shape)}")

    with torch.no_grad():
        hidden = model.backbone(input_ids)             # (1, T, D)
    hidden = hidden[:, -1, :].detach()                  # (1, D)
    hidden.requires_grad_(True)
    print(f"Last-token hidden: {tuple(hidden.shape)}")

    # Full PGM evidence (input + concept embeddings).  K/U are constants —
    # only `input` carries gradients.
    evidence = {
        "input": hidden,
        "K": model.known_embeddings,
        "U": model.unknown_embeddings,
    }

    # ── 3. Yes / No target tokens ──────────────────────────────────────
    tokenizer = model.tokenizer
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    target_indices = torch.tensor([yes_id, no_id], device=device)
    print(f"\nToken IDs: Yes={yes_id}, No={no_id}")

    output = model.inference.query(["new_token"], evidence=evidence, return_logits=True)
    relevant = output.logits[0, target_indices]
    rel_probs = F.softmax(relevant, dim=0)
    print(f"P(Yes | prompt) ≈ {rel_probs[0].item():.2%}")
    print(f"P(No  | prompt) ≈ {rel_probs[1].item():.2%}")

    # ── 4. Jacobian of Yes/No logits w.r.t. input hidden ───────────────
    cache_in = os.path.join(out_dir, "input_gradients.pt")
    if os.path.exists(cache_in):
        print(f"\nLoading cached input gradients from {cache_in}")
        input_gradients = torch.load(cache_in)
    else:
        print("\nComputing ∂(Yes/No logits) / ∂(input)…")
        input_gradients = _compute_jacobian_sliced(
            model.inference, ["new_token"], evidence, target_indices=target_indices
        )
        torch.save(input_gradients.cpu(), cache_in)
    print(f"Input Jacobian shape: {tuple(input_gradients.shape)}")

    # ── 5. Jacobian of h_bar w.r.t. input hidden ───────────────────────
    cache_hbar = os.path.join(out_dir, "h_bar_gradients.pt")
    if os.path.exists(cache_hbar):
        print(f"\nLoading cached h_bar gradients from {cache_hbar}")
        h_bar_gradients = torch.load(cache_hbar)
    else:
        print("\nComputing ∂(h_bar) / ∂(input) via JVP…")
        h_bar_gradients = _compute_jacobian_jvp(model.inference, ["h_bar"], evidence)
        torch.save(h_bar_gradients.cpu(), cache_hbar)
    print(f"h_bar Jacobian shape: {tuple(h_bar_gradients.shape)}")

    # ── 6. Symmetry-II metric ──────────────────────────────────────────
    metric = normalized_gradient_alignment_metric(
        h_bar_gradients.cpu().float(),
        input_gradients.cpu().float(),
    )
    print(f"\nSymmetry-II metric: {metric.item():.8f}")

    with open(os.path.join(out_dir, "results.csv"), "a") as f:
        f.write(f"h_bar,{metric.item()}\n")


if __name__ == "__main__":
    main()
