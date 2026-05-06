import os

import torch
from torch.distributions import RelaxedBernoulli, RelaxedOneHotCategorical
from torch_concepts.distributions import Delta
import torch.nn.functional as F
from torch.func import jacrev, jvp, vmap

from torch_concepts import ConceptVariable, LatentVariable
from torch_concepts.nn import ParametricCPD, ProbabilisticModel, DeterministicInference

from torch_concepts.steerling import (
    SteerlingBackbone, SteerlingLatentToConcept, SteerlingLMHead,
    SteerlingConceptsToLatentEmbeddings, SteerlingLatentToLatentFusion,
    load_steerling_concept_names,
    prepare_steerling_evidence, prepare_generation_sequence,
)


def normalized_gradient_alignment_metric(g_c, g_y):
    """
    g_c, g_y shape: (Batch=1, InputDim, EmbDim)
    We treat InputDim as the number of vectors spanning the subspace.
    """
    # 1. Get Orthonormal Bases
    # We want the basis for the space spanned by the InputDim vectors.
    # QR expects (M, N) and returns Q (M, K).
    # We transpose to (EmbDim, InputDim) to find the basis in EmbDim space.
    q_c, _ = torch.linalg.qr(g_c.reshape([1, 1, -1]).transpose(1, 2), mode='reduced')
    q_y, _ = torch.linalg.qr(g_y.reshape([1, 1, -1]).transpose(1, 2), mode='reduced')

    # k is the number of basis vectors (rank)
    k_c = q_c.shape[2]
    k_y = q_y.shape[2]

    # 3. Compute the Projection Matrices: P = Q @ Q.T
    # This matrix represents the subspace spanned by the gradients
    p_c = torch.bmm(q_c, q_c.transpose(1, 2))
    p_y = torch.bmm(q_y, q_y.transpose(1, 2))

    # 3. Frobenius Norm of the difference
    raw_loss = torch.linalg.matrix_norm(p_c - p_y, ord='fro')

    # 4. Normalization
    # The max distance between two subspaces of rank k_c and k_y
    # is sqrt(k_c + k_y) if they are completely orthogonal.
    max_val = torch.sqrt(torch.tensor(k_c + k_y, dtype=raw_loss.dtype, device=raw_loss.device))

    return raw_loss / max_val


# def _compute_jacobian(inference, query, evidence):
#     def inference_fn(q, e):
#         outputs = inference.query(q, evidence=e, return_logits=True)
#         return outputs.logits

#     jacobian_op = vmap(jacrev(inference_fn))
#     return jacobian_op(query, evidence)

def _compute_jacobian_sliced(inference, query, evidence, target_indices):
    """Efficient Jacobian for a small slice of outputs (e.g., Yes/No logits).
    Runs one backward pass per element in target_indices."""
    input_tensor = evidence["input"]

    def inference_fn(e_tensor):
        outputs = inference.query(query=query, evidence={"input": e_tensor}, return_logits=True)
        return outputs.logits[..., target_indices]

    return jacrev(inference_fn, chunk_size=1)(input_tensor)


def _compute_jacobian_jvp(inference, query, evidence):
    """Jacobian via forward-mode AD (JVP), one column at a time.
    Avoids vmap and graph retention — much lower peak memory than jacrev.
    Runs one forward pass per input dimension with no compute graph stored."""
    input_tensor = evidence["input"]

    def f(x):
        outputs = inference.query(query=query, evidence={"input": x}, return_logits=True)
        return outputs.logits

    cols = []
    last_jv = None
    from tqdm import tqdm
    for i in tqdm(range(input_tensor.numel()), desc="JVP Jacobian", unit="col"):
        tangent = torch.zeros_like(input_tensor)
        tangent.reshape(-1)[i] = 1.0
        _, jv = jvp(f, (input_tensor,), (tangent,))
        cols.append(jv.detach().cpu())
        last_jv = jv
        torch.cuda.empty_cache()

    # stack along new last dim: (*out_shape, in_numel), then reshape to (*out_shape, *in_shape)
    J = torch.stack(cols, dim=-1)
    return J.reshape(*last_jv.shape, *input_tensor.shape)


# def _compute_jacobian_full(inference, query, evidence):
#     """Full Jacobian for moderate-sized outputs (e.g., h_bar with dim=4096).
#     Runs one backward pass per output element."""
#     input_tensor = evidence["input"]

#     def inference_fn(e_tensor):
#         outputs = inference.query(query=query, evidence={"input": e_tensor}, return_logits=True)
#         return outputs.logits

#     return jacrev(inference_fn, chunk_size=1)(input_tensor)



def main():

    os.makedirs("steerling_experiment", exist_ok=True)
    n_unsup = 101196
    emb_dim = 4096

    prompt = ("Know that Socrates is older than Plato and Plato is older than Aristotle. "
              "Question: Is Socrates older than Plato?")
    n_new_tokens = 1

    # ── 1. Load backbone and prepare evidence ──────────────────────────
    backbone = SteerlingBackbone(pretrained=True, freeze=True, device="cuda")
    data = prepare_steerling_evidence(backbone, prompt, n_new_tokens)
    input_ids = data["input_ids"]       # (1, T_prompt + n_new_tokens)
    hidden = data["hidden"]             # (1, T_prompt + n_new_tokens, 4096)
    print(f"Prompt: {prompt!r}")
    print(f"Hidden states: {hidden.shape}")

    # ── 2. Concept encoder heads ───────────────────────────────────────
    lm_head = SteerlingLMHead(pretrained=True, freeze=True)
    lm_head.eval()


    # ── 3. Variables ───────────────────────────────────────────────────
    latent = LatentVariable("input", size=emb_dim, distribution=Delta)

    # concepts
    sup_concepts_names = load_steerling_concept_names()
    k = ConceptVariable(sup_concepts_names, distribution=RelaxedBernoulli)
    unsup_concepts_names = [f"unsup_{i}" for i in range(n_unsup)]
    u = ConceptVariable(unsup_concepts_names, distribution=RelaxedBernoulli)

    # concept-based hidden state
    k_hat = LatentVariable("k_hat", size=emb_dim, distribution=Delta)
    u_hat = LatentVariable("u_hat", size=emb_dim, distribution=Delta)
    # fused concept latent 
    h_bar = LatentVariable("h_bar", size=emb_dim, distribution=Delta)

    # token prediction
    new_token = ConceptVariable("new_token", size=lm_head.vocab_size, distribution=RelaxedOneHotCategorical)


    # ── 4. CPDs ────────────────────────────────────────────────────────
    latent_cpd = ParametricCPD("input", parents=[], parametrization=torch.nn.Identity())
    
    # parametrize concepts with the pretrained concept encoder
    sup_concepts_head = SteerlingLatentToConcept(pretrained=True, freeze=True, topk=None)
    sup_concepts_head.eval()
    k_cpd = ParametricCPD(sup_concepts_names, parents=["input"], 
                          parametrization=sup_concepts_head,
                          shared=True, shared_name="sup_concepts")
    unsup_concepts_head = SteerlingLatentToConcept(pretrained=True, freeze=True, is_unknown=True, topk=None)
    unsup_concepts_head.eval()
    u_cpd = ParametricCPD(unsup_concepts_names, parents=["input"], 
                          parametrization=unsup_concepts_head,
                          shared=True, shared_name="unsup_concepts")
    
    # intermediate latent contributions:
    # k_hat = known_concepts @ known_embeddings
    # u_hat = unknown_concepts @ unknown_embeddings
    k_hat_cpd = ParametricCPD("k_hat", parents=sup_concepts_names, 
                              parametrization=SteerlingConceptsToLatentEmbeddings(embeddings=sup_concepts_head.get_embeddings()))
    u_hat_cpd = ParametricCPD("u_hat", parents=unsup_concepts_names, 
                              parametrization=SteerlingConceptsToLatentEmbeddings(embeddings=unsup_concepts_head.get_embeddings()))

    # fuse intermediate latents into concept_latent
    h_bar_cpd = ParametricCPD("h_bar", parents=["k_hat", "u_hat"],
                              parametrization=SteerlingLatentToLatentFusion(latent_dim=emb_dim))
    
    new_token_cpd = ParametricCPD("new_token", parents=["h_bar"], parametrization=lm_head)

    # ── Query new tokens one at a time ─────────────────────
    # Build full PGM with token prediction
    concept_model = ProbabilisticModel(
        variables=[latent] + k + u + [k_hat, u_hat, h_bar, new_token],
        factors=[
            latent_cpd,
            k_cpd,
            u_cpd,
            k_hat_cpd,
            u_hat_cpd,
            h_bar_cpd,
            new_token_cpd,
        ],
    )
    inference = DeterministicInference(concept_model)

    # ── Get token IDs for "Yes" and "No" ─────────────────────
    print("\nPreparing target token IDs for 'Yes' and 'No'...")
    tokenizer = backbone.tokenizer
    assert tokenizer is not None, "Backbone tokenizer should be initialized"
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    print(f"Token ID for 'Yes': {yes_id}, Token ID for 'No': {no_id}")
    target_indices = torch.tensor([yes_id, no_id], device=hidden.device)
    print(f"Target indices for 'Yes': {target_indices}")

    # ── Gradient of new token probabilities w.r.t. input hidden states ─────────────────────
    # Infer the probabilities of "Yes" and "No" based on the hidden states
    print(f"\nQuerying new token probabilities for 'Yes' and 'No'...")
    hidden = backbone(input_ids).float()
    hidden = hidden[:, -1, :] # focus on the last token's
    hidden.requires_grad_(True)  # Enable gradients for hidden states
    output = inference.query(["new_token"], evidence={"input": hidden}, return_logits=True)
    print(f"Predicted logits for 'Yes': {output.logits.shape}")

    # This slice maintains the gradient path back to the model weights
    print(f"\nExtracting relevant logits for 'Yes' and 'No'...")
    relevant_logits = output.logits[0, target_indices]
    relative_probs = F.softmax(relevant_logits, dim=0)
    prob_yes_rel = relative_probs[0]
    print(f"Relative Probability 'Yes': {prob_yes_rel.item():.2%}")
    print(f"Relative Probability 'No':  {(1 - prob_yes_rel).item():.2%}")

    # Compute gradients of the "Yes" probability w.r.t. input hidden states
    input_gradients_path = os.path.join("steerling_experiment", "input_gradients.pt")
    if os.path.exists(input_gradients_path):
        print(f"\nLoading cached input gradients from {input_gradients_path}...")
        input_gradients = torch.load(input_gradients_path)
    else:
        print(f"\nComputing gradients of 'Yes' probability w.r.t. input hidden states...")
        input_gradients = _compute_jacobian_sliced(inference, ["new_token"], evidence={"input": hidden}, target_indices=target_indices)
        torch.save(input_gradients.cpu(), input_gradients_path)
    print(f"Input gradients type: {type(input_gradients)}")
    print(f"Input gradients: {input_gradients}")
    print("Gradient shape (should match hidden states):")
    print(f"  Gradients: {input_gradients.shape}")
    print(f"  Hidden states: {hidden.shape}")

    # ── Gradient of h_bar w.r.t. input hidden states ─────────────────────
    h_bar_gradients_path = os.path.join("steerling_experiment", "h_bar_gradients.pt")
    if os.path.exists(h_bar_gradients_path):
        print(f"\nLoading cached h_bar gradients from {h_bar_gradients_path}...")
        h_bar_gradients = torch.load(h_bar_gradients_path)
    else:
        print(f"\nComputing gradients of h_bar w.r.t. input hidden states...")
        h_bar_gradients = _compute_jacobian_jvp(inference, ["h_bar"], evidence={"input": hidden})
        torch.save(h_bar_gradients.cpu(), h_bar_gradients_path)
    print("h_bar Gradient shape (should match hidden states):")
    print(f"  h_bar Gradients: {h_bar_gradients.shape}")
    print(f"  Hidden states: {hidden.shape}")

    # Compute the Symmetry II Metric (Gradient Alignment Loss)
    symmetryII_metric = normalized_gradient_alignment_metric(h_bar_gradients.cpu().float(), input_gradients.cpu().float())
    print(f"SymmetryII metric: {symmetryII_metric.item():.8f}")

    with open(os.path.join("steerling_experiment", "results.csv"), "a") as f:
        f.write(f"h_bar,{symmetryII_metric.item()}\n")






if __name__ == "__main__":
    main()
