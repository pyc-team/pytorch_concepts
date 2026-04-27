import torch
from torch.distributions import Bernoulli, Categorical
from torch_concepts.distributions import Delta

from torch_concepts import ConceptVariable, LatentVariable
from torch_concepts.nn import ParametricCPD, ProbabilisticModel, DeterministicInference

from torch_concepts.steerling import (
    SteerlingBackbone, SteerlingLatentToConcept, SteerlingLMHead,
    SteerlingConceptsToLatentEmbeddings, SteerlingLatentToLatentFusion,
    load_steerling_concept_names,
    prepare_steerling_evidence,
)


def main():

    n_unsup = 101196
    emb_dim = 4096

    prompt = "To improve research in interpretable machine learning, we need"
    n_new_tokens = 20

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
    k = ConceptVariable(sup_concepts_names, distribution=Bernoulli)
    unsup_concepts_names = [f"unsup_{i}" for i in range(n_unsup)]
    u = ConceptVariable(unsup_concepts_names, distribution=Bernoulli)

    # concept-based hidden state
    k_hat = LatentVariable("k_hat", size=emb_dim, distribution=Delta)
    u_hat = LatentVariable("u_hat", size=emb_dim, distribution=Delta)
    # fused concept latent 
    h_bar = LatentVariable("h_bar", size=emb_dim, distribution=Delta)

    # token prediction
    new_token = ConceptVariable("new_token", size=lm_head.vocab_size, distribution=Categorical)


    # ── 4. CPDs ────────────────────────────────────────────────────────
    latent_cpd = ParametricCPD("input", parents=[], parametrization=torch.nn.Identity())
    
    # parametrize concepts with the pretrained concept encoder
    sup_concepts_head = SteerlingLatentToConcept(pretrained=True, freeze=True)
    sup_concepts_head.eval()
    k_cpd = ParametricCPD(sup_concepts_names, parents=["input"], 
                          parametrization=sup_concepts_head,
                          shared=True, shared_name="sup_concepts")
    unsup_concepts_head = SteerlingLatentToConcept(pretrained=True, freeze=True, is_unknown=True)
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


    # ── Query to check input is returned correctly ─────────────────────
    concept_model = ProbabilisticModel(
        variables=[latent],
        factors=[latent_cpd],
    )
    inference = DeterministicInference(concept_model)
    assert torch.allclose(inference.query(['input'], evidence={'input':hidden}), 
                          hidden), "Input query does not match hidden states!"

    # ── Query to check concepts are returned correctly ─────────────────────
    concept_model = ProbabilisticModel(
        variables=[latent] + k + u,
        factors=[latent_cpd, k_cpd, u_cpd],
    )
    inference = DeterministicInference(concept_model)
    assert torch.allclose(inference.query(['input'], evidence={'input':hidden}), 
                          hidden), "Input query does not match hidden states!"
    # query a known concept
    print(f"\nQuerying known concept {sup_concepts_names[42]}:")
    print(inference.query([sup_concepts_names[42]], evidence={'input':hidden}))
    # query an unknown concept
    print(f"\nQuerying unknown concept {unsup_concepts_names[1231]}:")
    print(inference.query([unsup_concepts_names[1231]], evidence={'input':hidden}))

    # ── Query to check concept_latent (reconstructed hidden state) ─────────────────────
    concept_model = ProbabilisticModel(
        variables=[latent] + k + u + [k_hat, u_hat, h_bar],
        factors=[
            latent_cpd,
            k_cpd,
            u_cpd,
            k_hat_cpd,
            u_hat_cpd,
            h_bar_cpd,
        ],
    )
    inference = DeterministicInference(concept_model)
    print(f"\nQuerying h_bar:")
    result = inference.query(["h_bar"], evidence={'input':hidden})
    print(f"  shape: {result.shape}")  # should be (1, T, 4096)
    print(f"  values: {result}")  # should be close to hidden states



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

    tokenizer = backbone.tokenizer
    assert tokenizer is not None, "Backbone tokenizer should be initialized"
    mask_id = tokenizer.mask_token_id
    prompt_len = (input_ids[0] != mask_id).sum().item()

    print(f"\nGenerating {n_new_tokens} tokens one at a time:")
    for step in range(n_new_tokens):
        with torch.no_grad():
            # 1. Query token probabilities through the concept bottleneck
            hidden = backbone(input_ids).float()
            token_probs = inference.query(["new_token"], evidence={"input": hidden})

            # 2. Pick the most confident masked position, take argmax
            masked_positions = (input_ids[0] == mask_id).nonzero(as_tuple=False).squeeze(-1)
            if masked_positions.numel() == 0:
                break
            masked_probs = token_probs[0, masked_positions]          # (n_masked, vocab)
            confidences = masked_probs.max(dim=-1).values            # (n_masked,)
            best = confidences.argmax()
            seq_idx = masked_positions[best].item()
            chosen_token = masked_probs[best].argmax().item()

            # 3. Fill the chosen token into input_ids
            input_ids[0, seq_idx] = chosen_token
            decoded = tokenizer.decode([chosen_token])
            print(f"  step {step+1}: position {seq_idx} → {decoded!r}")

    # Final output
    generated_ids = input_ids[0, prompt_len:].tolist()
    generated_text = tokenizer.decode(generated_ids)
    print(f"\n{prompt}{generated_text}")









if __name__ == "__main__":
    main()
