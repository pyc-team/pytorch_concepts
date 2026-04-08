import torch
from torch.distributions import Bernoulli, Categorical
from torch_concepts.distributions import Delta

from torch_concepts import ConceptVariable, LatentVariable
from torch_concepts.nn import ParametricCPD, ProbabilisticModel, DeterministicInference

from torch_concepts.steerling import (
    SteerlingBackbone, SteerlingLatentToConcept, SteerlingLMHead,
    SteerlingConceptExogenousToLatent, load_steerling_concept_names,
    prepare_steerling_evidence,
)


def main():

    n_sup = 33732
    n_unsup = 101196
    emb_dim = 4096

    prompt = "To improve research in interpretable machine learning, we need"
    n_new_tokens = 20

    # ── 1. Load backbone and prepare evidence ──────────────────────────
    backbone = SteerlingBackbone(pretrained=True, freeze=True, device="cpu")
    data = prepare_steerling_evidence(backbone, prompt, n_new_tokens)
    input_ids = data["input_ids"]       # (1, T_prompt + n_new_tokens)
    hidden = data["hidden"]             # (1, T_prompt + n_new_tokens, 4096)
    print(f"Prompt: {prompt!r}")
    print(f"Hidden states: {hidden.shape}")

    # ── 2. Concept encoder heads ───────────────────────────────────────
    lm_head = SteerlingLMHead(pretrained=True, freeze=True)


    # ── 3. Variables ───────────────────────────────────────────────────
    latent = LatentVariable("input", size=emb_dim, distribution=Delta)

    # concepts
    sup_concepts_names = load_steerling_concept_names()
    sup_concepts = ConceptVariable(sup_concepts_names, distribution=Bernoulli)
    unsup_concepts_names = [f"unsup_{i}" for i in range(n_unsup)]
    unsup_concepts = ConceptVariable(unsup_concepts_names, distribution=Bernoulli)

    # concept-based hidden state
    concept_latent = LatentVariable("concept_latent", size=emb_dim, distribution=Delta)

    new_token = LatentVariable("new_token", size=lm_head.vocab_size, distribution=Categorical)


    # ── 4. CPDs ────────────────────────────────────────────────────────
    latent_cpd = ParametricCPD("input", parents=[], parametrization=torch.nn.Identity())
    
    # parametrize concepts with the pretrained concept encoder
    sup_concepts_head = SteerlingLatentToConcept(pretrained=True, freeze=True)
    sup_concepts_cpd = ParametricCPD(sup_concepts_names, parents=["input"], parametrization=sup_concepts_head,
                                     shared=True, shared_name="sup_concepts")
    unsup_concepts_head = SteerlingLatentToConcept(pretrained=True, freeze=True, is_unknown=True)
    unsup_concepts_cpd = ParametricCPD(unsup_concepts_names, parents=["input"], parametrization=unsup_concepts_head,
                                       shared=True, shared_name="unsup_concepts")
    
    # parametrize concept_latent as concepts @ embeddings
    # embeddings are stored inside the layer (as buffers), so the CPD
    # only needs concept parents — the inference engine concatenates them
    # via the {'concepts'} path: cat(parent_concepts, dim=-1) → (B, T, N)
    concept_latent_layer = SteerlingConceptExogenousToLatent(
        known_embeddings=sup_concepts_head.get_embeddings(),      # (n_sup, D)
        unknown_embeddings=unsup_concepts_head.get_embeddings(),  # (n_unsup, D)
    )
    concept_latent_cpd = ParametricCPD("concept_latent", 
                                       parents=sup_concepts_names + unsup_concepts_names, 
                                       parametrization=concept_latent_layer)
    
    new_token_cpd = ParametricCPD("new_token", parents=["concept_latent"], parametrization=lm_head)


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
        variables=[latent] + sup_concepts + unsup_concepts,
        factors=[latent_cpd, sup_concepts_cpd, unsup_concepts_cpd],
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
        variables=[latent] + sup_concepts + unsup_concepts + [concept_latent],
        factors=[latent_cpd, sup_concepts_cpd, unsup_concepts_cpd, concept_latent_cpd],
    )
    inference = DeterministicInference(concept_model)
    print(f"\nQuerying concept_latent:")
    result = inference.query(["concept_latent"], evidence={'input':hidden})
    print(f"  shape: {result.shape}")  # should be (1, T, 4096)
    print(f"  values: {result}")



    # ── Query new tokens one at a time ─────────────────────
    # Build full PGM with token prediction
    concept_model = ProbabilisticModel(
        variables=[latent] + sup_concepts + unsup_concepts + [concept_latent, new_token],
        factors=[latent_cpd, sup_concepts_cpd, unsup_concepts_cpd, concept_latent_cpd, new_token_cpd],
    )
    inference = DeterministicInference(concept_model)

    tokenizer = backbone.tokenizer
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
