"""
Causal Effects: Seeing versus Doing
===================================

How much does smoking raise the risk of cancer? In the model below, a genotype
makes people both more likely to smoke and more likely to get cancer::

    genotype ---------------------+
       |                          v
       +--> smoking --> tar --> cancer

so smokers get cancer more often than non-smokers partly because of their
genes. The two questions differ:

- *seeing*: how much more often do smokers get cancer?
  ``P(cancer | smoking)`` conditions on the evidence; ``RejectionSampling``
  answers it by keeping only the samples consistent with the evidence.
- *doing*: how much more often would people get cancer if they were made to
  smoke? ``P(cancer | do(smoking))`` overrides smoking and changes only its
  descendants, as ``intervention`` does on any engine and as evidence does on
  a forward engine such as ``AncestralSamplingInference``.

The mechanisms are fixed probability tables, so the answers can be checked by
hand: seeing suggests an effect of +0.51, while the causal effect is +0.24.
"""

# %%
import torch
from torch.distributions import Bernoulli

from torch_concepts import ConceptVariable, seed_everything
from torch_concepts.nn import (
    AncestralSamplingInference,
    BayesianNetwork,
    CallableConceptToConcept,
    DoIntervention,
    FixedPrior,
    ParametricCPD,
    RejectionSampling,
    UniformPolicy,
    intervention,
)

seed_everything(42)

# %%
# Model
# -----
# Each CPD gives the probability of its variable from its parents' values.
genotype = ConceptVariable("genotype", distribution=Bernoulli)
smoking = ConceptVariable("smoking", distribution=Bernoulli)
tar = ConceptVariable("tar", distribution=Bernoulli)
cancer = ConceptVariable("cancer", distribution=Bernoulli)


def probability(fn):
    return {"probs": CallableConceptToConcept(fn, use_bias=False)}


scm = BayesianNetwork(
    variables=[genotype, smoking, tar, cancer],
    factors=[
        ParametricCPD(
            genotype,
            parents=[],
            parametrization={"probs": FixedPrior(torch.tensor([0.3]))},
        ),
        ParametricCPD(
            smoking,
            parents=[genotype],
            parametrization=probability(lambda g: 0.2 + 0.6 * g),
        ),
        ParametricCPD(
            tar,
            parents=[smoking],
            parametrization=probability(lambda s: 0.1 + 0.8 * s),
        ),
        # parents are concatenated on the last axis: [genotype, tar]
        ParametricCPD(
            cancer,
            parents=[genotype, tar],
            parametrization=probability(
                lambda gt: 0.1 + 0.5 * gt[..., :1] + 0.3 * gt[..., 1:]
            ),
        ),
    ],
)


# %%
# Seeing
# ------
def report(name, p):
    print(f"{name:<34} {p[0]:.2f} vs {p[1]:.2f} -> {p[0] - p[1]:+.2f}")


one, zero = torch.ones(1, 1), torch.zeros(1, 1)
rejection = RejectionSampling(scm, n_samples=100_000)
seen = []
for s in (one, zero):
    out = rejection.query({"cancer": one}, evidence={"smoking": s})
    seen.append(out.probabilities.item())
report("P(cancer | smoking = 1 vs 0)", seen)

# %%
# Doing
# -----
done = []
for value in (1.0, 0.0):
    with intervention(scm, DoIntervention(value), UniformPolicy(), ["smoking"]):
        out = rejection.query({"cancer": one}, evidence={})
    done.append(out.probabilities.item())
report("P(cancer | do(smoking = 1 vs 0))", done)

# A forward engine reads evidence as an intervention: the clamped value only
# reaches the descendants of smoking, never the genotype. ``exact=True`` draws
# hard samples (the default relaxed ones are meant for training).
ancestral = AncestralSamplingInference(scm, exact=True)
forward = []
for s in (one, zero):
    out = ancestral.query(["cancer"], evidence={"smoking": s.expand(100_000, 1)})
    forward.append(out.samples["cancer"].mean().item())
report("ancestral sampling, smoking = 1 vs 0", forward)
