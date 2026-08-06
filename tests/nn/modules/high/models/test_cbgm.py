"""Smoke tests for the Concept Bottleneck Generative Model.

CBGM sets ``param_for_discrete_var = "probs"``, so every head it builds must end
in the activation that maps a raw output into ``[0, 1]`` (Bernoulli) or onto the
simplex (categorical). ``_flexible_parametrization`` composes that activation
onto every head from the family's ``DistributionSpec``, so the heads passed in —
including the user's ``decoder`` — are raw. These tests pin the resulting
parameters to their domains.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from torch_concepts.annotations import Annotations
from torch_concepts.nn import ConceptBottleneckGenerativeModel, MLP

pytest.importorskip("pyro", reason="CBGM's default inference engine needs pyro-ppl")

INPUT_SIZE, LATENT_SIZE, EMBEDDING_SIZE = 24, 8, 4


def build_model(annotations, plate=None):
    n_contexts = len(annotations.labels) + 1
    return ConceptBottleneckGenerativeModel(
        input_size=INPUT_SIZE,
        annotations=annotations,
        encoder=MLP(INPUT_SIZE, 16, LATENT_SIZE),
        # Raw: the model composes the observation's `probs` sigmoid on top.
        decoder=MLP(n_contexts * EMBEDDING_SIZE, 16, INPUT_SIZE),
        latent_size=LATENT_SIZE,
        embedding_size=EMBEDDING_SIZE,
        observation=Bernoulli,
        plate=plate,
    )


@pytest.fixture
def binary_annotations():
    return Annotations(labels=["a", "b"], cardinalities=[1, 1], types=["binary", "binary"])


@pytest.fixture
def categorical_annotations():
    return Annotations(
        labels=["digit", "color"], cardinalities=[4, 3],
        types=["categorical", "categorical"],
    )


class TestConceptBottleneckGenerativeModel:
    def test_binary_concepts_and_observation_are_probabilities(self, binary_annotations):
        model = build_model(binary_annotations)
        out = model(query=list(model.pgm.variables), input=torch.rand(6, INPUT_SIZE))
        # `probs` is one annotated tensor holding every queried variable that has
        # them — both concepts and the reconstructed observation.
        assert bool(((out.probs >= 0) & (out.probs <= 1)).all())
        assert out.probs["input"].shape == (6, INPUT_SIZE)
        for name in ("a", "b"):
            assert out.probs[name].shape == (6, 1)

    def test_categorical_concepts_normalise_per_concept(self, categorical_annotations):
        model = build_model(categorical_annotations, plate=False)
        out = model(query=list(model.pgm.variables), input=torch.rand(6, INPUT_SIZE))
        for name, cardinality in zip(["digit", "color"], [4, 3]):
            probs = out.probs[name]
            assert probs.shape[-1] == cardinality
            assert torch.allclose(probs.sum(-1), torch.ones(probs.shape[:-1]), atol=1e-5)

    def test_a_categorical_plates_cpd_normalises_each_member(self):
        """The plate's concept CPD emits one simplex per member, not one per row.

        Asserted on the CPD's own output rather than on ``out.probs``: the
        queried tensor for a plate is renormalised downstream over the flattened
        width, which is a property of the plate query path, not of the head this
        test covers.
        """
        # Same cardinality on both concepts, so they can share one plate.
        annotations = Annotations(
            labels=["d1", "d2"], cardinalities=[3, 3],
            types=["categorical", "categorical"],
        )
        model = build_model(annotations, plate=True)
        plate = model.pgm.variables["concepts"]
        assert plate.members == ["d1", "d2"] and plate.member_size == 3

        emitted = {}
        model.pgm.factors["concepts"].register_forward_hook(
            lambda mod, inp, out: emitted.update(probs=out["probs"].detach())
        )
        model(query=list(model.pgm.variables), input=torch.rand(6, INPUT_SIZE))

        probs = emitted["probs"]
        assert probs.shape == (6, 6)
        assert torch.allclose(probs.reshape(6, 2, 3).sum(-1), torch.ones(6, 2), atol=1e-5)

    def test_the_latent_prior_and_guide_produce_a_positive_scale(self, binary_annotations):
        model = build_model(binary_annotations)
        out = model(query=list(model.pgm.variables), input=torch.rand(6, INPUT_SIZE))
        assert bool((out.scale["z"] > 0).all())

    def test_gradients_reach_the_decoder(self, binary_annotations):
        model = build_model(binary_annotations)
        out = model(query=list(model.pgm.variables), input=torch.rand(6, INPUT_SIZE))
        out.probs["input"].sum().backward()
        assert any(p.grad is not None for p in model.decoder.parameters())


class CountingBackbone(nn.Module):
    """A backbone that records how many times it ran."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.out_features = out_features
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.linear(x)


class TestGuideSharesOneBackbonePass:
    """The guide's ``loc`` and ``scale`` are heads over one shared trunk.

    Without it, ``second='auto'`` deep-copies ``first`` into the scale head — a
    second copy of the encoder's weights, and a second forward pass through them
    on every step. With a pretrained backbone in there, that is the difference
    between one ResNet pass per batch and two.
    """

    def _model(self, annotations, backbone):
        n_contexts = len(annotations.labels) + 1
        return ConceptBottleneckGenerativeModel(
            input_size=INPUT_SIZE,
            annotations=annotations,
            backbone=backbone,
            decoder=MLP(n_contexts * EMBEDDING_SIZE, 16, INPUT_SIZE),
            latent_size=LATENT_SIZE,
            embedding_size=EMBEDDING_SIZE,
            observation=Bernoulli,
            plate=False,
        )

    def test_the_backbone_runs_once_per_forward(self, binary_annotations):
        backbone = CountingBackbone(INPUT_SIZE, 32)
        model = self._model(binary_annotations, backbone)
        model(query=list(model.pgm.variables), input=torch.rand(6, INPUT_SIZE))
        assert backbone.calls == 1

    def test_the_backbone_is_not_duplicated_into_the_scale_head(
        self, binary_annotations
    ):
        backbone = CountingBackbone(INPUT_SIZE, 32)
        model = self._model(binary_annotations, backbone)
        guide = model.pgm.guides["z"]
        # One instance, reachable through the trunk and nowhere else.
        assert sum(m is backbone for m in guide.modules()) == 1
        assert sum(m is backbone for m in guide.trunk.modules()) == 1

    def test_both_parameters_still_depend_on_the_input(self, binary_annotations):
        model = self._model(binary_annotations, CountingBackbone(INPUT_SIZE, 32))
        a = model(query=list(model.pgm.variables), input=torch.rand(6, INPUT_SIZE))
        b = model(query=list(model.pgm.variables), input=torch.rand(6, INPUT_SIZE))
        # An amortised posterior: a different image gives a different q(z | x).
        assert not torch.allclose(
            a.guide_params["loc"]["z"], b.guide_params["loc"]["z"]
        )
        assert not torch.allclose(
            a.guide_params["scale"]["z"], b.guide_params["scale"]["z"]
        )


class TestContinuousConcepts:
    def test_a_continuous_concept_builds_and_reports_loc_and_scale(self):
        """`param_for_discrete_var` is 'probs', which a Normal does not have, so
        the discrete activation must only be applied to discrete variables."""
        annotations = Annotations(
            labels=["a", "h"], cardinalities=[1, 1], types=["binary", "continuous"]
        )
        model = build_model(annotations)
        out = model(query=list(model.pgm.variables), input=torch.rand(4, INPUT_SIZE))
        assert sorted(out.params["h"]) == ["loc", "scale"]
        assert sorted(out.params["a"]) == ["probs"]
        assert bool((out.scale["h"] > 0).all())

    def test_the_scale_head_is_independent_of_the_location(self):
        """`loc` and `scale` need their own scoring layers.

        Putting the shared scoring layer in a `trunk` and leaving the heads as
        bare activations makes `scale == softplus(loc)`: a parameter-free head
        pinning the spread to the mean.
        """
        annotations = Annotations(
            labels=["a", "h"], cardinalities=[1, 1], types=["binary", "continuous"]
        )
        cpd = build_model(annotations, plate=False).pgm.factors["h"]
        embedding = torch.randn(5, 1, EMBEDDING_SIZE)
        params = cpd(parent_values={"h_embedding": embedding})

        assert not torch.allclose(params["scale"], F.softplus(params["loc"]))
        for head in ("loc", "scale"):
            assert sum(p.numel() for p in cpd.parametrization[head].parameters()) > 0
