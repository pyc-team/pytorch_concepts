"""Smoke tests for the Conditional Variational Autoencoder.

The CVAE is the generative baseline for
:class:`~torch_concepts.nn.ConceptBottleneckGenerativeModel`: same variational
machinery, but the concepts are *given* to the decoder rather than predicted from
``z``. Three things follow, and they are what these tests pin:

* the concepts are graph roots with a learnable marginal ``p(c)``, so their CPDs
  are parent-less priors that still have to land in their own domain (a
  per-member simplex for a categorical plate, ``[0, 1]`` for a binary concept);
* the prior on ``z`` is the paper's **conditional** one, ``p(z | c)``, so the
  graph runs ``concepts -> z -> input`` and the KL's target moves with ``c``;
* the guide and the decoder read the condition through **one**
  :class:`~torch_concepts.nn.modules.high.models.cvae.ConceptEmbedding`, so ``c``
  has a single learned representation;
* ``param_for_discrete_var = "probs"``, so every discrete head ends in the
  activation ``_flexible_parametrization`` composes on top of the raw layer;
* the observation is a ``Delta`` — one ``value`` head, no ``scale``, and a
  generated sample equal to the decoder's output rather than to it plus noise.
"""
import pytest
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from torch_concepts.annotations import Annotations
from torch_concepts.nn import (
    AncestralSamplingInference,
    CompositeLoss,
    ConceptLoss,
    ConditionalVariationalAutoencoder,
    KLDivergenceLoss,
    MLP,
    ReconstructionLoss,
)
from torch_concepts.distributions import Delta

pytest.importorskip("pyro", reason="CVAE's default inference engine needs pyro-ppl")

INPUT_SIZE, LATENT_SIZE, EMBEDDING_SIZE = 24, 8, 4


def build_model(annotations, plate=None, input_size=INPUT_SIZE, **kwargs):
    """A CVAE whose decoder is sized for ``[z | embedded c]``.

    The condition is ``n_concepts * EMBEDDING_SIZE`` wide however the concepts are
    distributed, which is the point of the per-concept embedding — so this width
    is a function of the annotations alone, exactly as `condition_size` documents.
    """
    condition_size = len(annotations.labels) * EMBEDDING_SIZE
    flat_input = input_size if isinstance(input_size, int) else int(torch.tensor(input_size).prod())
    return ConditionalVariationalAutoencoder(
        input_size=input_size,
        annotations=annotations,
        encoder=MLP(flat_input, 16, LATENT_SIZE),
        # The decoder's output is the reconstruction, unactivated.
        decoder=MLP(LATENT_SIZE + condition_size, 16, flat_input),
        latent_size=LATENT_SIZE,
        embedding_size=EMBEDDING_SIZE,
        plate=plate,
        **kwargs,
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


def binary_query(model, batch=5):
    """A `default_query` over random binary concept values."""
    return model.default_query(torch.randint(0, 2, (batch, 2)).float())


class TestConditionalVariationalAutoencoder:
    def test_binary_concepts_are_probabilities(self, binary_annotations):
        model = build_model(binary_annotations, plate=False)
        out = model(query=binary_query(model), input=torch.rand(5, INPUT_SIZE))
        for name in ("a", "b"):
            probs = out.probs[name]
            assert bool(((probs >= 0) & (probs <= 1)).all())
        # The observation is a Delta, so it reports `value` rather than `probs`.
        assert "input" not in out.probs.annotation.labels

    def test_categorical_marginals_normalise_per_concept(self, categorical_annotations):
        model = build_model(categorical_annotations, plate=False)
        for name, cardinality in (("digit", 4), ("color", 3)):
            probs = model.pgm.factors[name](parent_values={})["probs"]
            assert probs.shape == (cardinality,)
            assert torch.allclose(probs.sum(), torch.ones(()))

    def test_a_categorical_plates_marginal_normalises_each_member(self):
        """The plate's prior is one flat `LearnablePrior(size)`, so nothing but a
        per-member softmax keeps each member on its own simplex — a single softmax
        over the whole parameter would leave every member summing to less than 1."""
        annotations = Annotations(
            labels=["d1", "d2"], cardinalities=[4, 4],
            types=["categorical", "categorical"],
        )
        model = build_model(annotations, plate=True)
        assert "concepts" in model.pgm.variables  # one plate, both members

        probs = model.pgm.factors["concepts"](parent_values={})["probs"]
        assert probs.shape == (8,)
        assert torch.allclose(probs[:4].sum(), torch.ones(()))
        assert torch.allclose(probs[4:].sum(), torch.ones(()))

    def test_the_guide_reports_a_positive_scale(self, binary_annotations):
        model = build_model(binary_annotations, plate=False)
        out = model(query=binary_query(model), input=torch.rand(5, INPUT_SIZE))
        assert bool((out.guide_params["scale"]["z"] > 0).all())

    def test_gradients_reach_the_decoder(self, binary_annotations):
        model = build_model(binary_annotations, plate=False)
        out = model(query=binary_query(model), input=torch.rand(5, INPUT_SIZE))
        out.value["input"].sum().backward()
        assert any(
            p.grad is not None and bool((p.grad != 0).any())
            for p in model.decoder.parameters()
        )


class TestSharedConditionEmbedding:
    """``c`` has ONE learned representation: `k` layers total, not `2k`.

    The guide, the decoder and the conditional prior all read the condition
    through the same `ConceptEmbedding`. Separate copies would train independently
    and let `q(z | x, c)`, `p(z | c)` and the decoder disagree about what a
    concept means.
    """

    def test_the_guide_and_the_decoder_share_one_instance(self, binary_annotations):
        model = build_model(binary_annotations, plate=False)
        embedder = model.condition_embedding
        assert sum(m is embedder for m in model.pgm.guides["z"].modules()) == 1
        assert sum(m is embedder for m in model.pgm.factors["input"].modules()) == 1

    def test_the_conditional_prior_shares_it_too(self, binary_annotations):
        model = build_model(binary_annotations, plate=False)
        embedder = model.condition_embedding
        assert sum(m is embedder for m in model.pgm.factors["z"].modules()) == 1

    def test_condition_size_is_known_from_the_annotations(self, categorical_annotations):
        """The caller has to size the decoder before constructing the model, so
        `condition_size` must be a function of the annotations, not of the built
        layers — and the two must agree."""
        model = build_model(categorical_annotations, plate=False)
        expected = len(categorical_annotations.labels) * EMBEDDING_SIZE
        assert model.condition_size == expected
        assert model.condition_embedding.out_features == expected

    def test_one_layer_per_concept_sized_to_that_concepts_own_width(
        self, categorical_annotations
    ):
        """A binary concept is 1 input column and a 4-way categorical is 4; each
        gets its own `Linear(size_i, m)` so the decoder weighs them alike."""
        annotations = Annotations(
            labels=["a", "digit"], cardinalities=[1, 4], types=["binary", "categorical"]
        )
        model = build_model(annotations, plate=False)
        layers = model.condition_embedding.embeddings
        assert len(layers) == 2
        assert [layer.in_features for layer in layers] == [1, 4]
        assert {layer.out_features for layer in layers} == {EMBEDDING_SIZE}

    def test_backward_reaches_the_embedding_layers(self, binary_annotations):
        model = build_model(binary_annotations, plate=False)
        out = model(query=binary_query(model), input=torch.rand(5, INPUT_SIZE))
        out.value["input"].sum().backward()
        for layer in model.condition_embedding.embeddings:
            assert layer.weight.grad is not None
            assert bool((layer.weight.grad != 0).any())


class TestConditionalPrior:
    """``p(z | c)`` — the paper's prior, replacing a fixed ``N(0, I)``.

    The latent is drawn per condition, so ``z`` only has to carry what ``c`` does
    not. The consequence worth knowing: the KL's target is now learnable, which is
    what these tests pin — the reported ``params['z']`` must be the conditional
    prior, and it must actually move with ``c``.
    """

    def test_z_is_no_longer_a_root(self, binary_annotations):
        model = build_model(binary_annotations, plate=False)
        cpd = model.pgm.factors["z"]
        assert not cpd.is_root
        assert [p.name for p in cpd.parents] == ["a", "b"]

    def test_the_graph_runs_concepts_then_z_then_input(self, binary_annotations):
        model = build_model(binary_annotations, plate=False)
        levels = [[v.name for v in level] for level in model.pgm.levels]
        assert levels == [["a", "b"], ["z"], ["input"]]

    def test_the_prior_moves_with_the_condition(self, binary_annotations):
        """Same image, different concepts: `p(z | c)` must differ. A prior that
        ignored `c` would be the old fixed one wearing a network."""
        model = build_model(binary_annotations, plate=False)
        x = torch.rand(5, INPUT_SIZE)
        a = model(query=model.default_query(torch.zeros(5, 2)), input=x)
        b = model(query=model.default_query(torch.ones(5, 2)), input=x)
        assert not torch.allclose(a.params["z"]["loc"], b.params["z"]["loc"])
        assert not torch.allclose(a.params["z"]["scale"], b.params["z"]["scale"])

    def test_it_is_not_a_standard_normal(self, binary_annotations):
        model = build_model(binary_annotations, plate=False)
        out = model(query=binary_query(model), input=torch.rand(5, INPUT_SIZE))
        loc, scale = out.params["z"]["loc"], out.params["z"]["scale"]
        assert not torch.allclose(loc, torch.zeros_like(loc), atol=1e-3)
        assert bool((scale > 0).all())

    def test_the_kl_targets_the_conditional_prior(self, binary_annotations):
        """`KLDivergenceLoss` needs no change: it reads `p` from `params['z']`,
        which is now the conditional prior rather than a fixed N(0, I)."""
        model = build_model(binary_annotations, plate=False)
        out = model(query=binary_query(model), input=torch.rand(5, INPUT_SIZE))
        expected = torch.distributions.kl_divergence(
            torch.distributions.Normal(**out.guide_params["z"]),
            torch.distributions.Normal(**out.params["z"]),
        ).sum(-1).mean()
        assert torch.allclose(KLDivergenceLoss(latents=["z"])(out), expected)

    def test_the_kl_trains_the_prior_network(self, binary_annotations):
        model = build_model(binary_annotations, plate=False)
        out = model(query=binary_query(model), input=torch.rand(5, INPUT_SIZE))
        KLDivergenceLoss(latents=["z"])(out).backward()
        grads = [p.grad for p in model.prior_encoder.parameters() if p.grad is not None]
        assert grads and any(bool((g != 0).any()) for g in grads)

    def test_the_default_prior_encoder_is_an_mlp_sized_from_the_model(
        self, binary_annotations
    ):
        """Derived from `condition_size`/`latent_size`, not a magic width. The
        hidden layer is what lets the prior model concept *interactions*."""
        model = build_model(binary_annotations, plate=False)
        assert isinstance(model.prior_encoder, MLP)
        assert model.prior_encoder.out_features == LATENT_SIZE

    def test_a_supplied_prior_encoder_sizes_the_heads(self, binary_annotations):
        condition_size = len(binary_annotations.labels) * EMBEDDING_SIZE
        model = build_model(
            binary_annotations, plate=False,
            prior_encoder=MLP(condition_size, 32, 12),
        )
        assert model.prior_encoder.out_features == 12
        for head in ("loc", "scale"):
            layer = model.pgm.factors["z"].parametrization[head]
            # `scale` is Sequential(Linear, softplus); `loc` is the bare Linear.
            layer = layer[0] if isinstance(layer, nn.Sequential) else layer
            assert layer.in_features == 12

    def test_a_prior_encoder_without_out_features_is_rejected(self, binary_annotations):
        """Fail naming the module to fix, not later on a shape mismatch."""
        with pytest.raises(ValueError, match="conditional prior"):
            build_model(binary_annotations, plate=False, prior_encoder=nn.ReLU())

    @pytest.mark.parametrize(
        "kwargs", [{"plate": False}, {"plate": True}, {"conditional_prior": False}]
    )
    def test_generation_runs_the_full_chain(self, kwargs):
        """p(c) -> p(z | c) -> p(input | z, c), with no evidence at all."""
        annotations = Annotations(
            labels=["a", "b"], cardinalities=[1, 1], types=["binary", "binary"]
        )
        model = build_model(annotations, **kwargs)
        model.eval()
        engine = AncestralSamplingInference(model.pgm, p_int=1.0)
        names = [
            v.name for v in model.pgm.variables.values() if v.variable_type == "concept"
        ]
        out = engine.query(query=["input", *names], evidence={}, n_samples=4)
        assert out.value["input"].shape == (4, INPUT_SIZE)


class TestPriorSwitch:
    """``conditional_prior`` picks `p(z | c)` or a fixed `N(0, I)`; the guide never moves."""

    @pytest.mark.parametrize("conditional_prior", [True, False])
    def test_the_guide_always_reads_every_concept(
        self, binary_annotations, conditional_prior
    ):
        """`q(z | x, c)` is hard-coded — the paper's recognition network, either prior."""
        model = build_model(
            binary_annotations, plate=False, conditional_prior=conditional_prior
        )
        assert [p.name for p in model.pgm.guides["z"].parents] == ["input", "a", "b"]

    @pytest.mark.parametrize("conditional_prior", [True, False])
    def test_the_posterior_moves_with_the_condition(
        self, binary_annotations, conditional_prior
    ):
        model = build_model(
            binary_annotations, plate=False, conditional_prior=conditional_prior
        )
        x = torch.rand(5, INPUT_SIZE)
        a = model(query=model.default_query(torch.zeros(5, 2)), input=x)
        b = model(query=model.default_query(torch.ones(5, 2)), input=x)
        # Same image, different condition: q(z | x, c) must move regardless of prior.
        assert not torch.allclose(
            a.guide_params["loc"]["z"], b.guide_params["loc"]["z"]
        )

    def test_the_conditional_prior_reads_the_concepts(self, binary_annotations):
        model = build_model(binary_annotations, plate=False, conditional_prior=True)
        assert [p.name for p in model.pgm.factors["z"].parents] == ["a", "b"]
        assert model.prior_encoder is not None

    def test_the_fixed_prior_is_a_parentless_standard_normal(self, binary_annotations):
        """`N(0, I)` exactly, and carrying no gradient — the KL target must not drift."""
        model = build_model(binary_annotations, plate=False, conditional_prior=False)
        factor = model.pgm.factors["z"]
        assert list(factor.parents) == []
        assert model.prior_encoder is None
        assert torch.equal(factor.parametrization["loc"](), torch.zeros(LATENT_SIZE))
        assert torch.equal(factor.parametrization["scale"](), torch.ones(LATENT_SIZE))
        assert not any(p.requires_grad for p in factor.parameters())


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
    """The guide's ``loc`` and ``scale`` are two readouts over one shared trunk.

    Without it the scale head is a second copy of the encoder's weights and a
    second forward pass through them on every step — with a pretrained backbone in
    there, the difference between one ResNet pass per batch and two.
    """

    def _model(self, annotations, backbone):
        condition_size = len(annotations.labels) * EMBEDDING_SIZE
        return ConditionalVariationalAutoencoder(
            input_size=INPUT_SIZE,
            annotations=annotations,
            backbone=backbone,
            decoder=MLP(LATENT_SIZE + condition_size, 16, INPUT_SIZE),
            latent_size=LATENT_SIZE,
            embedding_size=EMBEDDING_SIZE,
            plate=False,
        )

    def test_the_backbone_runs_once_per_forward(self, binary_annotations):
        backbone = CountingBackbone(INPUT_SIZE, 32)
        model = self._model(binary_annotations, backbone)
        model(query=binary_query(model, 6), input=torch.rand(6, INPUT_SIZE))
        assert backbone.calls == 1

    def test_the_backbone_is_not_duplicated_into_the_scale_head(self, binary_annotations):
        backbone = CountingBackbone(INPUT_SIZE, 32)
        model = self._model(binary_annotations, backbone)
        guide = model.pgm.guides["z"]
        # One instance, reachable through the trunk and nowhere else.
        assert sum(m is backbone for m in guide.modules()) == 1
        assert sum(m is backbone for m in guide.trunk.modules()) == 1

    def test_both_parameters_still_depend_on_the_input(self, binary_annotations):
        model = self._model(binary_annotations, CountingBackbone(INPUT_SIZE, 32))
        c = torch.randint(0, 2, (6, 2)).float()
        a = model(query=model.default_query(c), input=torch.rand(6, INPUT_SIZE))
        b = model(query=model.default_query(c), input=torch.rand(6, INPUT_SIZE))
        # An amortised posterior: a different image gives a different q(z | x, c).
        assert not torch.allclose(a.guide_params["loc"]["z"], b.guide_params["loc"]["z"])
        assert not torch.allclose(
            a.guide_params["scale"]["z"], b.guide_params["scale"]["z"]
        )


class TestDeltaObservation:
    """The observation is a point mass: one ``value`` head and no ``scale``.

    The decoder's output is the reconstruction verbatim, so there is no sigma
    left to act as a hidden reconstruction weight and no noise on a generated
    sample.
    """

    def test_reports_value_and_nothing_else(self, binary_annotations):
        model = build_model(binary_annotations, plate=False)
        out = model(query=binary_query(model), input=torch.rand(5, INPUT_SIZE))
        assert sorted(out.params["input"]) == ["value"]
        assert out.value["input"].shape == (5, INPUT_SIZE)

    def test_the_cpd_allocates_a_single_head(self, binary_annotations):
        """No `scale` entry: not a GlobalScale, not a second copy of the decoder."""
        cpd = build_model(binary_annotations, plate=False).pgm.factors["input"]
        assert set(cpd.parametrization) == {"value"}

    def test_the_decoder_head_still_shares_the_condition_embedding(
        self, binary_annotations
    ):
        """One head now, but it must still read `c` through the guide's embedding."""
        model = build_model(binary_annotations, plate=False)
        head = model.pgm.factors["input"].parametrization["value"]
        assert sum(m is model.condition_embedding for m in head.modules()) == 1

    def test_reconstruction_loss_is_the_squared_error(self, binary_annotations):
        """0.5 * ||x - v||^2 is the sigma=1 Gaussian NLL minus its constant, so
        the switch off `Normal` leaves the tuned loss weights meaning the same."""
        model = build_model(binary_annotations, plate=False)
        x = torch.rand(5, INPUT_SIZE)
        out = model(query=binary_query(model), input=x)
        loss = ReconstructionLoss(variable="input")(out)
        expected = (0.5 * (out.value["input"] - x).pow(2).sum(-1)).mean()
        assert torch.isfinite(loss)
        assert torch.allclose(loss, expected)

    def test_the_sum_reduction_still_aggregates_the_batch(self, binary_annotations):
        model = build_model(binary_annotations, plate=False)
        x = torch.rand(5, INPUT_SIZE)
        out = model(query=binary_query(model), input=x)
        mean = ReconstructionLoss(variable="input")(out)
        total = ReconstructionLoss(variable="input", reduction="sum")(out)
        assert torch.allclose(total, mean * 5)


class TestTrainingAndGeneration:
    def test_default_extra_publishes_the_evidence(self, binary_annotations):
        """`BaseModel.forward` fills `out.extra` from this hook, and it is the only
        way `ReconstructionLoss` ever sees the observed image."""
        model = build_model(binary_annotations, plate=False)
        x = torch.rand(5, INPUT_SIZE)
        out = model(query=binary_query(model), input=x)
        assert out.extra is not None
        assert torch.equal(out.extra["evidence"]["input"], x)

    def test_a_full_elbo_step_reaches_every_learnable_part(self, binary_annotations):
        """recon + kl + concept: the reconstruction trains the decoder and the shared
        embedding, the concept term trains the marginal p(c). A missing `default_extra`
        fails this at the reconstruction term."""
        model = build_model(binary_annotations, plate=False)
        loss_fn = CompositeLoss(
            terms=[
                ReconstructionLoss(variable="input"),
                KLDivergenceLoss(latents=["z"]),
                # The model reports `probs`, so the binary term scores
                # probabilities rather than logits.
                ConceptLoss(binary=nn.BCELoss()),
            ],
            weights=[1.0, 1.0, 1.0],
        )
        c = torch.randint(0, 2, (5, 2)).float()
        out = model(query=model.default_query(c), input=torch.rand(5, INPUT_SIZE))
        terms = loss_fn.breakdown(out, model.prepare_target(c))
        assert all(torch.isfinite(t) for t in terms.values())

        sum(terms.values()).backward()
        prior = model.pgm.factors["a"].parametrization["probs"]
        for module in (model.decoder, model.condition_embedding, prior):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            assert grads and any(bool((g != 0).any()) for g in grads)

    def test_unconditional_generation_through_ancestral_sampling(self, binary_annotations):
        """The learnable marginal p(c) exists so the model can be drawn from with no
        evidence at all — FID needs draws, not reconstructions. This is also the path
        that exercises a multi-dimensional observation's decode."""
        image_shape = (1, 8, 8)
        model = build_model(binary_annotations, plate=False, input_size=image_shape)
        model.eval()

        engine = AncestralSamplingInference(model.pgm, p_int=1.0)
        out = engine.query(query=["input", "a", "b"], evidence={}, n_samples=3)
        assert out.value["input"].shape == (3, 64)
        assert out.probs["a"].shape == (3, 1)
        # The draw IS the decoder output: no `loc + noise` to discard.
        assert torch.allclose(out.samples["input"], out.value["input"])


class TestContinuousConcepts:
    def test_a_continuous_concept_reports_loc_and_a_positive_scale(self):
        """`param_for_discrete_var` is 'probs', which a Normal does not have, so the
        discrete activation must only be applied to discrete variables."""
        annotations = Annotations(
            labels=["a", "h"], cardinalities=[1, 1], types=["binary", "continuous"]
        )
        model = build_model(annotations, plate=False)
        c = torch.rand(5, 2)
        out = model(query=model.default_query(c), input=torch.rand(5, INPUT_SIZE))
        assert sorted(out.params["h"]) == ["loc", "scale"]
        assert sorted(out.params["a"]) == ["probs"]
        assert bool((out.scale["h"] > 0).all())

    def test_a_multivariate_normal_concept_sizes_its_cholesky_factor(self):
        """`_prior_heads` sizes the marginal from `param_sizes`, not from `size`: a
        `scale_tril` needs the `size * (size + 1) // 2` free Cholesky entries."""
        annotations = Annotations(labels=["v"], cardinalities=[3], types=["continuous"])
        model = build_model(
            annotations,
            plate=False,
            variable_distributions={"continuous": MultivariateNormal},
        )
        params = model.pgm.factors["v"](parent_values={})
        assert params["loc"].shape == (3,)
        assert params["scale_tril"].shape == (3, 3)
        assert bool((params["scale_tril"].diagonal() > 0).all())
