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
from torch.distributions import Bernoulli, Normal

from torch_concepts.annotations import Annotations
from torch_concepts.nn import (
    AncestralSamplingInference,
    ConceptBottleneckGenerativeModel,
    MLP,
    ReconstructionLoss,
)

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


class TestBinaryStateEmbeddings:
    """A binary concept is scored by one Bernoulli probability but mixed from
    TWO independent context embeddings (w+, w-), matching the CEM/CBGM papers
    (Espinosa Zarlenga et al., NeurIPS 2022; Ismail et al., ICLR 2024, Sec 3.1)
    instead of fabricating the second state with a learned splitter.
    """

    def test_binary_embedding_variable_has_two_rows(self, binary_annotations):
        model = build_model(binary_annotations, plate=False)
        for name in ("a_embedding", "b_embedding"):
            assert tuple(model.pgm.variables[name].shape) == (2, EMBEDDING_SIZE)

    def test_binary_plate_gets_two_rows_per_member(self):
        annotations = Annotations(
            labels=["b0", "b1"], cardinalities=[1, 1], types=["binary", "binary"],
        )
        model = build_model(annotations, plate=True)
        # A single plate: both binary concepts homogeneous -> one embedding
        # variable holding both members' rows, member-major.
        assert tuple(model.pgm.variables["embeddings"].shape) == (4, EMBEDDING_SIZE)

    def test_concept_probability_shape_is_unaffected(self, binary_annotations):
        """The concept variable itself stays a single Bernoulli — only its
        embedding grew, not its output width."""
        model = build_model(binary_annotations, plate=False)
        out = model(query=list(model.pgm.variables), input=torch.rand(6, INPUT_SIZE))
        for name in ("a", "b"):
            assert out.probs[name].shape == (6, 1)

    def test_mixing_and_bottleneck_width_are_unaffected(self):
        """The bottleneck stays (k+1)*m regardless of how many embedding rows
        a binary concept contributes internally."""
        annotations = Annotations(
            labels=["a", "digit"], cardinalities=[1, 3], types=["binary", "categorical"],
        )
        model = build_model(annotations, plate=False)
        n_concepts = len(annotations.labels)
        assert tuple(model.pgm.variables["mixing"].shape) == (n_concepts, EMBEDDING_SIZE)
        # The decoder was sized for (k+1)*m at construction (see `build_model`);
        # a forward pass only succeeds if the bottleneck the model actually
        # assembles still matches that width.
        model(query=list(model.pgm.variables), input=torch.rand(3, INPUT_SIZE))

    def test_no_learned_splitter_anywhere_in_the_model(self, binary_annotations):
        model = build_model(binary_annotations, plate=False)
        names = [n for n, _ in model.named_parameters()]
        assert not any("splitter" in n for n in names)

    def test_backward_reaches_the_binary_embedding_encoder(self, binary_annotations):
        """Both w+ and w- must be trainable: the reconstruction gradient has to
        reach the encoder that produces them, not just the concept head."""
        model = build_model(binary_annotations, plate=False)
        out = model(query=list(model.pgm.variables), input=torch.rand(6, INPUT_SIZE))
        out.probs["input"].sum().backward()

        emb_cpd = model.pgm.factors["a_embedding"]
        grads = [p.grad for p in emb_cpd.parameters()]
        assert grads and all(g is not None for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)


class TestNormalObservation:
    """``observation=Normal`` with ``global_scale=True`` (the default): ``loc``
    from the decoder, ``scale`` a single learnable value shared by every pixel
    and every sample, instead of a whole second copy of the decoder.
    """

    def _model(self, binary_annotations, global_scale=True, input_size=INPUT_SIZE):
        n_contexts = len(binary_annotations.labels) + 1
        flat_size = input_size if isinstance(input_size, int) else 1
        if not isinstance(input_size, int):
            for d in input_size:
                flat_size *= d
        return ConceptBottleneckGenerativeModel(
            input_size=input_size,
            annotations=binary_annotations,
            encoder=MLP(flat_size, 16, LATENT_SIZE),
            decoder=MLP(n_contexts * EMBEDDING_SIZE, 16, flat_size),
            latent_size=LATENT_SIZE,
            embedding_size=EMBEDDING_SIZE,
            observation=Normal,
            global_scale=global_scale,
            plate=False,
        )

    def test_reports_loc_and_positive_scale(self, binary_annotations):
        model = self._model(binary_annotations)
        out = model(query=list(model.pgm.variables), input=torch.rand(6, INPUT_SIZE))
        assert out.loc["input"].shape == (6, INPUT_SIZE)
        assert bool((out.scale["input"] > 0).all())

    def test_global_scale_has_exactly_one_parameter(self, binary_annotations):
        model = self._model(binary_annotations, global_scale=True)
        scale_head = model.pgm.factors["input"].parametrization["scale"]
        assert sum(p.numel() for p in scale_head.parameters()) == 1

    def test_global_scale_false_reproduces_the_decoder_copy(self, binary_annotations):
        model = self._model(binary_annotations, global_scale=False)
        scale_head = model.pgm.factors["input"].parametrization["scale"]
        assert sum(p.numel() for p in scale_head.parameters()) > 1
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        scale_params = sum(p.numel() for p in scale_head.parameters())
        assert scale_params >= decoder_params  # a full independent copy

    def test_reconstruction_loss_is_finite(self, binary_annotations):
        model = self._model(binary_annotations)
        x = torch.rand(6, INPUT_SIZE)
        out = model(query=list(model.pgm.variables), input=x)
        out.extra = {"evidence": {"input": x}}
        loss = ReconstructionLoss(variable="input")(out)
        assert torch.isfinite(loss)

    def test_generation_through_ancestral_hard_sampling(self, binary_annotations):
        """The scale head's ``(B, size)`` output must survive an *unconditioned*
        decode of a multi-dimensional observation — the exact path
        ``run_generative_analysis.py`` uses to produce ``overview.png`` and the
        steering figures. A scale collapsed to ``(1,)``/``()`` would raise deep
        inside the relaxed-distribution builder here, not at training time."""
        image_shape = (1, 8, 8)
        model = self._model(binary_annotations, input_size=image_shape)
        model.eval()

        concepts = [v for v in model.pgm.variables.values() if v.variable_type == "concept"]
        query = ["input", *(v.name for v in concepts)]
        engine = AncestralSamplingInference(model.pgm, p_int=1.0, hard=True)

        out = engine.query(query=query, evidence={}, n_samples=3)
        flat_size = image_shape[0] * image_shape[1] * image_shape[2]
        assert out.loc["input"].shape == (3, flat_size)
        assert bool((out.scale["input"] > 0).all())


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


class TestSoftMixing:
    """``soft_mixing`` samples the concepts instead of teacher-forcing them.

    Under a hard 0/1 mixing score the unselected state embedding receives
    *exactly* zero reconstruction gradient — it is trained only by the concept
    head, so an intervention that selects it hands the decoder an input it never
    learned to decode. That is what makes a hard-mixed model unsteerable, and
    what these tests pin.
    """

    def _model(self, annotations, soft_mixing):
        n_contexts = len(annotations.labels) + 1
        return ConceptBottleneckGenerativeModel(
            input_size=INPUT_SIZE,
            annotations=annotations,
            encoder=MLP(INPUT_SIZE, 16, LATENT_SIZE),
            decoder=MLP(n_contexts * EMBEDDING_SIZE, 16, INPUT_SIZE),
            latent_size=LATENT_SIZE,
            embedding_size=EMBEDDING_SIZE,
            observation=Normal,
            scale_init=0.3,
            scale_learnable=False,
            soft_mixing=soft_mixing,
            plate=False,
        )

    @staticmethod
    def _state_embedding_weight(model, name="a_embedding"):
        parametrization = model.pgm.factors[name].parametrization["value"]
        return next(p for _, p in parametrization.named_parameters() if p.dim() == 2)

    def _reconstruction_grads(self, soft_mixing):
        """(‖grad w+‖, ‖grad w-‖) from reconstruction alone, all labels positive."""
        annotations = Annotations(labels=["a"], cardinalities=[1], types=["binary"])
        torch.manual_seed(0)
        model = self._model(annotations, soft_mixing=soft_mixing)
        weight = self._state_embedding_weight(model)

        x = torch.rand(8, INPUT_SIZE)
        query = model.default_query(torch.ones(8, 1))
        model.zero_grad()
        out = model(query=query, input=x)
        out.extra = {"evidence": {"input": x}}
        ReconstructionLoss(variable="input")(out).backward()

        rows = weight.shape[0] // 2
        return weight.grad[:rows].norm(), weight.grad[rows:].norm()

    def test_hard_mixing_starves_the_unselected_state(self):
        positive, negative = self._reconstruction_grads(soft_mixing=False)
        assert positive > 0
        assert negative == 0.0

    def test_soft_mixing_trains_both_states(self):
        positive, negative = self._reconstruction_grads(soft_mixing=True)
        assert positive > 0
        assert negative > 0

    def test_soft_mixing_leaves_the_concepts_unforced(self):
        annotations = Annotations(labels=["a"], cardinalities=[1], types=["binary"])
        ground_truth = torch.ones(4, 1)
        assert self._model(annotations, True).default_query(ground_truth)["a"] is None
        forced = self._model(annotations, False).default_query(ground_truth)["a"]
        assert torch.equal(forced, ground_truth)

    def test_soft_mixing_switches_the_engines_to_soft_draws(self):
        annotations = Annotations(labels=["a"], cardinalities=[1], types=["binary"])
        assert self._model(annotations, True).train_inference.hard is False
        assert self._model(annotations, False).train_inference.hard is True

    def test_fixed_scale_head_has_no_parameters(self):
        annotations = Annotations(labels=["a"], cardinalities=[1], types=["binary"])
        head = self._model(annotations, True).pgm.factors["input"].parametrization["scale"]
        assert sum(p.numel() for p in head.parameters()) == 0

    @pytest.mark.parametrize("soft_mixing", [False, True])
    def test_concepts_still_report_probs(self, soft_mixing):
        """Soft draws must not cost the concepts their reported ``probs``.

        A soft engine builds the plain relaxed families rather than their
        straight-through subclasses. If the parameter harvester only knows the
        latter, every concept site falls through and reports nothing — which
        surfaces only far downstream, as metrics raising ``KeyError`` and a
        concept loss silently reading zero.
        """
        annotations = Annotations(
            labels=["a", "d"], cardinalities=[1, 4], types=["binary", "categorical"]
        )
        model = self._model(annotations, soft_mixing=soft_mixing)
        ground_truth = torch.tensor([[1.0, 2.0]]).expand(4, -1)
        out = model(
            query=model.default_query(ground_truth), input=torch.rand(4, INPUT_SIZE)
        )

        assert "probs" in out.params
        for name in ("a", "d"):
            assert name in out.params["probs"], f"no probs reported for {name!r}"
        assert bool(((out.probs["a"] >= 0) & (out.probs["a"] <= 1)).all())
        assert torch.allclose(out.probs["d"].sum(-1), torch.ones(4), atol=1e-5)


class TestTemperatureAnnealing:
    """The relaxation temperature must anneal during training and hold at eval.

    A relaxed sample is only useful while its gradient is: soft early so every
    concept state is trained, then sharp so the bottleneck ends up committing to
    a state. Evaluation must then read the value training *reached* — decoding at
    the initial temperature would sample far softer codes than the trained
    decoder ever saw.
    """

    SCHEDULE = {
        "initial_temperature": 1.0,
        "annealing": "exponential",
        "annealing_rate": 0.5,
        "final_temperature": 0.1,
    }

    def _model(self, annotations, schedule=True):
        n_contexts = len(annotations.labels) + 1
        kwargs = (
            {"inference_kwargs": dict(self.SCHEDULE),
             "train_inference_kwargs": dict(self.SCHEDULE)}
            if schedule else {}
        )
        return ConceptBottleneckGenerativeModel(
            input_size=INPUT_SIZE,
            annotations=annotations,
            encoder=MLP(INPUT_SIZE, 16, LATENT_SIZE),
            decoder=MLP(n_contexts * EMBEDDING_SIZE, 16, INPUT_SIZE),
            latent_size=LATENT_SIZE,
            embedding_size=EMBEDDING_SIZE,
            observation=Normal,
            scale_init=0.3,
            scale_learnable=False,
            soft_mixing=True,
            plate=False,
            lightning=True,  # the temperature hook is a LightningModule hook
            **kwargs,
        )

    @staticmethod
    def _train_batches(model, n):
        """Lightning calls ``on_train_batch_end`` once per optimiser step."""
        model.train()
        for i in range(n):
            model.on_train_batch_end(None, None, i)

    def test_temperature_decreases_then_settles(self, binary_annotations):
        model = self._model(binary_annotations)
        assert float(model.train_inference.temperature) == pytest.approx(1.0)
        self._train_batches(model, 1)
        after_one = float(model.train_inference.temperature)
        assert 0.1 < after_one < 1.0
        self._train_batches(model, 200)
        assert float(model.train_inference.temperature) == pytest.approx(0.1)

    def test_eval_engine_reads_the_training_temperature(self, binary_annotations):
        model = self._model(binary_annotations)
        self._train_batches(model, 5)
        assert float(model.eval_inference.temperature) == pytest.approx(
            float(model.train_inference.temperature)
        )

    def test_eval_mode_does_not_anneal_further(self, binary_annotations):
        model = self._model(binary_annotations)
        self._train_batches(model, 5)
        settled = float(model.train_inference.temperature)
        model.eval()
        model(query=list(model.pgm.variables), input=torch.rand(4, INPUT_SIZE))
        assert float(model.train_inference.temperature) == pytest.approx(settled)

    def test_temperature_survives_a_checkpoint(self, binary_annotations):
        """The analysis script rebuilds from a checkpoint, so the annealed value
        has to travel in ``state_dict`` — otherwise it decodes at the initial
        temperature."""
        trained = self._model(binary_annotations)
        self._train_batches(trained, 200)

        fresh = self._model(binary_annotations)
        assert float(fresh.train_inference.temperature) == pytest.approx(1.0)
        fresh.load_state_dict(trained.state_dict())
        assert float(fresh.train_inference.temperature) == pytest.approx(0.1)
        assert float(fresh.eval_inference.temperature) == pytest.approx(0.1)

    def test_the_default_schedule_is_still_constant(self, binary_annotations):
        """Library default unchanged; only conf/model/cbgm.yaml opts in."""
        model = self._model(binary_annotations, schedule=False)
        self._train_batches(model, 10)
        assert float(model.train_inference.temperature) == pytest.approx(1.0)
