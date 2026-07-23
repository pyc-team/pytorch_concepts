"""Tests for fixed, learnable, and refined concept-graph construction."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from torch_concepts import ConceptGraph
from torch_concepts.construct_graph import (
    GraphGenerator,
    GraphGeneratorFixed,
    GraphGeneratorFixedSpec,
    GraphGeneratorLearnable,
)
from torch_concepts.data import ToyDataset
from torch_concepts.data.base.datamodule import ConceptDataModule
import torch_concepts.construct_graph as graph_module


@pytest.fixture
def dataset():
    """Small duck-typed dataset sufficient for every generator callback."""
    names = ["rain", "wet_grass", "traffic"]
    return SimpleNamespace(
        concept_names=names,
        concepts=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]
        ),
        graph_native=ConceptGraph(
            torch.tensor(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
            node_names=names,
        ),
        label_descriptions={
            "rain": "whether it rains",
            "wet_grass": "whether the grass is wet",
        },
    )


class TestGeneratorContract:
    """Verify the API contract shared by fixed and learnable generators."""

    def test_base_generator_is_abstract(self):
        """The common base cannot be instantiated without `generate`."""
        with pytest.raises(TypeError):
            GraphGenerator(name="anything", source="anything")

    def test_fixed_and_learnable_registries_are_separate(self):
        """A source must belong to exactly one generator family."""
        assert "GroundTruth" in GraphGeneratorFixed._sources
        assert "WANDA" not in GraphGeneratorFixed._sources
        assert "WANDA" in GraphGeneratorLearnable._sources
        assert "GroundTruth" not in GraphGeneratorLearnable._sources

    @pytest.mark.parametrize("generator_cls", [GraphGeneratorFixed, GraphGeneratorLearnable])
    def test_unknown_source_reports_available_sources(self, generator_cls):
        """Unknown sources produce a useful error instead of failing later."""
        with pytest.raises(ValueError, match="Unknown source.*registered sources"):
            generator_cls(name="missing", source="NotRegistered")

    def test_repr_exposes_configuration_and_family(self):
        generator = GraphGeneratorFixed(name="ground_truth", source="GroundTruth")
        assert repr(generator) == (
            "GraphGeneratorFixed(name='ground_truth', "
            "source='GroundTruth', trainable=False)"
        )


class TestGroundTruthConstruction:
    """Exercise graph reuse from a dataset native graph."""

    def test_returns_native_graph_and_updates_common_state(self, dataset):
        """Return the native object and mark the generator as fitted."""
        generator = GraphGeneratorFixed(name="ground_truth", source="GroundTruth")

        assert generator.graph is None
        assert generator.fitted is False
        graph = generator.generate(dataset)

        assert graph is dataset.graph_native
        assert generator.graph is graph
        assert generator.fitted is True

    def test_requires_native_graph(self, dataset):
        """Fail cleanly when the dataset has no native graph."""
        dataset.graph_native = None
        generator = GraphGeneratorFixed(name="ground_truth", source="GroundTruth")

        with pytest.raises(ValueError, match="dataset.graph_native"):
            generator.generate(dataset)
        assert generator.graph is None
        assert generator.fitted is False

    def test_rejects_wrong_name(self):
        with pytest.raises(ValueError, match="only name='ground_truth'"):
            GraphGeneratorFixed(name="native", source="GroundTruth")


class _CausalLearnGraph:
    def __init__(self, adjacency):
        self.graph = np.asarray(adjacency)


class TestCausalLearnConstruction:
    """Test PC and GES adapters without requiring the optional package."""

    def test_pc_passes_constraint_parameters_and_converts_endpoints(
        self, dataset, monkeypatch
    ):
        calls = []

        def pc(data, alpha, indep_test):
            calls.append((data, alpha, indep_test))
            # CausalLearn's [-1, 1] pair represents rain -> wet_grass.
            return SimpleNamespace(G=_CausalLearnGraph([[0, -1, 0], [1, 0, -1], [0, -1, 0]]))

        monkeypatch.setattr(graph_module, "_import_causallearn", lambda name: pc)
        generator = GraphGeneratorFixed(
            name="pc", source="Causallearn", alpha=0.2, indep_test="fisherz"
        )

        graph = generator.generate(dataset)

        np.testing.assert_array_equal(calls[0][0], dataset.concepts.numpy())
        assert calls[0][1:] == (0.2, "fisherz")
        torch.testing.assert_close(
            graph.data,
            torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]]),
        )
        assert graph.node_names == dataset.concept_names

    def test_ges_passes_score_and_accepts_mapping_result(self, dataset, monkeypatch):
        """Pass the score to GES and convert its mapping result."""
        calls = []

        def ges(data, score_func):
            calls.append((data, score_func))
            return {"G": _CausalLearnGraph([[0, -1, 0], [1, 0, 0], [0, 0, 0]])}

        monkeypatch.setattr(graph_module, "_import_causallearn", lambda name: ges)
        generator = GraphGeneratorFixed(
            name="ges", source="Causallearn", score_func="custom_score"
        )

        graph = generator.generate(dataset)

        assert calls[0][1] == "custom_score"
        torch.testing.assert_close(
            graph.data,
            torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )

    @pytest.mark.parametrize("name", ["not-a-method", "PC", ""])
    def test_rejects_unknown_algorithm(self, name):
        with pytest.raises(ValueError, match="Unknown CausalLearn name"):
            GraphGeneratorFixed(name=name, source="Causallearn")

    @pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1])
    def test_pc_rejects_alpha_outside_open_unit_interval(self, alpha):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            GraphGeneratorFixed(name="pc", source="Causallearn", alpha=alpha)

    def test_ges_does_not_apply_irrelevant_pc_alpha_validation(self):
        generator = GraphGeneratorFixed(name="ges", source="Causallearn", alpha=2)
        assert generator.alpha == 2


class TestLLMConstruction:
    """Exercise deterministic LLM construction without network requests."""

    def test_llm_queries_each_concept_pair_and_builds_graph_from_majority_votes(
        self, dataset
    ):
        """Query each concept pair and resolve repeated answers by majority vote.

        Also verify that the generated prompt includes the configured domain and
        the concepts names and descriptions.
        """
        prompts = []
        responses = iter(["A->B\nA->B\nnone", "B->A", "invalid"])

        def backend(prompt, repeats):
            prompts.append((prompt, repeats))
            return next(responses)

        generator = GraphGeneratorFixed(
            name="fake-model",
            source="LLM",
            llm_backend=backend,
            repeats=3,
            domain="weather",
            use_rag=False,
        )
        graph = generator.generate(dataset)

        torch.testing.assert_close(
            graph.data,
            torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        assert len(prompts) == 3  # n * (n - 1) / 2
        assert all(repeats == 3 for _, repeats in prompts)
        assert "in the domain of weather" in prompts[0][0]
        assert "rain — whether it rains" in prompts[0][0]

    @pytest.mark.parametrize("repeats", [0, -1, 1.5, True])
    def test_repeats_must_be_a_positive_non_boolean_integer(self, repeats):
        with pytest.raises(ValueError, match="positive integer"):
            GraphGeneratorFixed(
                name="fake", source="LLM", llm_backend=lambda *_args, **_kwargs: "none",
                repeats=repeats,
            )

    def test_validates_llm_and_rag_configuration(self):
        """Reject invalid backends and incomplete retrieval settings."""
        with pytest.raises(TypeError, match="llm_backend.*callable"):
            GraphGeneratorFixed(name="fake", source="LLM", llm_backend=object())
        with pytest.raises(ValueError, match="neither `rag` nor `documents`"):
            GraphGeneratorFixed(
                name="fake", source="LLM", llm_backend=lambda *_a, **_k: "none",
                use_rag=True,
            )
        with pytest.raises(ValueError, match="at least 1"):
            GraphGeneratorFixed(
                name="fake", source="LLM", llm_backend=lambda *_a, **_k: "none",
                n_retrieved=0,
            )

    @pytest.mark.parametrize(
        ("response", "expected"),
        [("A->B\nB->A", "none"), ("garbage", "none"), ("", "none")],
    )
    def test_invalid_or_tied_llm_votes_produce_no_edge(self, response, expected):
        assert graph_module._most_frequent_token(response) == expected


class TestRefinement:
    """Verify refinement capability checks and edge orientation."""

    @pytest.mark.parametrize(
        "refinement",
        [
            {"name": "ground_truth", "source": "GroundTruth"},
            {"name": "ges", "source": "Causallearn"},
        ],
    )
    def test_rejects_builtin_refiners_without_orient_edges(self, refinement):
        """Require a dedicated orientation callback from every refiner."""
        with pytest.raises(TypeError, match="does not support edge orientation"):
            GraphGeneratorFixed(
                name="ground_truth", source="GroundTruth", refinement=refinement
            )

    def test_llm_refinement_orients_only_ambiguous_pairs(self, dataset):
        """Query reciprocal edges while leaving the native graph untouched."""
        dataset.graph_native = ConceptGraph(
            torch.tensor([[0.0, -1.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            node_names=dataset.concept_names,
        )
        calls = []

        def backend(prompt, repeats):
            calls.append(prompt)
            return "B->A"

        generator = GraphGeneratorFixed(
            name="ground_truth",
            source="GroundTruth",
            refinement={
                "name": "fake-model", "source": "LLM", "llm_backend": backend,
                "use_rag": False,
            },
        )
        graph = generator.generate(dataset)

        torch.testing.assert_close(
            graph.data,
            torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )
        assert len(calls) == 1
        # Refinement returns a new snapshot and leaves native data untouched.
        assert dataset.graph_native.data[0, 1] == -1

    def test_fully_directed_graph_skips_refinement_backend(self, dataset):
        """Avoid calls when every edge is already directed."""
        def backend(*_args, **_kwargs):
            pytest.fail("The refiner must not be called for a fully directed graph")

        generator = GraphGeneratorFixed(
            name="ground_truth",
            source="GroundTruth",
            refinement={
                "name": "fake-model", "source": "LLM", "llm_backend": backend,
                "use_rag": False,
            },
        )
        assert generator.generate(dataset) is dataset.graph_native


class TestWANDAConstruction:
    """Exercise differentiable construction and snapshot materialization."""

    def test_forward_is_differentiable_and_generate_materializes_detached_snapshot(
        self, dataset
    ):
        generator = GraphGeneratorLearnable(
            name="wanda",
            source="WANDA",
            concept_names=dataset.concept_names,
            hard_threshold=False,
        )
        with torch.no_grad():
            generator.np_params.copy_(torch.tensor([[0.0], [2.0], [1.0]]))

        adjacency = generator()
        assert adjacency.requires_grad
        assert adjacency.grad_fn is not None

        graph = generator.generate(dataset)
        torch.testing.assert_close(
            graph.data,
            torch.tensor([[0.0, 2.0, 1.0], [-2.0, 0.0, -1.0], [-1.0, 1.0, 0.0]]),
        )
        assert graph.data.requires_grad is False
        assert generator.graph is graph
        assert generator.fitted is True

    def test_hard_threshold_has_zero_diagonal_and_binary_forward(self, dataset):
        """Hard WANDA emits a binary adjacency without self-loops."""
        generator = GraphGeneratorLearnable(
            name="wanda", source="WANDA", concept_names=dataset.concept_names,
            hard_threshold=True, threshold_init=0.5,
        )
        with torch.no_grad():
            generator.np_params.copy_(torch.tensor([[0.0], [2.0], [1.0]]))

        adjacency = generator()
        assert set(adjacency.detach().unique().tolist()) <= {0.0, 1.0}
        torch.testing.assert_close(torch.diag(adjacency), torch.zeros(3))

    def test_generate_rejects_different_concept_names_without_changing_state(self, dataset):
        generator = GraphGeneratorLearnable(
            name="wanda", source="WANDA", concept_names=["other"]
        )
        with pytest.raises(ValueError, match="concept names must match"):
            generator.generate(dataset)
        assert generator.graph is None
        assert generator.fitted is False

    def test_wanda_validates_name_and_threshold(self, dataset):
        with pytest.raises(ValueError, match="only name='wanda'"):
            GraphGeneratorLearnable(
                name="other", source="WANDA", concept_names=dataset.concept_names
            )
        with pytest.raises(ValueError, match="non-negative"):
            GraphGeneratorLearnable(
                name="wanda", source="WANDA", concept_names=dataset.concept_names,
                threshold_init=-0.1,
            )


class TestDataModuleIntegration:
    """Check generator selection through `precompute_graph`."""

    def test_fixed_generator_is_materialized_and_can_replace_dataset_graph(self):
        """Fixed generation immediately populates the DataModule graph."""
        dataset = ToyDataset(dataset="xor", n_gen=12, seed=42)
        datamodule = ConceptDataModule(dataset)
        native = dataset.graph_native

        datamodule.precompute_graph(name="ground_truth", source="GroundTruth")

        assert isinstance(datamodule.graph_generator, GraphGeneratorFixed)
        assert datamodule.graph_generator.fitted is True
        assert datamodule.graph is native

    def test_use_as_gt_false_retains_unmaterialized_learnable_generator(self):
        """Retain WANDA without exporting it before end-to-end training."""
        dataset = ToyDataset(dataset="xor", n_gen=12, seed=42)
        datamodule = ConceptDataModule(dataset)
        graph_before = datamodule.graph

        datamodule.precompute_graph(
            name="wanda", source="WANDA", use_as_gt=False,
            concept_names=list(dataset.concept_names),
        )

        assert isinstance(datamodule.graph_generator, GraphGeneratorLearnable)
        assert datamodule.graph_generator.fitted is False
        assert datamodule.graph_generator.graph is None
        assert datamodule.graph is graph_before

    def test_dispatch_rejects_unknown_and_ambiguous_sources(self, monkeypatch):
        """Require a source registered in exactly one generator family."""
        datamodule = ConceptDataModule(ToyDataset(dataset="xor", n_gen=12, seed=42))
        with pytest.raises(ValueError, match="must identify exactly one"):
            datamodule.precompute_graph(name="x", source="missing")

        fixed_init = GraphGeneratorFixed._sources["GroundTruth"]
        monkeypatch.setitem(GraphGeneratorLearnable._sources, "GroundTruth", fixed_init)
        with pytest.raises(ValueError, match="must identify exactly one"):
            datamodule.precompute_graph(name="ground_truth", source="GroundTruth")
