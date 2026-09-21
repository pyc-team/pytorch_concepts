import time

import pytest
import torch
from torch import nn

from torch_concepts import Annotations, ConceptGraph
from torch_concepts.nn import CausalCGM, CGMTrainingLoss, ConceptLoss
from torch_concepts.nn.modules.low.graph_aggregator import GraphAggregator
from torch_concepts.nn.modules.low.predictors.mix import (
    MixConceptEmbeddingToConceptEmbedding,
)
from torch_concepts.nn.modules.low.predictors.neural_structural_equations import (
    NeuralStructuralEquations,
)
from torch_concepts.nn.modules.outputs import ModelOutput
from torch_concepts.tensor import AnnotatedTensor


def _binary_annotations(labels):
    return Annotations(
        labels=list(labels),
        cardinalities=[1] * len(labels),
        types=["binary"] * len(labels),
    )


def _mixed_annotations():
    return Annotations(
        labels=["b", "cat", "z"],
        cardinalities=[1, 3, 1],
        types=["binary", "categorical", "continuous"],
    )


class _CountingGenerator(nn.Module):
    def __init__(self, adjacency):
        super().__init__()
        self.adjacency = nn.Parameter(adjacency.clone())
        self.calls = 0

    def forward(self):
        self.calls += 1
        return self.adjacency


class TestGraphAggregator:
    def test_fixed_adjacency_matches_manual_source_to_target_sum(self):
        adjacency = torch.tensor(
            [
                [0.0, 2.0, 0.5],
                [1.0, 0.0, 3.0],
                [4.0, 0.0, 0.0],
            ]
        )
        embeddings = torch.arange(24.0).reshape(2, 3, 4)

        layer = GraphAggregator(adjacency=adjacency)
        actual = layer(embeddings)
        expected = torch.matmul(embeddings.transpose(-2, -1), adjacency).transpose(-2, -1)

        assert torch.allclose(actual, expected)
        assert torch.allclose(layer.adjacency, adjacency)

    def test_source_and_target_selection_use_original_adjacency_indices(self):
        adjacency = torch.tensor(
            [
                [0.0, 10.0, 20.0],
                [30.0, 0.0, 40.0],
                [50.0, 60.0, 0.0],
            ]
        )
        source_embeddings = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
            ]
        )

        layer = GraphAggregator(adjacency=adjacency)
        actual = layer(
            source_embeddings,
            source_concepts=[2, 0],
            target_concept=1,
        )
        expected = 60.0 * source_embeddings[:, 0] + 10.0 * source_embeddings[:, 1]

        assert torch.allclose(actual, expected)

    def test_generator_result_is_cached_until_clear_and_keeps_gradients(self):
        generator = _CountingGenerator(torch.tensor([[0.0, 1.0], [2.0, 0.0]]))
        layer = GraphAggregator(generator=generator)
        embeddings = torch.randn(3, 2, 5, requires_grad=True)

        first = layer(embeddings)
        second = layer(embeddings)

        assert generator.calls == 1
        assert torch.allclose(first, second)
        first.sum().backward()
        assert generator.adjacency.grad is not None

        layer.clear()
        layer(embeddings)
        assert generator.calls == 2

    def test_adjacency_property_requires_a_generated_or_used_graph(self):
        layer = GraphAggregator(adjacency=torch.eye(2))

        with pytest.raises(RuntimeError, match="has not generated"):
            _ = layer.adjacency

    def test_training_generator_rejects_explicit_adjacency_bypass(self):
        layer = GraphAggregator(
            generator=_CountingGenerator(torch.tensor([[0.0, 1.0], [0.0, 0.0]]))
        )
        embeddings = torch.randn(1, 2, 3)

        with pytest.raises(RuntimeError, match="explicit adjacency"):
            layer(embeddings, adjacency=torch.eye(2))

        layer.eval()
        assert layer(embeddings, adjacency=torch.eye(2)).shape == (1, 2, 3)

    @pytest.mark.parametrize(
        "adjacency, embeddings, error",
        [
            (torch.ones(2, 3), torch.randn(1, 2, 4), "must be square"),
            (torch.eye(3), torch.randn(1, 2, 4), "rows must match"),
        ],
    )
    def test_shape_errors_are_explicit(self, adjacency, embeddings, error):
        with pytest.raises(ValueError, match=error):
            GraphAggregator(adjacency=adjacency)(embeddings)

    def test_target_concept_bounds_are_checked(self):
        with pytest.raises(IndexError, match="target_concept"):
            GraphAggregator(adjacency=torch.eye(2))(
                torch.randn(1, 2, 3),
                target_concept=2,
            )

    def test_constructor_requires_exactly_one_graph_source(self):
        with pytest.raises(ValueError, match="exactly one"):
            GraphAggregator()
        with pytest.raises(ValueError, match="exactly one"):
            GraphAggregator(generator=nn.Identity(), adjacency=torch.eye(2))


class TestMixConceptEmbeddingToConceptEmbedding:
    def test_mixes_binary_categorical_and_continuous_sources(self):
        layer = MixConceptEmbeddingToConceptEmbedding(
            in_embeddings=2,
            concept_types=["binary", "categorical", "continuous"],
            cardinalities=[1, 3, 1],
        )
        binary_embeddings = torch.tensor([[[10.0, 0.0], [0.0, 20.0]]])
        categorical_embeddings = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]]])
        continuous_embeddings = torch.tensor([[[2.0, 5.0]]])

        mixed = layer(
            concept_embeddings=[
                binary_embeddings,
                categorical_embeddings,
                continuous_embeddings,
            ],
            concept_values=[
                torch.tensor([[0.25]]),
                torch.tensor([[1]]),
                torch.tensor([[3.0]]),
            ],
        )

        expected = torch.tensor([[[2.5, 15.0], [0.0, 2.0], [6.0, 15.0]]])
        assert torch.allclose(mixed, expected)

    def test_complete_output_scatters_subset_sources_and_zeros_missing_concepts(self):
        layer = MixConceptEmbeddingToConceptEmbedding(
            in_embeddings=2,
            concept_types=["binary", "categorical", "continuous"],
            cardinalities=[1, 3, 1],
            complete_output=True,
        )

        mixed = layer(
            concept_embeddings=[
                torch.tensor([[[2.0, 4.0]]]),
                torch.tensor([[[1.0, 10.0], [20.0, 2.0]]]),
            ],
            concept_values=[
                torch.tensor([[5.0]]),
                torch.tensor([[0.75]]),
            ],
            source_concepts=[2, 0],
        )

        assert mixed.shape == (1, 3, 2)
        assert torch.allclose(mixed[:, 2], torch.tensor([[10.0, 20.0]]))
        assert torch.allclose(mixed[:, 1], torch.zeros(1, 2))
        assert torch.allclose(mixed[:, 0], torch.tensor([[5.75, 8.0]]))

    def test_binary_embedding_expansion_uses_one_state_embedding(self):
        layer = MixConceptEmbeddingToConceptEmbedding(
            in_embeddings=3,
            concept_types=["binary"],
            cardinalities=[1],
            expand_binary_embeddings=True,
        )
        output = layer(
            concept_embeddings=[torch.randn(2, 1, 3)],
            concept_values=[torch.rand(2, 1)],
        )
        assert output.shape == (2, 1, 3)

    @pytest.mark.parametrize(
        "kwargs, error",
        [
            (
                dict(concept_embeddings=[], concept_values=[]),
                "At least one",
            ),
            (
                dict(
                    concept_embeddings=[torch.randn(1, 2, 3)],
                    concept_values=[],
                ),
                "one concept value",
            ),
            (
                dict(
                    concept_embeddings=[torch.randn(1, 2, 3)],
                    concept_values=[torch.rand(1, 1)],
                    source_concepts=[0, 1],
                ),
                "one index",
            ),
            (
                dict(
                    concept_embeddings=[torch.randn(1, 2, 3), torch.randn(1, 2, 3)],
                    concept_values=[torch.rand(1, 1), torch.rand(1, 1)],
                    source_concepts=[0, 0],
                ),
                "duplicates",
            ),
        ],
    )
    def test_input_validation(self, kwargs, error):
        layer = MixConceptEmbeddingToConceptEmbedding(
            in_embeddings=3,
            concept_types=["binary", "binary"],
            cardinalities=[1, 1],
        )
        with pytest.raises(ValueError, match=error):
            layer(**kwargs)

    def test_source_concepts_bounds_and_state_shapes_are_checked(self):
        layer = MixConceptEmbeddingToConceptEmbedding(
            in_embeddings=3,
            concept_types=["binary"],
            cardinalities=[1],
        )
        with pytest.raises(IndexError, match="out-of-range"):
            layer([torch.randn(1, 2, 3)], [torch.rand(1, 1)], source_concepts=[3])
        with pytest.raises(ValueError, match="two state embeddings"):
            layer([torch.randn(1, 1, 3)], [torch.rand(1, 1)])
        with pytest.raises(ValueError, match="Embedding width"):
            layer([torch.randn(1, 2, 4)], [torch.rand(1, 1)])


class TestNeuralStructuralEquations:
    def test_forward_shapes_for_all_selected_and_single_targets(self):
        equations = NeuralStructuralEquations(
            in_embeddings=4,
            out_concept_types=["binary", "categorical", "continuous"],
            cardinalities_out_concepts=[1, 3, 1],
            shared_n_layers=1,
            concept_n_layers=2,
        )
        contexts = torch.randn(5, 3, 4)

        all_outputs = equations(contexts)
        selected = equations(contexts, target_concepts=[2, 0])
        single_context = equations(contexts[:, 1], target_concept=1)

        assert all_outputs.shape == (5, 5)
        assert selected.shape == (5, 2)
        assert single_context.shape == (5, 3)
        assert torch.allclose(selected, torch.cat([all_outputs[:, 4:5], all_outputs[:, 0:1]], dim=-1))

    def test_for_targets_is_a_view_over_the_same_equations(self):
        equations = NeuralStructuralEquations(
            in_embeddings=3,
            out_concept_types=["binary", "binary", "binary"],
            cardinalities_out_concepts=[1, 1, 1],
        )
        contexts = torch.randn(2, 3, 3)

        assert torch.allclose(
            equations.for_targets([1, 2])(contexts),
            equations(contexts, target_concepts=[1, 2]),
        )

    @pytest.mark.parametrize(
        "kwargs, error",
        [
            (dict(shared_n_layers=-1), "shared_n_layers"),
            (dict(concept_n_layers=0), "concept_n_layers"),
            (
                dict(
                    out_concept_types=["binary", "binary"],
                    cardinalities_out_concepts=[1],
                ),
                "same length",
            ),
        ],
    )
    def test_constructor_validation(self, kwargs, error):
        base = dict(
            in_embeddings=2,
            out_concept_types=["binary"],
            cardinalities_out_concepts=[1],
        )
        base.update(kwargs)
        with pytest.raises(ValueError, match=error):
            NeuralStructuralEquations(**base)

    def test_forward_validation(self):
        equations = NeuralStructuralEquations(
            in_embeddings=2,
            out_concept_types=["binary", "binary"],
            cardinalities_out_concepts=[1, 1],
        )
        with pytest.raises(ValueError, match="either target_concept"):
            equations(torch.randn(1, 2, 2), target_concept=0, target_concepts=[0])
        with pytest.raises(IndexError, match="target_concept"):
            equations(torch.randn(1, 2, 2), target_concept=2)
        with pytest.raises(ValueError, match="trailing shape"):
            equations(torch.randn(1, 3, 2))


class TestCGMTrainingLoss:
    def _output(self, *, include_prior=True):
        ann = _binary_annotations(["c1", "c2", "y"])
        target = AnnotatedTensor(
            torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]),
            ann.to_concept_space(),
            axis=-1,
        )
        output = ModelOutput(
            logits=AnnotatedTensor(
                torch.tensor(
                    [[-1.0, 0.5, 2.0], [1.5, -0.5, -2.0]],
                    requires_grad=True,
                ),
                ann,
                axis=-1,
            ),
            target=target,
        )
        if include_prior:
            prior_ann = _binary_annotations(["c1__copy", "c2__copy", "y__copy"])
            output.params["prior_logits"] = AnnotatedTensor(
                torch.tensor(
                    [[-0.5, 0.25, 1.0], [0.75, -0.25, -1.0]],
                    requires_grad=True,
                ),
                prior_ann,
                axis=-1,
            )
        return output, target

    def test_breakdown_scores_posterior_and_prior_by_concept_and_task_groups(self):
        output, target = self._output()
        prediction = ConceptLoss(binary=nn.BCEWithLogitsLoss())
        loss = CGMTrainingLoss(prediction_loss=prediction, lambda_dag=0.0)
        loss.task_names = ["y"]

        terms = loss.breakdown(output, target)

        expected_posterior = (
            nn.functional.binary_cross_entropy_with_logits(
                output.logits.tensor[:, :2],
                target.tensor[:, :2],
            )
            + nn.functional.binary_cross_entropy_with_logits(
                output.logits.tensor[:, 2:],
                target.tensor[:, 2:],
            )
        )
        expected_prior = (
            nn.functional.binary_cross_entropy_with_logits(
                output.params["prior_logits"].tensor[:, :2],
                target.tensor[:, :2],
            )
            + nn.functional.binary_cross_entropy_with_logits(
                output.params["prior_logits"].tensor[:, 2:],
                target.tensor[:, 2:],
            )
        )

        assert set(terms) == {"posterior", "prior"}
        assert torch.allclose(terms["posterior"], expected_posterior)
        assert torch.allclose(terms["prior"], expected_prior)
        assert torch.allclose(loss(output, target), expected_posterior + expected_prior)

    def test_evaluation_loss_scores_outputs_without_prior_quantities(self):
        output, target = self._output(include_prior=False)
        loss = CGMTrainingLoss(
            prediction_loss=ConceptLoss(binary=nn.BCEWithLogitsLoss()),
            evaluation_loss=ConceptLoss(binary=nn.BCEWithLogitsLoss()),
            lambda_dag=0.0,
        )
        loss.task_names = ["y"]

        terms = loss.breakdown(output, target)

        assert terms["posterior"].ndim == 0
        assert terms["prior"].item() == 0.0

    def test_configure_terms_adds_graph_regularizers_for_trainable_graphs(self):
        class _Generator:
            trainable = True
            task_names = ["y"]

        output, target = self._output()
        output.params["adjacency"] = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0]])
        output.params["low_logits"] = output.logits - 0.5
        output.params["high_logits"] = output.logits + 0.5

        loss = CGMTrainingLoss(lambda_dag=2.0, lambda_cace=0.3)
        loss.configure_terms(_Generator())
        terms = loss.breakdown(output, target)

        assert loss.term_names == ["posterior", "prior", "dagma", "cace"]
        assert loss.weights == [1.0, 1.0, 2.0, 0.3]
        assert terms["dagma"].ndim == 0
        assert terms["cace"].ndim == 0

    @pytest.mark.parametrize(
        "prior_labels, error",
        [
            (["c1", "c2", "y"], "must end"),
            (["c1__copy", "c1__copy", "y__copy"], "duplicate"),
            (["c1__copy", "missing__copy", "y__copy"], "no matching"),
        ],
    )
    def test_prior_label_validation(self, prior_labels, error):
        output, target = self._output(include_prior=False)
        output.params["prior_logits"] = AnnotatedTensor(
            torch.zeros(2, 3),
            _binary_annotations(prior_labels),
            axis=-1,
        )
        loss = CGMTrainingLoss(lambda_dag=0.0)

        with pytest.raises(ValueError, match=error):
            loss.breakdown(output, target)

    def test_empty_prediction_target_is_an_error(self):
        output, _ = self._output()
        target = AnnotatedTensor(
            torch.empty(2, 0),
            Annotations(labels=[], cardinalities=[], types=[]),
            axis=-1,
        )
        loss = CGMTrainingLoss(lambda_dag=0.0)
        loss.task_names = ["c1", "c2", "y"]

        with pytest.raises(ValueError, match="empty target"):
            loss._split_prediction_loss(ConceptLoss(binary=nn.BCEWithLogitsLoss()), output, target)


class TestCausalCGM:
    def _model(self, **kwargs):
        ann = _binary_annotations(["c", "y"])
        graph = ConceptGraph(
            torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
            node_names=["c", "y"],
        )
        return CausalCGM(
            input_size=3,
            annotations=ann,
            task_names="y",
            embedding_size=2,
            graph=graph,
            run_interventions=False,
            **kwargs,
        )

    def test_training_forward_returns_posterior_prior_and_adjacency_for_loss(self):
        torch.manual_seed(0)
        model = self._model()
        model.train()
        target = model.prepare_target(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))

        output = model(input=torch.randn(2, 3), target=target)
        loss = CGMTrainingLoss(lambda_dag=0.0)(output, target)
        loss.backward()

        assert set(output.params) == {"logits", "prior_logits", "adjacency"}
        assert output.logits.shape == (2, 2)
        assert output.params["prior_logits"].shape == (2, 2)
        assert torch.allclose(output.params["adjacency"], model.graph.data)
        assert any(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    def test_eval_forward_materializes_eval_network_once_and_reports_concepts(self):
        torch.manual_seed(0)
        model = self._model()
        model.eval()

        with torch.no_grad():
            first = model(input=torch.randn(2, 3))
            eval_pgm = model.eval_pgm
            second = model(input=torch.randn(2, 3))

        assert "logits" in first.params
        assert first.logits.shape == (2, 2)
        assert second.logits.shape == (2, 2)
        assert model.eval_pgm is eval_pgm
        assert model._eval_pgm_stale is False

    def test_default_query_teacher_forces_copies_only_in_training(self):
        model = self._model()
        target = model.prepare_target(torch.tensor([[0.0, 1.0]]))

        train_query = model.default_query(target, step="train")
        eval_query = model.default_query(None, step="eval")

        assert set(train_query) == {"c__copy", "y__copy", "c", "y"}
        assert train_query["c"] is None
        assert train_query["y"] is None
        assert torch.allclose(train_query["c__copy"], torch.tensor([[0.0]]))
        assert eval_query == {"c": None, "y": None}

    def test_forward_rejects_ambiguous_or_missing_training_targets(self):
        model = self._model()
        model.train()

        with pytest.raises(ValueError, match="requires `target`"):
            model(input=torch.randn(1, 3))
        with pytest.raises(ValueError, match="either `query` or `target`"):
            model(
                query={"c": None},
                target=model.prepare_target(torch.tensor([[1.0, 0.0]])),
                input=torch.randn(1, 3),
            )

    @pytest.mark.parametrize(
        "kwargs, error",
        [
            (dict(task_names=[]), "at least one task"),
            (dict(task_names="missing"), "must be present"),
            (
                dict(
                    task_names="y",
                    graph=ConceptGraph(torch.eye(2), node_names=["c", "y"]),
                    graph_generator=nn.Identity(),
                ),
                "either `graph` or `graph_generator`",
            ),
            (
                dict(
                    annotations=_binary_annotations(["y"]),
                    task_names="y",
                    graph=ConceptGraph(torch.zeros(1, 1), node_names=["y"]),
                ),
                "at least one intervenable",
            ),
        ],
    )
    def test_constructor_validation(self, kwargs, error):
        base = dict(
            input_size=3,
            annotations=_binary_annotations(["c", "y"]),
            task_names="y",
            graph=ConceptGraph(torch.zeros(2, 2), node_names=["c", "y"]),
        )
        base.update(kwargs)
        with pytest.raises(ValueError, match=error):
            CausalCGM(**base)


def test_cgm_low_level_blocks_compose_for_one_manual_path():
    torch.manual_seed(0)
    ann = _mixed_annotations()
    mixer = MixConceptEmbeddingToConceptEmbedding(
        in_embeddings=3,
        concept_types=ann.types,
        cardinalities=ann.cardinalities,
        complete_output=True,
    )
    aggregator = GraphAggregator(
        adjacency=torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.5, 0.0, 0.0],
            ]
        )
    )
    equations = NeuralStructuralEquations(
        in_embeddings=3,
        out_concept_types=ann.types,
        cardinalities_out_concepts=ann.cardinalities,
    )
    embeddings = [
        torch.randn(4, 2, 3),
        torch.randn(4, 3, 3),
        torch.randn(4, 1, 3),
    ]
    values = [
        torch.rand(4, 1),
        torch.randint(0, 3, (4, 1)),
        torch.randn(4, 1),
    ]

    mixed = mixer(embeddings, values)
    aggregated = aggregator(mixed)
    output = equations(aggregated)

    assert mixed.shape == (4, 3, 3)
    assert aggregated.shape == (4, 3, 3)
    assert output.shape == (4, ann.size)


class _UnittestStylePytestReporter:
    def __init__(self):
        self.count = 0
        self.failed = False

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.count += 1
            if report.passed:
                print(".", end="", flush=True)
            elif report.failed:
                self.failed = True
                print("F", end="", flush=True)
        elif report.when in {"setup", "teardown"} and report.failed:
            self.failed = True
            print("E", end="", flush=True)


def _main():
    reporter = _UnittestStylePytestReporter()
    start = time.perf_counter()
    exit_code = pytest.main(
        [__file__, "-s", "-p", "no:terminal"],
        plugins=[reporter],
    )
    elapsed = time.perf_counter() - start
    print()
    print("-" * 70)
    print(f"Ran {reporter.count} tests in {elapsed:.3f}s")
    print()
    print("OK" if exit_code == 0 and not reporter.failed else "FAILED")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(_main())
