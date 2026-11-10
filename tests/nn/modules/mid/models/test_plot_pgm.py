"""Tests for plot_pgm, the drawing of a model's PGM to a figure file."""
import math
from itertools import combinations

import pytest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal

from torch_concepts import Annotations, ConceptGraph
from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    BayesianNetwork,
    CausallyReliableConceptBottleneckModel,
    ConceptBottleneckModel,
    ConceptEmbeddingModel,
    ConceptMemoryReasoner,
    ConceptVariable,
    DeterministicInference,
    EmbeddingVariable,
    LearnablePrior,
    MarkovNetwork,
    ParametricCPD,
    ParametricPotential,
    plot_pgm,
)
from torch_concepts.nn.modules.mid.graph.plot import _FILLS, _GAP, _bend, _draw, _layer_name, _module_name

plt = pytest.importorskip("matplotlib.pyplot", reason="plot_pgm needs matplotlib")


def _annotations():
    return Annotations(labels=["c1", "c2", "task"], cardinalities=[1, 1, 1])


def _cbm():
    return ConceptBottleneckModel(input_size=4, annotations=_annotations(), task_names=["task"])


def _c2bm():
    graph = ConceptGraph(torch.tensor([[0, 1, 1], [0, 0, 1], [0, 0, 0]]), node_names=["c1", "c2", "task"])
    return CausallyReliableConceptBottleneckModel(input_size=4, annotations=_annotations(), graph=graph)


def _cmr():
    return ConceptMemoryReasoner(input_size=4, annotations=_annotations(), task_names=["task"])


def _bayesian_network():
    """x -> y, with a Normal y behind a trunk."""
    x = EmbeddingVariable("x", distribution=Delta, size=4)
    y = ConceptVariable("y", distribution=Normal)
    return BayesianNetwork(variables=[x, y], factors=[
        ParametricCPD(x, parents=[], parametrization=LearnablePrior(4)),
        ParametricCPD(y, parents=[x], trunk=nn.Sequential(nn.Linear(4, 8), nn.ReLU()),
                      parametrization={"loc": nn.Linear(8, 1),
                                       "scale": nn.Sequential(nn.Linear(8, 1), nn.Softplus())}),
    ])


def _markov_network():
    a, b, c = (ConceptVariable(n, distribution=Bernoulli) for n in ["a", "b", "c"])
    return MarkovNetwork(variables=[a, b, c], factors=[
        ParametricPotential(scope=[a], parametrization=nn.Linear(1, 1, bias=False)),
        ParametricPotential(scope=[a, b, c], parametrization=nn.Linear(3, 1)),
    ])


def _is_pdf(path):
    return path.read_bytes().startswith(b"%PDF")


@pytest.fixture
def drawn():
    """Draw a PGM; the figure is closed after the test."""
    figures = []

    def draw(pgm):
        figures.append(_draw(pgm))
        figures[-1].canvas.draw()
        return figures[-1]

    yield draw
    for fig in figures:
        plt.close(fig)


def _texts(fig):
    """(layer names, {node name: text}) drawn on ``fig``."""
    texts = fig.axes[0].texts
    return ([t for t in texts if t.get_gid() == "layer_name"],
            {t.get_text(): t for t in texts if t.get_gid() != "layer_name"})


class TestPlotPGM:
    def test_a_high_level_model(self, tmp_path):
        path = tmp_path / "cbm.pdf"
        plot_pgm(_cbm(), path)
        assert _is_pdf(path)

    def test_a_model_with_an_edge_skipping_a_row(self, tmp_path):
        # CEM: tasks has parents concepts and embeddings, one row above concepts
        path = tmp_path / "cem.pdf"
        plot_pgm(ConceptEmbeddingModel(input_size=4, annotations=_annotations(), task_names=["task"]), path)
        assert _is_pdf(path)

    def test_an_inference_engine(self, tmp_path):
        path = tmp_path / "engine.pdf"
        plot_pgm(DeterministicInference(_bayesian_network()), path)
        assert _is_pdf(path)

    def test_a_probabilistic_model(self, tmp_path):
        path = tmp_path / "pgm.pdf"
        plot_pgm(_bayesian_network(), path)
        assert _is_pdf(path)

    def test_an_undirected_model(self, tmp_path):
        path = tmp_path / "markov.pdf"
        plot_pgm(_markov_network(), path)
        assert _is_pdf(path)

    def test_the_format_follows_the_extension(self, tmp_path):
        path = tmp_path / "pgm.png"
        plot_pgm(_bayesian_network(), path, layer_names=False)
        assert path.read_bytes().startswith(b"\x89PNG")

    def test_a_model_without_a_pgm_is_rejected(self, tmp_path):
        with pytest.raises(TypeError, match="Linear has no probabilistic model"):
            plot_pgm(nn.Linear(2, 2), tmp_path / "pgm.pdf")


class TestDrawing:
    def test_a_directed_model_goes_top_to_bottom(self, drawn):
        _, nodes = _texts(drawn(_cbm().pgm))
        chain = [nodes[name].get_position() for name in ["input", "latent", "concepts", "tasks"]]
        assert all(a[0] == b[0] and a[1] > b[1] for a, b in zip(chain, chain[1:]))

    def test_a_layer_name_is_right_of_its_child_level_with_its_top(self, drawn):
        pgm = _cbm().pgm
        fig = drawn(pgm)
        names, nodes = _texts(fig)
        name = next(t for t in names if t.get_text() == "LinearConceptToConcept")
        (x, y), (node_x, node_y) = name.get_position(), nodes["tasks"].get_position()
        size = fig.axes[0].collections[0].get_sizes()[list(pgm.variables).index("tasks")]
        radius = math.sqrt(size) / 2 / 72  # one data unit is one inch
        assert x > node_x + radius and y == pytest.approx(node_y + radius)

    def test_a_root_sits_right_above_its_first_child(self, drawn):
        # CMR: rule_roles has no parent and feeds the tasks, three rows below the input
        _, nodes = _texts(drawn(_cmr().pgm))
        assert nodes["rule_roles"].get_position()[1] == nodes["concepts"].get_position()[1]

    def test_concepts_are_yellow_and_embeddings_blue(self, drawn):
        pgm = _cbm().pgm
        fills = drawn(pgm).axes[0].collections[0].get_facecolors()
        expected = [_FILLS[ConceptVariable] if isinstance(v, ConceptVariable) else _FILLS[EmbeddingVariable]
                    for v in pgm.variables.values()]
        assert [tuple(f) for f in fills] == [plt.matplotlib.colors.to_rgba(c) for c in expected]

    def test_one_layer_name_per_factor_with_edges(self, drawn):
        # C2BM: every CPD but the input's prior has parents
        pgm = _c2bm().pgm
        names, _ = _texts(drawn(pgm))
        assert sorted(t.get_text() for t in names) == sorted(
            _layer_name(f) for f in pgm.factors.values() if f.parents)

    def test_a_layer_class_has_one_color(self, drawn):
        names, _ = _texts(drawn(_c2bm().pgm))
        colors = {}
        for text in names:
            colors.setdefault(text.get_text(), set()).add(text.get_color())
        assert all(len(c) == 1 for c in colors.values())
        assert len({c.pop() for c in colors.values()}) == len(colors)  # 4 classes, 4 colors

    @pytest.mark.parametrize("model", [_c2bm, _cmr])
    def test_layer_names_overlap_neither_each_other_nor_the_nodes(self, drawn, model):
        fig = drawn(model().pgm)
        renderer = fig.canvas.get_renderer()
        boxes = [t.get_window_extent(renderer) for t in _texts(fig)[0]]
        assert not any(a.overlaps(b) for a, b in combinations(boxes, 2))

        nodes = fig.axes[0].collections[0]
        centers = fig.axes[0].transData.transform(nodes.get_offsets())
        radii = [math.sqrt(size) / 2 * fig.dpi / 72 for size in nodes.get_sizes()]
        for box in boxes:
            for (x, y), radius in zip(centers, radii):
                nearest = (min(max(x, box.x0), box.x1), min(max(y, box.y0), box.y1))
                assert math.dist(nearest, (x, y)) > radius


class TestLayerName:
    def test_a_single_layer_is_named_by_its_class(self):
        assert _layer_name(_cbm().pgm.factors["tasks"]) == "LinearConceptToConcept"

    def test_a_trunk_and_several_parameters_get_one_line_each(self):
        assert _layer_name(_bayesian_network().factors["y"]) == (
            "trunk: Sequential(Linear, ReLU)\nloc: Linear\nscale: Sequential(Linear, Softplus)")

    def test_a_sequential_lists_its_modules_also_nested(self):
        assert _module_name(nn.Sequential(nn.Linear(2, 2), nn.Sequential(nn.ReLU(), nn.Dropout()))) == (
            "Sequential(Linear, Sequential(ReLU, Dropout))")

    def test_a_potential(self):
        potentials = list(_markov_network().factors.values())
        assert _layer_name(potentials[1]) == "Linear"


class TestBend:
    radius = {"a": 0.25, "b": 0.25, "c": 0.25}

    def test_an_edge_with_nothing_in_the_way_is_straight(self):
        pos = {"a": (0.0, 0.0), "b": (2.0, -1.0), "c": (0.0, -2.0)}
        assert _bend(pos, self.radius, "a", "c") == 0

    def test_an_edge_bends_around_a_node_in_the_way_and_clears_it(self):
        pos = {"a": (0.0, 0.0), "b": (0.0, -1.0), "c": (0.0, -2.0)}
        rad = _bend(pos, self.radius, "a", "c")
        # an arc3 departs from the straight edge by rad * length / 2 halfway, where b is
        assert rad * 2 / 2 >= self.radius["b"] + _GAP
