"""Tests for concept graph."""

import pytest
import torch
from torch_concepts import ConceptGraph
import sys
from ._helpers import _learnable
from ._helpers import _seed_weights
from torch_concepts.concept_graph import get_predecessors
from torch_concepts.concept_graph import get_successors
from torch_concepts.concept_graph import to_networkx_graph
from ._helpers import _materialize


def test_load_exported_graph_preserves_values_and_node_order(tmp_path):
    generator = _learnable(
        "dagma_cgm", concept_names=["b", "a", "isolated"],
        initialization=_seed_weights(torch.tensor([
            [0., .5, 0.], [0., 0., 0.], [0., 0., 0.],
        ])),
    ).double().eval()
    path = tmp_path / "graph.pt"
    _materialize(generator).save(path)
    graph = ConceptGraph.load(path, map_location=torch.device("cpu"))
    assert graph.node_names == ["b", "a", "isolated"]
    assert graph.data.device.type == "cpu"
    assert graph.data.dtype == torch.float64
    assert not graph.data.requires_grad
    torch.testing.assert_close(graph.data, _materialize(generator).data)
    assert graph.has_edge("b", "a")
    assert graph.is_dag()


@pytest.mark.parametrize("payload", [
    {}, torch.zeros(2, 2),
    {"adjacency": [[0]], "node_names": ["a"]},
    {"adjacency": torch.zeros(1, 1), "node_names": None},
    {"adjacency": torch.zeros(1, 1), "node_names": [0]},
    {"adjacency": torch.zeros(2, 3), "node_names": ["a", "b"]},
    {"adjacency": torch.zeros(2, 2), "node_names": ["a"]},
])
def test_load_graph_rejects_invalid_payloads(tmp_path, payload):
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        ConceptGraph.load(path)


def test_sparse_constructor_preserves_gradients_for_zero_weights():
    weights = torch.tensor([0., .5], requires_grad=True)
    indices = torch.tensor([[0, 1], [1, 0]])
    graph = ConceptGraph.from_sparse(indices, weights, 2)
    assert graph.edge_index is indices
    assert graph.edge_weight is weights
    torch.testing.assert_close(graph.edge_index, indices)
    torch.testing.assert_close(graph.edge_weight, weights)
    graph.data.sum().backward()
    torch.testing.assert_close(weights.grad, torch.ones_like(weights))
    changed_weights = graph.edge_weight
    with torch.no_grad():
        changed_weights[0] = 1
    graph = ConceptGraph.from_sparse(graph.edge_index, changed_weights, 2)
    torch.testing.assert_close(graph.edge_weight, torch.tensor([1., .5]))
    assert graph.data[0, 1] == 1
    # Dense access reconstructs a tensor; editing it does not alter sparse data.
    with torch.no_grad():
        graph.data[0, 1] = 2
    assert graph.edge_weight[0] == 1


@pytest.mark.parametrize("n_nodes", [0, 3])
def test_dense_training_graph_keeps_zero_pairs_and_their_gradients(n_nodes):
    adjacency = torch.zeros(n_nodes, n_nodes).T.requires_grad_()
    graph = ConceptGraph(adjacency)
    assert graph.edge_index.shape == (2, n_nodes ** 2)
    assert graph.edge_weight.shape == (n_nodes ** 2,)
    assert graph.is_dag()
    assert graph.get_root_nodes() == graph.node_names
    coefficients = torch.arange(n_nodes ** 2).reshape(n_nodes, n_nodes) + 1
    (graph.data * coefficients).sum().backward()
    torch.testing.assert_close(adjacency.grad, coefficients.float())
    with torch.no_grad():
        converted = graph.to(dtype=torch.float64).clone()
    torch.testing.assert_close(converted.edge_index, graph.edge_index)
    torch.testing.assert_close(converted.edge_weight, graph.edge_weight.double())


def test_sparse_clone_and_conversion_preserve_zero_candidates():
    indices = torch.tensor([[0, 1], [1, 0]])
    graph = ConceptGraph.from_sparse(indices, torch.zeros(2), 3)
    converted = graph.clone().to(dtype=torch.float64)
    torch.testing.assert_close(converted.edge_index, indices)
    assert converted.edge_weight.shape == (2,)
    assert converted.n_nodes == 3
    converted.edge_weight[0] = 1
    assert graph.edge_weight[0] == 0
    assert converted.has_edge(0, 1)


def test_dense_parameter_in_no_grad_produces_detached_weights():
    adjacency = torch.nn.Parameter(torch.tensor([[0., .5], [0., 0.]]))
    with torch.no_grad():
        graph = ConceptGraph(adjacency)
    assert not graph.edge_weight.requires_grad
    assert not graph.data.requires_grad
    torch.testing.assert_close(graph.data, adjacency)


def test_is_dag_matches_networkx_without_modifying_graph():
    import networkx as nx
    from torch_concepts.graphs.utils import contains_cycle

    # Includes DAGs with converging paths, disconnected cycles and self-loops.
    for mask in range(1 << 9):
        adjacency = torch.tensor([(mask >> bit) & 1 for bit in range(9)]).reshape(3, 3)
        for weights in (adjacency, adjacency.float() * -.4):
            before = weights.clone()
            network = nx.from_numpy_array(weights.numpy(), create_using=nx.DiGraph)
            assert ConceptGraph(weights).is_dag() == nx.is_directed_acyclic_graph(network)
            assert contains_cycle(weights) == (not nx.is_directed_acyclic_graph(network))
            torch.testing.assert_close(weights, before)
    assert ConceptGraph(torch.empty(0, 0)).is_dag()
    assert not contains_cycle(torch.empty(0, 0))


def test_names_and_indices_use_the_same_node_order():
    adjacency = torch.tensor([[0., .4, .8], [0., 0., 0.], [0., .2, 0.]])
    graph = ConceptGraph(adjacency, ["10", "a:b", "node"])
    assert graph.get_predecessors("a:b") == ["10", "node"]
    assert graph.get_predecessors(1) == ["10", "node"]
    assert get_predecessors(adjacency, 1) == [0, 2]
    assert get_successors(adjacency, 0) == [1, 2]
    assert get_predecessors(graph, 1) == ["10", "node"]
    renamed = ["x", "y", "z"]
    assert get_predecessors(graph, "y", renamed) == ["x", "z"]
    assert get_predecessors(graph, 1, renamed) == ["x", "z"]
    assert list(to_networkx_graph(adjacency, renamed).nodes) == renamed
    with pytest.raises(ValueError):
        graph["0", "a:b"]  # A numeric string is a name, not an index.
    with pytest.raises(IndexError):
        graph.get_successors(3)


def test_boolean_adjacency_keeps_names_and_directions():
    graph = ConceptGraph(torch.tensor([[False, True], [False, False]]), ["b", "a"])
    assert graph.is_dag()
    assert graph.get_successors("b") == ["a"]


@pytest.mark.parametrize("indices,weights,n_nodes", [
    (torch.tensor([[0, 0], [1, 1]]), torch.ones(2), 2),
    (torch.tensor([0, 1]), torch.ones(1), 2),
    (torch.tensor([[0.], [1.]]), torch.ones(1), 2),
    (torch.tensor([[0], [2]]), torch.ones(1), 2),
    (torch.tensor([[-1], [0]]), torch.ones(1), 2),
    (torch.tensor([[0], [1]]), torch.ones(2), 2),
    (torch.empty(2, 0, dtype=torch.long), torch.empty(0), -1),
])
def test_sparse_constructor_rejects_ambiguous_or_invalid_edges(indices, weights, n_nodes):
    with pytest.raises(ValueError):
        ConceptGraph.from_sparse(indices, weights, n_nodes)


@pytest.mark.parametrize("encoding", ["weights", "endpoints"])
def test_encoding_and_gradients_survive_conversion_and_save(tmp_path, encoding):
    adjacency = torch.tensor([[0., -.3], [0., 0.]], requires_grad=True)
    graph = ConceptGraph(adjacency, ["z", "a"], edge_encoding=encoding)
    converted = graph.to(dtype=torch.float64).clone()
    assert converted.edge_encoding == encoding
    assert converted.node_names == ["z", "a"]
    converted.data.sum().backward()
    torch.testing.assert_close(adjacency.grad, torch.ones_like(adjacency))
    converted.save(tmp_path / "graph.pt")
    loaded = ConceptGraph.load(tmp_path / "graph.pt")
    assert loaded.edge_encoding == encoding and loaded.node_names == ["z", "a"]
    assert loaded.data.dtype == torch.float64


@pytest.mark.parametrize("encoding,expected_edges", [("weights", 1), ("endpoints", 2)])
def test_plot_distinguishes_weights_endpoints_and_self_loops(tmp_path, monkeypatch, encoding, expected_edges):
    import io
    import pydot
    from PIL import Image
    rendered = []
    image = io.BytesIO()
    Image.new("RGB", (1, 1)).save(image, format="PNG")

    def render(dot, **kwargs):
        rendered.append(dot)
        return image.getvalue()

    monkeypatch.setattr(pydot.Dot, "create_png", render)
    graph = ConceptGraph(torch.tensor([[.2, -1.], [-1., 0.]]), ["a:b", "node"], edge_encoding=encoding)
    graph.plot(tmp_path / "plot.png")
    assert len(rendered[0].get_edges()) == expected_edges
    assert any(edge.get_source() == edge.get_destination() for edge in rendered[0].get_edges())
    assert [node.get_label() for node in rendered[0].get_nodes()] == graph.node_names


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_plot_binarizes_weights_like_causal_cgm_without_changing_graph(tmp_path, monkeypatch, dtype):
    import io
    import pydot
    from PIL import Image

    image = io.BytesIO()
    Image.new("RGB", (1, 1)).save(image, format="PNG")
    rendered = []

    def render(dot, **kwargs):
        rendered.append(dot)
        return image.getvalue()

    monkeypatch.setattr(pydot.Dot, "create_png", render)
    adjacency = torch.tensor([
        [.005, .009, .01],
        [.011, .02, -.8],
        [.8, 0., .01],
    ], dtype=dtype, requires_grad=True)
    original = adjacency.detach().clone()
    graph = ConceptGraph(adjacency, ["a", "b", "c"], edge_encoding="weights")
    result = graph.plot(tmp_path / "weighted")

    assert result.exists() and result.suffix == ".png"
    assert {(edge.get_source(), edge.get_destination())
            for edge in rendered[0].get_edges()} == {
        ("n1", "n0"), ("n1", "n1"), ("n2", "n0"),
    }
    assert [node.get_label() for node in rendered[0].get_nodes()] == graph.node_names
    torch.testing.assert_close(graph.data, original)
    graph.data.sum().backward()
    torch.testing.assert_close(adjacency.grad, torch.ones_like(adjacency))


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("field,value", [
    ("edge_index", [[0, 1], [1, 2]]),
    ("edge_index", torch.tensor([0, 1])),
    ("edge_index", torch.tensor([[0., 1.], [1., 2.]])),
    ("edge_index", torch.tensor([[0, 1], [1, 3]])),
    ("edge_index", torch.tensor([[-1, 1], [1, 2]])),
    ("edge_index", torch.tensor([[0, 0], [1, 1]])),
    ("edge_index", torch.tensor([[0], [1]])),
    ("edge_weight", [1., 2.]),
    ("edge_weight", torch.ones(1)),
    ("edge_weight", torch.ones(2, 1)),
    ("edge_weight", torch.ones(2, device="meta")),
])
def test_invalid_sparse_setters_preserve_graph_and_cache(field, value, requires_grad):
    adjacency = torch.tensor([[0., .5, 0.], [0., 0., .8], [0., 0., 0.]], requires_grad=requires_grad)
    graph = ConceptGraph(adjacency, ["a", "b", "c"])
    network = graph.to_networkx()
    with pytest.raises(ValueError):
        setattr(graph, field, value)
    torch.testing.assert_close(graph.data, adjacency)
    assert graph.to_networkx() is network


@pytest.mark.parametrize("encoding,adjacency,expected", [
    ("weights", [[0., 1.], [1., 0.]], True),
    ("endpoints", [[0., 1.], [0., 0.]], True),
    ("endpoints", [[0., -1.], [-1., 0.]], False),
])
def test_fully_directed_uses_edge_encoding(encoding, adjacency, expected):
    assert ConceptGraph(torch.tensor(adjacency), edge_encoding=encoding).is_fully_directed() is expected


@pytest.mark.parametrize("adjacency,expected", [
    ([[0., 1.], [0., 0.]], ("n0", "n1", None)),
    ([[0., 0.], [1., 0.]], ("n1", "n0", None)),
    ([[0., -1.], [-1., 0.]], ("n0", "n1", "none")),
    ([[0., 1.], [2., 0.]], ("n0", "n1", "both")),
    ([[0., -1.], [1., 0.]], ("n0", "n1", "both")),
    ([[0., 0.], [0., 0.]], None),
])
def test_plot_preserves_endpoint_directions(tmp_path, monkeypatch, adjacency, expected):
    import io
    import pydot
    from PIL import Image

    image = io.BytesIO()
    Image.new("RGB", (1, 1)).save(image, format="PNG")
    rendered = []

    def render(dot, **kwargs):
        rendered.append(dot)
        return image.getvalue()

    monkeypatch.setattr(pydot.Dot, "create_png", render)
    graph = ConceptGraph(torch.tensor(adjacency), edge_encoding="endpoints")
    result = graph.plot(tmp_path / "graph")
    assert result.suffix == ".png" and result.exists()
    edges = rendered[0].get_edges()
    if expected is None:
        assert not edges
    else:
        assert len(edges) == 1
        edge = edges[0]
        assert (edge.get_source(), edge.get_destination(), edge.get_dir()) == expected
        if adjacency[1][0] == 2:
            assert edge.get_arrowhead() == "odot"


def test_plot_reports_missing_dependencies_and_graphviz(tmp_path, monkeypatch):
    import pydot

    graph = ConceptGraph(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="rankdir"):
        graph.plot(tmp_path / "graph", rankdir="invalid")
    with monkeypatch.context() as missing:
        missing.setitem(sys.modules, "pydot", None)
        with pytest.raises(ImportError, match="requires.*pydot"):
            graph.plot(tmp_path / "graph")

    def fail(*args, **kwargs):
        raise OSError("dot unavailable")

    monkeypatch.setattr(pydot.Dot, "create_png", fail)
    with pytest.raises(RuntimeError, match="Graphviz could not render"):
        graph.plot(tmp_path / "graph")


@pytest.mark.parametrize("data,names,encoding", [
    ([[0., 1.], [0., 0.]], None, "weights"),
    (torch.zeros(2, 2), "ab", "weights"),
    (torch.zeros(2, 2), ["a", "a"], "weights"),
    (torch.zeros(2, 2), ["a", 1], "weights"),
    (torch.zeros(2, 2), None, "unknown"),
])
def test_dense_constructor_rejects_invalid_types_and_names(data, names, encoding):
    with pytest.raises(ValueError):
        ConceptGraph(data, names, edge_encoding=encoding)


@pytest.mark.parametrize("names,encoding", [
    ("ab", "weights"), (["a"], "weights"),
    (["a", "a"], "weights"), (None, "unknown"),
])
def test_sparse_constructor_rejects_invalid_names_and_encoding(names, encoding):
    with pytest.raises(ValueError):
        ConceptGraph.from_sparse(torch.tensor([[0], [1]]), torch.ones(1), 2, names, edge_encoding=encoding)


def test_dense_weight_setter_preserves_input_and_new_weight_gradients():
    adjacency = torch.tensor([[0., .5], [0., 0.]], requires_grad=True)
    graph = ConceptGraph(adjacency)
    weights = torch.tensor([0., .8, 0., 0.], requires_grad=True)
    graph.edge_weight = weights
    graph.data.sum().backward()
    assert adjacency.grad is None
    assert adjacency[0, 1] == .5
    torch.testing.assert_close(weights.grad, torch.ones_like(weights))


def test_original_dense_row_access_and_sparse_method_threshold():
    adjacency = torch.tensor([[0., -.2, .7], [0., 0., .9], [0., 0., 0.]])
    graph = ConceptGraph(adjacency)
    torch.testing.assert_close(graph[0], adjacency[0])
    indices, weights = graph.dense_to_sparse(threshold=.6)
    torch.testing.assert_close(indices, torch.tensor([[0, 1], [2, 2]]))
    torch.testing.assert_close(weights, torch.tensor([.7, .9]))
    with pytest.raises(ValueError, match="contains a cycle"):
        ConceptGraph(torch.tensor([[0., 1.], [1., 0.]])).get_levels()


@pytest.mark.parametrize("helper", [get_predecessors, get_successors])
def test_neighbor_helpers_handle_named_networkx_and_invalid_indices(helper):
    graph = ConceptGraph(torch.tensor([[0., 1.], [0., 0.]]), ["a", "b"])
    node = 1 if helper is get_predecessors else 0
    expected = ["a"] if helper is get_predecessors else ["b"]
    assert helper(graph.to_networkx(), node, graph.node_names) == expected
    with pytest.raises(IndexError):
        helper(graph, 2)


@pytest.mark.parametrize("requires_grad", [False, True])
def test_valid_sparse_setters_invalidate_cache_and_preserve_gradients(requires_grad):
    graph = ConceptGraph(torch.tensor([[0., 1.], [0., 0.]], requires_grad=requires_grad), ["a", "b"])
    network = graph.to_networkx()
    graph.edge_index = graph.edge_index.flip(0)
    assert graph.to_networkx() is not network
    assert graph.get_successors("b") == ["a"]
    network = graph.to_networkx()
    weights = (graph.edge_weight.detach() * .7).requires_grad_()
    graph.edge_weight = weights
    assert graph.to_networkx() is not network
    assert graph.get_edge_weight("b", "a") == pytest.approx(.7)
    graph.data.sum().backward()
    torch.testing.assert_close(weights.grad, torch.ones_like(weights))


@pytest.mark.parametrize("names", [["b", "a"], ["a", "changed"], ["a"], ["a", "b", "c"]])
def test_graph_rejects_modified_names_even_with_networkx_cache(names, tmp_path):
    graph = ConceptGraph(torch.tensor([[0., 1.], [0., 0.]]), ["a", "b"])
    graph.to_networkx()
    graph.node_names[:] = names
    operations = [
        lambda: graph.data, lambda: graph["a", "b"],
        lambda: graph.get_edge_weight("a", "b"), lambda: graph.get_successors("a"),
        graph.get_root_nodes, graph.to_networkx, graph.clone,
        lambda: graph.save(tmp_path / "graph.pt"),
    ]
    for operation in operations:
        with pytest.raises(ValueError, match="Node names and order"):
            operation()
