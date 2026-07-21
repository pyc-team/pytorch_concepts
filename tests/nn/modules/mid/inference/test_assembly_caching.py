"""Output assembly is cached per query signature (BaseInference._output_labels
and ._annotate). These tests pin the *correctness* of that reuse: a cached plan
must never be handed to a query it was not built for.
"""
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.nn.modules.mid.variable import ConceptVariable, EmbeddingVariable
from torch_concepts.nn.modules.mid.factors.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.graph.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.mid.inference.torch.deterministic import DeterministicInference
from torch_concepts.nn.modules.low.priors import LearnablePrior
from torch_concepts.distributions import Delta


def _engine():
    """x -> concepts(plate of c1,c2,c3) -> y(Normal, two quantities).

    ``a`` and ``b`` are two further Bernoulli children of ``x``: two *separate*
    variables reporting the *same* quantity, which is what makes the annotation
    signature observable (see the evidence-coverage test).
    """
    x = EmbeddingVariable("x", distribution=Delta, size=4)
    concepts = ConceptVariable("concepts", members=["c1", "c2", "c3"],
                               distribution=dist.Bernoulli)
    y = ConceptVariable("y", distribution=dist.Normal, size=2)
    a = ConceptVariable("a", distribution=dist.Bernoulli)
    b = ConceptVariable("b", distribution=dist.Bernoulli)
    factors = [
        ParametricCPD(x, parametrization=LearnablePrior(4)),
        ParametricCPD(concepts, parents=[x],
                      parametrization=nn.Sequential(nn.Linear(4, 3), nn.Sigmoid())),
        ParametricCPD(y, parents=[concepts], parametrization={
            "loc": nn.Linear(3, 2),
            "scale": nn.Sequential(nn.Linear(3, 2), nn.Softplus()),
        }),
        ParametricCPD(a, parents=[x],
                      parametrization=nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())),
        ParametricCPD(b, parents=[x],
                      parametrization=nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())),
    ]
    return DeterministicInference(
        BayesianNetwork(variables=[x, concepts, y, a, b], factors=factors)
    )


def _labels(out):
    return {q: list(t.annotation.labels) for q, t in out.params.items()}


# ---------------------------------------------------------------- reuse ----

def test_output_labels_reuses_the_cached_chunks():
    eng = _engine()
    first = eng._output_labels(["concepts"])
    assert eng._output_labels(["concepts"]) is first  # same object, not rebuilt
    assert eng._output_labels(["c1"]) is not first    # different signature


def test_query_order_is_part_of_the_signature():
    """Labels follow query order, so the same names in a different order are a
    different plan — the cache must not treat the two as interchangeable."""
    eng = _engine()
    x = torch.randn(2, 4)
    assert _labels(eng.query(query=["a", "b"], evidence={"x": x})) == {"probs": ["a", "b"]}
    assert _labels(eng.query(query=["b", "a"], evidence={"x": x})) == {"probs": ["b", "a"]}
    # and the plate case, where reordering changes which chunk is the remainder
    assert _labels(eng.query(query=["c2", "concepts"], evidence={"x": x})) == {
        "probs": ["c2", "c1", "c3"]}
    assert _labels(eng.query(query=["concepts", "c2"], evidence={"x": x})) == {
        "probs": ["c1", "c2", "c3"]}


def test_annotation_is_reused_across_identical_queries():
    eng = _engine()
    x = torch.randn(2, 4)
    a = eng.query(query=["concepts"], evidence={"x": x})
    b = eng.query(query=["concepts"], evidence={"x": torch.randn(2, 4)})
    # Same signature -> the very same Annotations instance backs both outputs.
    assert a.params["probs"].annotation is b.params["probs"].annotation


# ----------------------------------------------------------- correctness ----

def test_labels_are_stable_across_repeated_and_interleaved_queries():
    """Interleaving different queries through one engine must not let a cached
    plan leak into a query it was not built for."""
    eng = _engine()
    x = torch.randn(5, 4)
    cases = [
        (["concepts"], {"probs": ["c1", "c2", "c3"]}),
        (["c2"], {"probs": ["c2"]}),
        (["c1", "c3"], {"probs": ["c1", "c3"]}),
        (["y"], {"loc": ["y"], "scale": ["y"]}),
        (["concepts", "y"],
         {"probs": ["c1", "c2", "c3"], "loc": ["y"], "scale": ["y"]}),
    ]
    for _ in range(3):  # first round fills the caches, later rounds hit them
        for names, expected in cases:
            out = eng.query(query=names, evidence={"x": x})
            assert _labels(out) == expected, names
            for tensor in out.params.values():
                assert tensor.shape[-1] == sum(tensor.annotation.cardinalities)


def test_member_then_plate_keeps_its_own_label_order():
    """``[member, plate]`` emits the member first and the plate's *remaining*
    members after it. Two such queries have identical label counts and widths,
    so a signature that ignored the member name would collide."""
    eng = _engine()
    x = torch.randn(2, 4)
    first = eng.query(query=["c2", "concepts"], evidence={"x": x})
    second = eng.query(query=["c3", "concepts"], evidence={"x": x})
    assert _labels(first) == {"probs": ["c2", "c1", "c3"]}
    assert _labels(second) == {"probs": ["c3", "c1", "c2"]}


def test_same_query_names_with_different_evidence_coverage():
    """Observing a queried variable drops it from ``params``; the annotation for
    the same query names must change with it rather than come back stale.

    ``a`` and ``b`` both report ``probs``, so observing one leaves the *same*
    query names and the *same* quantity with a different label set — the case a
    signature keyed only on (query names, quantity) would get wrong.
    """
    eng = _engine()
    x = torch.randn(2, 4)
    both = eng.query(query=["a", "b"], evidence={"x": x})
    assert _labels(both) == {"probs": ["a", "b"]}
    assert both.params["probs"].shape == (2, 2)

    one = eng.query(query=["a", "b"], evidence={"x": x, "b": torch.ones(2, 1)})
    assert _labels(one) == {"probs": ["a"]}
    assert one.params["probs"].shape == (2, 1)

    # ...and the full-coverage annotation is still intact afterwards.
    again = eng.query(query=["a", "b"], evidence={"x": x})
    assert _labels(again) == {"probs": ["a", "b"]}


def test_mixed_quantity_coverage_change():
    """Same as above across families: dropping the Bernoulli side must not
    disturb the Normal side's annotation."""
    eng = _engine()
    x = torch.randn(2, 4)
    free = eng.query(query=["concepts", "y"], evidence={"x": x})
    observed = eng.query(query=["concepts", "y"],
                         evidence={"x": x, "concepts": torch.rand(2, 3)})
    assert _labels(free) == {"probs": ["c1", "c2", "c3"],
                             "loc": ["y"], "scale": ["y"]}
    assert _labels(observed) == {"loc": ["y"], "scale": ["y"]}


def test_member_subset_columns_match_the_whole_plate():
    """Label slicing off a subset query must give the same columns as slicing
    the same labels out of the whole plate's tensor."""
    eng = _engine()
    x = torch.randn(4, 4)
    whole = eng.query(query=["concepts"], evidence={"x": x}).params["probs"]
    subset = eng.query(query=["c3", "c1"], evidence={"x": x}).params["probs"]
    assert list(subset.annotation.labels) == ["c3", "c1"]
    assert torch.allclose(subset["c1"].tensor, whole["c1"].tensor)
    assert torch.allclose(subset["c3"].tensor, whole["c3"].tensor)


def test_samples_assembly_is_cached_consistently():
    """_assemble_samples shares the cached chunks with _assemble_params but
    keys its annotation separately, so both must stay correct."""
    eng = _engine()
    per_variable = {"concepts": torch.rand(3, 3), "y": torch.randn(3, 2)}
    for _ in range(2):
        samples = eng._assemble_samples(per_variable, ["concepts", "y"])
        assert list(samples.annotation.labels) == ["c1", "c2", "c3", "y"]
        assert samples.shape == (3, 5)
        subset = eng._assemble_samples(per_variable, ["c2"])
        assert list(subset.annotation.labels) == ["c2"]
        assert subset.shape == (3, 1)


def test_label_metadata_and_types_survive_the_cache():
    eng = _engine()
    x = torch.randn(2, 4)
    for _ in range(2):
        out = eng.query(query=["concepts", "y"], evidence={"x": x})
        probs, loc = out.params["probs"].annotation, out.params["loc"].annotation
        assert probs.types == ["binary", "binary", "binary"]
        assert loc.types == ["continuous"]
        assert probs.metadata["c1"]["variable"] == "concepts"
        assert probs.metadata["c1"]["distribution"] == "Bernoulli"
        assert loc.metadata["y"]["distribution"] == "Normal"
