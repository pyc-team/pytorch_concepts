"""Plate-grouping behaviour for the bipartite models (CBM and CEM).

These tests exercise the shared level factories on
:class:`~torch_concepts.nn.modules.high.base.model.BaseModel`
(``build_concept_variables`` / ``build_concept_embedding_variables``) through the
two concrete models. They cover every layout the grouping can produce:

* a fully homogeneous level  -> one plate;
* several homogeneous families (binary + categorical) -> the minimum number of
  plates, both contiguous and *interleaved* in the annotation (the interleaved
  case stresses the CEM mixer, whose concept axis must follow group order);
* all-distinct concepts       -> individual variables;

and every ``plate`` preference (``None`` auto, ``True`` force, ``False`` forbid).

The load-bearing correctness check is that each queried concept/task returns
``logits`` of shape ``(batch, cardinality)`` — a wrong member slice, wrong
grouping, or scrambled concept/embedding pairing shows up as a wrong per-name
shape or a build/forward failure.
"""
import pytest
import torch
import torch.nn as nn

from torch_concepts.annotations import Annotations
from torch_concepts.nn.modules.high.models.cbm import ConceptBottleneckModel
from torch_concepts.nn.modules.high.models.cem import ConceptEmbeddingModel


# ---------------------------------------------------------------------------
# Annotation fixtures — (annotation factory, task_names)
# ---------------------------------------------------------------------------
def _ann_homo_bin():
    return Annotations(labels=['b0', 'b1', 'b2', 't'],
                       cardinalities=[1, 1, 1, 1],
                       types=['binary'] * 4)


def _ann_homo_cat():
    return Annotations(labels=['k0', 'k1', 'k2', 't'],
                       cardinalities=[3, 3, 3, 1],
                       types=['categorical', 'categorical', 'categorical', 'binary'])


def _ann_mixed():
    # 2 binary + 2 categorical concepts, types contiguous.
    return Annotations(labels=['b0', 'b1', 'k0', 'k1', 't'],
                       cardinalities=[1, 1, 3, 3, 1],
                       types=['binary', 'binary', 'categorical', 'categorical', 'binary'])


def _ann_interleaved():
    # Same families as _ann_mixed but interleaved: grouping REORDERS the concepts
    # relative to the annotation, which the CEM mixer must respect.
    return Annotations(labels=['b0', 'k0', 'b1', 'k1', 't'],
                       cardinalities=[1, 3, 1, 3, 1],
                       types=['binary', 'categorical', 'binary', 'categorical', 'binary'])


def _ann_all_distinct():
    return Annotations(labels=['a', 'b', 'c', 't'],
                       cardinalities=[2, 3, 4, 1],
                       types=['categorical', 'categorical', 'categorical', 'binary'])


CASES = {
    'homo_bin': (_ann_homo_bin, ['t']),
    'homo_cat': (_ann_homo_cat, ['t']),
    'mixed': (_ann_mixed, ['t']),
    'interleaved': (_ann_interleaved, ['t']),
    'all_distinct': (_ann_all_distinct, ['t']),
}

# Concepts (excluding the task) split, under each plate preference, into:
#   None / True -> one plate per (type, cardinality) family (a lone concept is a
#                  single-member plate);
#   False       -> one individual variable per concept.
EXPECTED_CONCEPT_VARS = {
    'homo_bin':     {None: 1, True: 1, False: 3},
    'homo_cat':     {None: 1, True: 1, False: 3},
    'mixed':        {None: 2, True: 2, False: 4},
    'interleaved':  {None: 2, True: 2, False: 4},
    'all_distinct': {None: 3, True: 3, False: 3},
}

INPUT_SIZE = 6
EMB = 8
BATCH = 4


def _make(kind, ann, task_names, plate):
    if kind == 'cbm':
        return ConceptBottleneckModel(input_size=INPUT_SIZE, annotations=ann,
                                      task_names=task_names, plate=plate)
    return ConceptEmbeddingModel(input_size=INPUT_SIZE, annotations=ann,
                                 task_names=task_names, plate=plate, embedding_size=EMB)


# ===========================================================================
# 1. Grouping counts — the "minimum number of plates" contract
# ===========================================================================
@pytest.mark.parametrize('case', list(CASES))
@pytest.mark.parametrize('plate', [None, True, False])
def test_concept_grouping_counts(case, plate):
    ann_fn, task_names = CASES[case]
    ann = ann_fn()
    m = _make('cbm', ann, task_names, plate)
    cvars = m.build_concept_variables(m.intermediate_concept_names, plate_name='concepts')

    assert len(cvars) == EXPECTED_CONCEPT_VARS[case][plate]
    # Every intermediate concept appears exactly once across the variables' members.
    members = [mem for v in cvars for mem in v.members]
    assert sorted(members) == sorted(m.intermediate_concept_names)
    if plate is False:
        # plate=False -> only individual (non-plate) variables.
        assert all(not v.is_plate for v in cvars)
    else:
        # None / True -> always plates, even single-member ones.
        assert all(v.is_plate for v in cvars)


@pytest.mark.parametrize('case', list(CASES))
@pytest.mark.parametrize('plate', [None, True, False])
def test_embeddings_align_with_concepts(case, plate):
    """CEM concept embeddings are grouped identically to the concepts and sized
    (n_states, embedding_size) per group."""
    ann_fn, task_names = CASES[case]
    ann = ann_fn()
    m = _make('cem', ann, task_names, plate)

    cvars = m.build_concept_variables(m.intermediate_concept_names, plate_name='concepts')
    evars = m.build_concept_embedding_variables(
        m.intermediate_concept_names, EMB, plate_name='embeddings')

    assert len(evars) == len(cvars)              # aligned 1:1
    for c, e in zip(cvars, evars):
        assert tuple(e.shape) == (c.size, EMB)   # rows == concept states, width == emb


# ===========================================================================
# 2. Forward — every queried name returns (batch, cardinality)
# ===========================================================================
@pytest.mark.parametrize('kind', ['cbm', 'cem'])
@pytest.mark.parametrize('case', list(CASES))
@pytest.mark.parametrize('plate', [None, True, False])
def test_forward_shapes(kind, case, plate):
    ann_fn, task_names = CASES[case]
    ann = ann_fn()
    m = _make(kind, ann, task_names, plate)
    m.eval()

    query = list(ann.labels)                     # all concepts + task
    x = torch.randn(BATCH, INPUT_SIZE)
    out = m(query=query, input=x)

    total = 0
    for name in query:
        card = ann.concept(name).cardinality
        assert name in out.params, f"{name} missing from params"
        logits = out.params[name]['logits']
        assert logits.shape == (BATCH, card), (
            f"{kind}/{case}/plate={plate}: {name} logits {tuple(logits.shape)} != {(BATCH, card)}"
        )
        total += card
    # Concatenated width matches the sum of cardinalities.
    concat = torch.cat([out.params[n]['logits'] for n in query], dim=1)
    assert concat.shape == (BATCH, total)


# ===========================================================================
# 3. Gradients flow through every layout
# ===========================================================================
@pytest.mark.parametrize('kind', ['cbm', 'cem'])
@pytest.mark.parametrize('case', list(CASES))
@pytest.mark.parametrize('plate', [None, True, False])
def test_gradients_flow(kind, case, plate):
    ann_fn, task_names = CASES[case]
    ann = ann_fn()
    m = _make(kind, ann, task_names, plate)
    m.train()

    query = list(ann.labels)
    x = torch.randn(BATCH, INPUT_SIZE)
    out = m(query=query, input=x)
    loss = torch.cat([out.params[n]['logits'] for n in query], dim=1).sum()
    loss.backward()

    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


# ===========================================================================
# 4. Interleaved types: the queried outputs are correct per concept
#    (stresses the CEM mixer's group-ordered concept axis)
# ===========================================================================
@pytest.mark.parametrize('kind', ['cbm', 'cem'])
@pytest.mark.parametrize('plate', [None, True])
def test_interleaved_types_forward(kind, plate):
    ann = _ann_interleaved()
    m = _make(kind, ann, ['t'], plate)
    m.eval()

    # Concepts are grouped into 2 plates (binary {b0,b1}, categorical {k0,k1}),
    # reordered relative to the annotation. Each concept must still come back at
    # its own cardinality.
    x = torch.randn(BATCH, INPUT_SIZE)
    out = m(query=['b0', 'k0', 'b1', 'k1', 't'], input=x)
    for name in ['b0', 'k0', 'b1', 'k1', 't']:
        card = ann.concept(name).cardinality
        assert out.params[name]['logits'].shape == (BATCH, card)


# ===========================================================================
# 5. End-to-end learning: overfit a fixed batch (validates full wiring)
# ===========================================================================
@pytest.mark.parametrize('kind', ['cbm', 'cem'])
def test_overfit_fixed_batch(kind):
    torch.manual_seed(0)
    ann = _ann_mixed()
    m = _make(kind, ann, ['t'], plate=None)
    m.train()

    query = list(ann.labels)
    total = sum(ann.concept(n).cardinality for n in query)
    x = torch.randn(BATCH, INPUT_SIZE)
    target = torch.randn(BATCH, total)           # fixed, deterministic objective

    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    losses = []
    for _ in range(200):
        opt.zero_grad()
        out = m(query=query, input=x)
        pred = torch.cat([out.params[n]['logits'] for n in query], dim=1)
        loss = nn.functional.mse_loss(pred, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # A fixed input through mostly-linear heads must be driveable toward the target.
    assert losses[-1] < 0.3 * losses[0], f"{kind}: loss {losses[0]:.3f} -> {losses[-1]:.3f}"


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
