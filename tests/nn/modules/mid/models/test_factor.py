"""
Comprehensive tests for the Factor class.

Tests cover:
- Construction: shape validation, batched vs unbatched, cardinality checks
- product(): scope union, shared variables, einsum, batched × unbatched
- marginalize(): sum-out correctness, error on missing variable
- set_evidence(): slicing, out-of-range errors
- normalize(): partition function, batched normalisation
- _align(): permutation and unsqueezing for broadcast
- __mul__: shorthand for product
- __repr__: string representation
"""
import pytest
import torch

from torch_concepts.nn.modules.mid.models.factor import Factor, _EINSUM_SUBSCRIPTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cards(*names_and_sizes):
    """Build a cardinality dict from (name, size) pairs."""
    return {n: s for n, s in names_and_sizes}


# ===========================================================================
# Construction
# ===========================================================================

class TestFactorConstruction:
    """Tests for Factor.__init__."""

    def test_unbatched_valid(self):
        f = Factor(torch.ones(2, 3), ['A', 'B'], _cards(('A', 2), ('B', 3)))
        assert f.variables == ['A', 'B']
        assert not f.batched

    def test_batched_valid(self):
        f = Factor(torch.ones(4, 2, 3), ['A', 'B'],
                   _cards(('A', 2), ('B', 3)), batched=True)
        assert f.batched
        assert f.values.shape == (4, 2, 3)

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="dimensions"):
            Factor(torch.ones(2, 3), ['A'], _cards(('A', 2)))

    def test_wrong_card_raises(self):
        with pytest.raises(ValueError, match="cardinality"):
            Factor(torch.ones(2, 3), ['A', 'B'],
                   _cards(('A', 2), ('B', 5)))

    def test_batched_wrong_ndim(self):
        with pytest.raises(ValueError, match="dimensions"):
            Factor(torch.ones(2, 3), ['A', 'B'],
                   _cards(('A', 2), ('B', 3)), batched=True)

    def test_extra_cardinalities_ignored(self):
        cards = _cards(('A', 2), ('B', 3), ('C', 4))
        f = Factor(torch.ones(2, 3), ['A', 'B'], cards)
        assert f.variables == ['A', 'B']


# ===========================================================================
# Product
# ===========================================================================

class TestFactorProduct:
    """Tests for Factor.product (and __mul__)."""

    def test_disjoint_scopes(self):
        fa = Factor(torch.tensor([0.6, 0.4]), ['A'], _cards(('A', 2), ('B', 3)))
        fb = Factor(torch.tensor([0.1, 0.3, 0.6]), ['B'], _cards(('A', 2), ('B', 3)))
        p = fa.product(fb)
        assert set(p.variables) == {'A', 'B'}
        # Values should be the outer product
        expected = torch.tensor([0.6, 0.4]).unsqueeze(1) * torch.tensor([0.1, 0.3, 0.6]).unsqueeze(0)
        torch.testing.assert_close(p.values, expected)

    def test_shared_scope(self):
        cards = _cards(('A', 2), ('B', 2))
        fab = Factor(torch.tensor([[0.3, 0.7], [0.9, 0.1]]), ['A', 'B'], cards)
        fb = Factor(torch.tensor([0.4, 0.6]), ['B'], cards)
        p = fab.product(fb)
        assert p.variables == ['A', 'B']
        expected = torch.tensor([[0.3 * 0.4, 0.7 * 0.6],
                                 [0.9 * 0.4, 0.1 * 0.6]])
        torch.testing.assert_close(p.values, expected)

    def test_mul_shorthand(self):
        fa = Factor(torch.tensor([0.5, 0.5]), ['A'], _cards(('A', 2)))
        fb = Factor(torch.tensor([0.3, 0.7]), ['A'], _cards(('A', 2)))
        p1 = fa.product(fb)
        p2 = fa * fb
        torch.testing.assert_close(p1.values, p2.values)

    def test_batched_times_unbatched(self):
        cards = _cards(('A', 2))
        fa = Factor(torch.rand(8, 2), ['A'], cards, batched=True)
        fb = Factor(torch.tensor([0.3, 0.7]), ['A'], cards, batched=False)
        p = fa.product(fb)
        assert p.batched
        assert p.values.shape == (8, 2)
        torch.testing.assert_close(p.values, fa.values * fb.values.unsqueeze(0))

    def test_unbatched_times_batched(self):
        cards = _cards(('A', 2))
        fa = Factor(torch.tensor([0.3, 0.7]), ['A'], cards)
        fb = Factor(torch.rand(8, 2), ['A'], cards, batched=True)
        p = fa.product(fb)
        assert p.batched
        assert p.values.shape == (8, 2)

    def test_batched_times_batched(self):
        cards = _cards(('A', 2), ('B', 3))
        fa = Factor(torch.rand(4, 2), ['A'], cards, batched=True)
        fb = Factor(torch.rand(4, 3), ['B'], cards, batched=True)
        p = fa.product(fb)
        assert p.batched
        assert p.values.shape == (4, 2, 3)

    def test_scope_too_large_raises(self):
        """Product whose union scope exceeds the einsum subscript limit."""
        n = len(_EINSUM_SUBSCRIPTS)
        # Build two factors whose union scope has n+1 vars (exceeds limit)
        vars_a = [f'v{i}' for i in range(n)]
        vars_b = [f'v{n}']  # one new var beyond the limit
        # Don't put cardinalities so the constructor won't validate sizes
        fa = Factor(torch.ones(*([1] * n)), vars_a, {})
        fb = Factor(torch.ones(1), vars_b, {})
        with pytest.raises(ValueError, match="einsum limit"):
            fa.product(fb)

    def test_product_shares_cardinalities(self):
        cards = _cards(('A', 2), ('B', 3))
        fa = Factor(torch.ones(2), ['A'], cards)
        fb = Factor(torch.ones(3), ['B'], cards)
        p = fa.product(fb)
        assert p.cardinalities is cards


# ===========================================================================
# Marginalize
# ===========================================================================

class TestFactorMarginalize:
    """Tests for Factor.marginalize."""

    def test_sum_out_one_var(self):
        cards = _cards(('A', 2), ('B', 3))
        vals = torch.tensor([[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0]])
        f = Factor(vals, ['A', 'B'], cards)
        m = f.marginalize('B')
        assert m.variables == ['A']
        torch.testing.assert_close(m.values, torch.tensor([6.0, 15.0]))

    def test_sum_out_first_var(self):
        cards = _cards(('A', 2), ('B', 3))
        vals = torch.tensor([[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0]])
        f = Factor(vals, ['A', 'B'], cards)
        m = f.marginalize('A')
        assert m.variables == ['B']
        torch.testing.assert_close(m.values, torch.tensor([5.0, 7.0, 9.0]))

    def test_batched_marginalize(self):
        cards = _cards(('A', 2), ('B', 3))
        vals = torch.rand(8, 2, 3)
        f = Factor(vals, ['A', 'B'], cards, batched=True)
        m = f.marginalize('A')
        assert m.batched
        assert m.values.shape == (8, 3)
        torch.testing.assert_close(m.values, vals.sum(dim=1))

    def test_missing_variable_raises(self):
        f = Factor(torch.ones(2), ['A'], _cards(('A', 2)))
        with pytest.raises(ValueError, match="not in the factor scope"):
            f.marginalize('X')


# ===========================================================================
# set_evidence
# ===========================================================================

class TestFactorSetEvidence:
    """Tests for Factor.set_evidence."""

    def test_slice_first_var(self):
        cards = _cards(('A', 2), ('B', 3))
        vals = torch.tensor([[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0]])
        f = Factor(vals, ['A', 'B'], cards)
        e = f.set_evidence('A', 0)
        assert e.variables == ['B']
        torch.testing.assert_close(e.values, torch.tensor([1.0, 2.0, 3.0]))

    def test_slice_second_var(self):
        cards = _cards(('A', 2), ('B', 3))
        vals = torch.tensor([[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0]])
        f = Factor(vals, ['A', 'B'], cards)
        e = f.set_evidence('B', 2)
        assert e.variables == ['A']
        torch.testing.assert_close(e.values, torch.tensor([3.0, 6.0]))

    def test_batched_evidence(self):
        cards = _cards(('A', 2), ('B', 3))
        vals = torch.rand(8, 2, 3)
        f = Factor(vals, ['A', 'B'], cards, batched=True)
        e = f.set_evidence('A', 1)
        assert e.batched
        assert e.variables == ['B']
        torch.testing.assert_close(e.values, vals[:, 1, :])

    def test_missing_variable_raises(self):
        f = Factor(torch.ones(2), ['A'], _cards(('A', 2)))
        with pytest.raises(ValueError, match="not in the factor scope"):
            f.set_evidence('X', 0)

    def test_out_of_range_raises(self):
        f = Factor(torch.ones(2), ['A'], _cards(('A', 2)))
        with pytest.raises(ValueError, match="out of range"):
            f.set_evidence('A', 5)

    def test_negative_state_raises(self):
        f = Factor(torch.ones(2), ['A'], _cards(('A', 2)))
        with pytest.raises(ValueError, match="out of range"):
            f.set_evidence('A', -1)


# ===========================================================================
# Normalize
# ===========================================================================

class TestFactorNormalize:
    """Tests for Factor.normalize."""

    def test_unbatched(self):
        f = Factor(torch.tensor([2.0, 3.0, 5.0]), ['A'], _cards(('A', 3)))
        Z, normed = f.normalize()
        assert Z.item() == pytest.approx(10.0)
        torch.testing.assert_close(normed.values, torch.tensor([0.2, 0.3, 0.5]))

    def test_batched(self):
        vals = torch.tensor([[1.0, 3.0], [2.0, 2.0]])
        f = Factor(vals, ['A'], _cards(('A', 2)), batched=True)
        Z, normed = f.normalize()
        assert Z.shape == (2,)
        torch.testing.assert_close(Z, torch.tensor([4.0, 4.0]))
        torch.testing.assert_close(normed.values.sum(dim=1),
                                   torch.tensor([1.0, 1.0]))

    def test_multi_dim_unbatched(self):
        vals = torch.ones(2, 3)
        f = Factor(vals, ['A', 'B'], _cards(('A', 2), ('B', 3)))
        Z, normed = f.normalize()
        assert Z.item() == pytest.approx(6.0)
        assert normed.values.sum().item() == pytest.approx(1.0)

    def test_multi_dim_batched(self):
        vals = torch.ones(4, 2, 3)
        f = Factor(vals, ['A', 'B'], _cards(('A', 2), ('B', 3)),
                   batched=True)
        Z, normed = f.normalize()
        assert Z.shape == (4,)
        for i in range(4):
            assert normed.values[i].sum().item() == pytest.approx(1.0)


# ===========================================================================
# __repr__
# ===========================================================================

class TestFactorRepr:
    def test_repr_contains_variables(self):
        f = Factor(torch.ones(2, 3), ['A', 'B'], _cards(('A', 2), ('B', 3)))
        r = repr(f)
        assert 'A' in r and 'B' in r
        assert '[2, 3]' in r


# ===========================================================================
# Gradient flow
# ===========================================================================

class TestFactorGradientFlow:
    """Make sure all operations are differentiable."""

    def test_product_grad(self):
        a = torch.tensor([0.3, 0.7], requires_grad=True)
        b = torch.tensor([0.4, 0.6], requires_grad=True)
        cards = _cards(('A', 2), ('B', 2))
        fa = Factor(a, ['A'], cards)
        fb = Factor(b, ['B'], cards)
        p = fa.product(fb)
        p.values.sum().backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_marginalize_grad(self):
        v = torch.rand(2, 3, requires_grad=True)
        f = Factor(v, ['A', 'B'], _cards(('A', 2), ('B', 3)))
        m = f.marginalize('B')
        m.values.sum().backward()
        assert v.grad is not None

    def test_normalize_grad(self):
        v = torch.rand(2, 3, requires_grad=True)
        f = Factor(v, ['A', 'B'], _cards(('A', 2), ('B', 3)))
        _, normed = f.normalize()
        normed.values.sum().backward()
        assert v.grad is not None

    def test_set_evidence_grad(self):
        v = torch.rand(2, 3, requires_grad=True)
        f = Factor(v, ['A', 'B'], _cards(('A', 2), ('B', 3)))
        e = f.set_evidence('A', 0)
        e.values.sum().backward()
        assert v.grad is not None
