"""PgmpyBaseInference — base class for pgmpy-backed inference engines."""

from __future__ import annotations

from ..base import BaseInference


def _import_pgmpy():
    """Lazily import the pgmpy pieces the engines build their models out of.

    pgmpy is a hard dependency (it is in ``requirements.txt``), so the
    ``ImportError`` branch is a courtesy for a broken environment rather than a
    real optional-backend guard. The reason the import is deferred at all is
    **cost**: ``import pgmpy.inference`` pulls in pandas, networkx and
    opt_einsum, adding ~0.4 s to ``import torch_concepts.nn`` — which every user
    pays, because :mod:`torch_concepts.nn` imports all engines eagerly, and
    which only this backend needs. ``sys.modules`` makes every call after the
    first free, so the one call site (at the top of ``query``) costs nothing.
    """
    try:
        from pgmpy.factors.discrete import DiscreteFactor
        from pgmpy.inference import VariableElimination
        from pgmpy.models import MarkovNetwork
    except ImportError as exc:  # pragma: no cover - pgmpy not installed
        raise ImportError(
            "pgmpy-based inference requires the `pgmpy` package. "
            "Install it with: pip install pgmpy"
        ) from exc
    return MarkovNetwork, DiscreteFactor, VariableElimination


class PgmpyBaseInference(BaseInference):
    """Marker base for pgmpy-backed inference engines.

    A pgmpy engine answers a query by **exporting** the model rather than by
    running the algorithm itself: it enumerates each factor into a static table
    for the observation at hand (see
    :func:`~torch_concepts.nn.modules.mid.inference.utils.factor_table`) and
    hands those tables to pgmpy, which is a NumPy library.

    Two consequences follow from that boundary and hold for every engine here:

    - **No gradients.** The tables cross into NumPy, so nothing downstream of
      the export is differentiable. These engines are for evaluating a model
      that has already been trained; train with a pure-PyTorch engine
      (:class:`~torch_concepts.nn.BeliefPropagation` for a factor graph) and
      evaluate with these.
    - **One pgmpy model per observation.** A factor here is parametrized by an
      ``nn.Module``, so its table depends on the input and every element of the
      batch gets its own. The batch is therefore a Python loop, not a tensor
      axis.

    Parameter sharing with the wrapped PGM is inherited from
    :class:`BaseInference`.
    """

    name = "PgmpyBaseInference"
