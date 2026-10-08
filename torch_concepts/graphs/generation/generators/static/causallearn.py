"""PC/GES discovery from the concept columns supplied for precomputation.

Causallearn is imported lazily. Directed endpoints become source-to-target
adjacency entries; ambiguous endpoints keep reciprocal nonzero entries.
The raw result can therefore contain cycles. Use an orientation/cycle
removal refinement before the default require_dag=True validation.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from torch_concepts.concept_graph import ConceptGraph
from ...base.static import GraphGeneratorStatic, GraphGeneratorStaticSpec

if TYPE_CHECKING:
    from torch_concepts.data.base.dataset import ConceptDataset


_CONSTRAINT_BASED = {"pc"}
_SCORE_BASED = {"ges"}


def _import_causallearn(method: str):
    """Lazily import and return the requested CausalLearn algorithm.

    Args:
        method: One of ``'pc'``,``'ges'``.

    Raises:
        ValueError: If ``method`` is not supported.
        ImportError: If ``causallearn`` is not installed.
    """
    try:
        if method == "pc":
            from causallearn.search.ConstraintBased.PC import pc
            return pc
        elif method == "ges":
            from causallearn.search.ScoreBased.GES import ges
            return ges
        else:
            raise ValueError(
                f"Unknown causallearn method '{method}'. "
                f"Supported: {sorted(_CONSTRAINT_BASED | _SCORE_BASED)}."
            )
    except ImportError as exc:
        raise ImportError(
            "CausalLearn-based graph generator requires the `causallearn` package. "
            "Install it with: pip install causal-learn"
        ) from exc


def _cl_graph_to_adj(cl_graph: Any) -> torch.Tensor:
    """Convert CausalLearn endpoint codes to float32 adjacency in node order.

    Endpoint differences of -2/+2 select the forward direction. Other endpoint
    codes are retained, so ambiguous pairs remain reciprocal nonzero entries.
    This conversion neither orients ambiguous edges nor removes cycles.
    """
    adj_np = np.array(cl_graph.graph, dtype=np.float32, copy=True)
    diff = adj_np - adj_np.T
    adj_np[diff == -2] = 1.0
    adj_np[diff == 2] = 0.0
    return torch.from_numpy(adj_np)


def _compute_causallearn(
    self: GraphGeneratorStatic,
    dataset: ConceptDataset,
) -> ConceptGraph:
    """Discover a graph from dataset.concepts in dataset.concept_names order.

    Concept observations are detached and moved to CPU/NumPy. PC uses alpha
    and indep_test; GES uses score_func. Return endpoint-converted adjacency
    without refinement; the common materialization lifecycle handles validation.
    """
    algorithm = _import_causallearn(self.name)
    data = dataset.concepts.detach().cpu().numpy()

    if self.name in _CONSTRAINT_BASED:
        result = algorithm(data, self.alpha, self.indep_test)
        cl_graph = result[0] if isinstance(result, tuple) else result.G
    else:
        cl_graph = algorithm(data, score_func=self.score_func)["G"]

    concept_names = list(dataset.concept_names)
    return ConceptGraph(
        _cl_graph_to_adj(cl_graph),
        node_names=concept_names,
    )


@GraphGeneratorStatic.register_source(
    "Causallearn", names=["pc", "ges"],
)
def _load_causallearn_source(
    generator: GraphGeneratorStatic,
    name: str,
    alpha: float = 0.05,
    indep_test: str = "chisq",
    score_func: str = "local_score_BDeu",
) -> GraphGeneratorStaticSpec:
    """Configure PC or GES and return the static compute callback.

    Parameters
    ----------
    generator : GraphGeneratorStatic
        Generator on which algorithm options are stored.
    name : {"pc", "ges"}
        Discovery method, registered under source="Causallearn".
    alpha : float, default 0.05
        PC significance level, strictly between zero and one. Ignored by GES.
    indep_test : str, default "chisq"
        Conditional independence test forwarded to PC; ignored by GES.
    score_func : str, default "local_score_BDeu"
        Score forwarded to GES; ignored by PC. The defaults target discrete
        observations; choose test/score settings appropriate to your concepts.

    Returns
    -------
    GraphGeneratorStaticSpec
        Dataset-based compute contract. causal-learn is imported at compute time.
    """
    supported = _CONSTRAINT_BASED | _SCORE_BASED
    if name not in supported:
        raise ValueError(
            f"Unknown CausalLearn name {name!r}. "
            f"Supported names: {sorted(supported)}."
        )
    if name in _CONSTRAINT_BASED and not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1.")
    # Normalize unused options so they cannot change the cache identity.
    generator.alpha = alpha if name == "pc" else None
    generator.indep_test = indep_test if name == "pc" else None
    generator.score_func = score_func if name == "ges" else None
    return GraphGeneratorStaticSpec(compute=_compute_causallearn)
