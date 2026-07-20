"""Backend-agnostic scaffolding for inference engines."""

from __future__ import annotations

import inspect
import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..distributions import spec_for
from ..graph.probabilistic_model import ProbabilisticModel
from ..variable import Variable
from .utils import flatten_event, leading_shape
from ...outputs import InferenceOutput, ParamDict
from .....annotations import Annotations
from .....tensor import AnnotatedTensor


class BaseInference(nn.Module):
    """Abstract base class for all inference engines.

    The engine *wraps* a :class:`ProbabilisticModel` by holding a reference
    to it (``self.pgm = pgm``). 

    Backend-specific subclasses (e.g., ``PyroBaseInference`` or ``TorchBaseInference``)
    layer engine-specific machinery on top.

    ``query`` and ``evidence`` are attribute-style containers keyed by PGM
    variable name. ``__call__`` delegates to :meth:`query`.

    Parameters
    ----------
    pgm : ProbabilisticModel
        The model to run inference on. Held **by reference** — because
        ``nn.Module.__setattr__`` registers it as a submodule, the engine shares
        parameters with the model rather than copying them, so one optimizer
        over ``pgm.parameters()`` trains the graph however many engines wrap it.

    Warns
    -----
    UserWarning
        If any root factor's parametrization requires input arguments; such a
        root must be supplied as constant evidence on every ``query`` call.
    """
    
    name: str = "BaseInference"

    def __init__(self, pgm: ProbabilisticModel):
        super().__init__()
        # NOTE: nn.Module.__setattr__ auto-registers ``pgm`` as a submodule, so
        # the engine shares parameters with the original PGM (no copy).
        self.pgm = pgm

        # (labels, widths) -> Annotations, reused across queries; see _annotate.
        self._annotation_cache: Dict[tuple, Annotations] = {}

        # Factor-based (not variable-keyed): a ProbabilisticModel keys factors by
        # factor name, and undirected potentials have no ``is_root``. Only root CPDs
        # whose parametrization needs inputs must receive constant evidence every call.
        roots_needing_input: List[str] = [
            f.name
            for f in pgm.factors.values()
            if getattr(f, "is_root", False)
            and any(
                len(inspect.signature(mod.forward).parameters) > 0
                for mod in f.parametrization.values()
            )
        ]
        if roots_needing_input:
            warnings.warn(
                f"{self.name}: the following root variables have a parametrization "
                f"that requires input arguments: {roots_needing_input}. "
                "These must be supplied as constant evidence on every query call.",
                UserWarning,
                stacklevel=2,
            )

    def _require_directed(self) -> None:
        """Guard for engines that need a topological order.

        The directed engines traverse ``pgm.levels`` / ``pgm.sorted_variables``,
        which only a :class:`BayesianNetwork` provides. Passing a general
        (undirected or mixed) :class:`ProbabilisticModel` raises a clear error
        pointing at :class:`BeliefPropagation`.
        """
        from ..graph.bayesian_network import BayesianNetwork

        if not isinstance(self.pgm, BayesianNetwork):
            raise TypeError(
                f"{self.name} requires a directed BayesianNetwork (an acyclic, "
                "all-CPD model with a topological order); got a general "
                f"{type(self.pgm).__name__}. Use BeliefPropagation for undirected "
                "or mixed (chain) graphs."
            )

    def _reject_plate_and_members(self, names, container: str) -> None:
        """Forbid observing a plate *and* any of its members at the same time.

        A plate may be observed either as a whole (its own name) or through a
        subset of its members — never both at once, because the two would
        contradict each other on the shared columns and the member would
        silently win. Several members together are fine; that is the "subset"
        case.
        """
        resolve = self.pgm.resolve
        names = set(names)
        for name in sorted(names):
            owner = resolve(name).name
            if name != owner and owner in names:
                raise ValueError(
                    f"{self.name}: {container} names both the plate {owner!r} and "
                    f"its member {name!r}. Observe either the whole plate or a "
                    "subset of its members, not both."
                )

    # ------------------------------------------------------------------
    # Leading (batch-like) dimensions
    # ------------------------------------------------------------------
    # Every engine accepts tensors shaped ``(*leading, *event)`` for any number
    # of leading dimensions — the last axis is the only operating one. These
    # helpers are the single place that split the two, so no engine hard-codes
    # ``shape[0]`` as "the batch".

    def _event_of(self, name: str) -> Tuple[Tuple[int, ...], int]:
        """The ``(event shape, flat size)`` a tensor for ``name`` carries.

        A plate *member* carries only its own block, not the whole plate's, so
        this is name-level rather than variable-level.
        """
        var = self.pgm.resolve(name)
        if name == var.name:
            return tuple(var.shape), var.size
        return (var.member_size,), var.member_size

    def _leading_shape(self, name: str, value: torch.Tensor) -> torch.Size:
        """Leading dimensions of a tensor supplied for ``name``."""
        event, size = self._event_of(name)
        return leading_shape(event, size, value, f"{self.name}: {name!r}")

    def _query_leading_shape(
        self,
        query: Dict[str, Optional[torch.Tensor]],
        evidence: Dict[str, torch.Tensor],
    ) -> torch.Size:
        """The leading shape shared by every supplied tensor.

        Falls back to ``(1,)`` when a query names only variables to compute and
        provides no tensor to read a batch shape from — the same convention the
        single-batch-dimension implementation used.
        """
        for name, value in evidence.items():
            return self._leading_shape(name, value)
        for name, value in query.items():
            if value is not None:
                return self._leading_shape(name, value)
        return torch.Size([1])

    @staticmethod
    def _collapse_leading(
        container: Dict[str, Optional[torch.Tensor]], leading: torch.Size
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Fold every tensor's leading dimensions into one batch axis.

        The estimator engines (rejection / importance sampling, and the Pyro
        backend, whose plates are tied to a single batch dimension) run their
        whole algorithm on one batch axis. Rather than thread ``n`` leading dims
        through those, they collapse on the way in and
        :meth:`_restore_leading` reshapes the estimate on the way out. The
        reshape is a view for the contiguous inputs these engines receive.
        """
        n_leading = len(leading)
        batch = math.prod(leading) if n_leading else 1
        return {
            name: (
                None if value is None
                else value.reshape(batch, *value.shape[n_leading:])
            )
            for name, value in container.items()
        }

    @staticmethod
    def _restore_leading(
        value: Optional[torch.Tensor], leading: torch.Size
    ) -> Optional[torch.Tensor]:
        """Undo :meth:`_collapse_leading` on a per-observation result."""
        if value is None:
            return None
        return value.reshape(*leading, *value.shape[1:])

    def _restore_output_leading(
        self, out: InferenceOutput, leading: torch.Size
    ) -> InferenceOutput:
        """Undo :meth:`_collapse_leading` across a whole assembled output.

        Reshaping an :class:`~torch_concepts.tensor.AnnotatedTensor` preserves
        its annotation (the annotated last axis is untouched), so the labels
        survive the round trip. A no-op for the ordinary single-batch case.
        """
        if len(leading) == 1:
            return out
        restore = self._restore_leading
        return type(out)(
            params={q: restore(t, leading) for q, t in out.params.items()},
            guide_params={q: restore(t, leading) for q, t in out.guide_params.items()},
            samples=restore(out.samples, leading),
            probabilities=restore(out.probabilities, leading),
        )

    def _validate_containers(
        self,
        query: Dict[str, Optional[torch.Tensor]],
        evidence: Dict[str, torch.Tensor],
    ) -> None:
        """Check that:
         - query and evidence keys are valid variable names,
         - evidence does not name both a plate and one of its members,
         - all values are tensors, and
         - leading (batch-like) shapes match.
        """

        all_names = getattr(self.pgm, "queryable_names", None)
        if all_names is None:
            all_names = {v.name for v in self.pgm.variables.values()}
        unknown_q = set(query.keys()) - all_names
        if unknown_q:
            raise ValueError(f"{self.name}: unknown query names {sorted(unknown_q)}.")
        unknown_e = set(evidence.keys()) - all_names
        if unknown_e:
            raise ValueError(f"{self.name}: unknown evidence names {sorted(unknown_e)}.")

        # Only after the names are known to be valid, so ``resolve`` cannot KeyError.
        self._reject_plate_and_members(evidence, "evidence")

        for name, val in evidence.items():
            if not isinstance(val, torch.Tensor):
                raise ValueError(
                    f"{self.name}: evidence[{name!r}] must be a Tensor, "
                    f"got {type(val).__name__}."
                )

        if not query and not evidence:
            raise ValueError(
                f"{self.name}: nothing to do — both `query` and `evidence` are "
                "empty. Provide at least one variable to compute (query) or to "
                "condition on (evidence)."
            )

        all_tensors = {name: val for name, val in query.items() if val is not None}
        all_tensors.update(evidence)
        # A tensor may carry any number of leading dimensions, so agreement is
        # on the whole leading *shape*, not just on a single batch size.
        leadings = {
            name: tuple(self._leading_shape(name, t))
            for name, t in all_tensors.items()
        }
        if len(set(leadings.values())) > 1:
            shapes = {name: tuple(t.shape) for name, t in all_tensors.items()}
            raise ValueError(
                f"{self.name}: mismatched leading (batch) dimensions {leadings} "
                f"for tensors of shape {shapes}."
            )

    @staticmethod
    def _normalize_query(
        query: Union[List[str], Dict[str, Optional[torch.Tensor]]],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Normalize query input to a dict mapping variable names to optional tensors."""
        if isinstance(query, list):
            return {name: None for name in query}
        return query

    # Uniform plate routing — the only plate awareness outside the models. Each
    # call is an identity/no-op for ordinary variables, so engines run the same
    # line for plate and non-plate names (never behind an is_plate branch).
    def _split_evidence(
        self, evidence: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
        """Split evidence into whole-variable entries and per-owner member entries.

        Returns ``(whole, member_evidence)`` where ``whole`` keeps every key that
        names a variable, and ``member_evidence[owner_name][member_name]`` collects
        keys that name plate members. O(#evidence), never O(#members).
        """
        whole: Dict[str, torch.Tensor] = {}
        member_evidence: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, value in evidence.items():
            var = self.pgm.resolve(name)
            if name == var.name:
                whole[name] = value
            else:
                member_evidence.setdefault(var.name, {})[name] = value
        return whole, member_evidence

    # ------------------------------------------------------------------
    # Output assembly (backend-agnostic)
    # ------------------------------------------------------------------
    # Engines compute per-variable parameter dicts internally, then hand them to
    # these two methods, which produce the quantity-keyed annotated tensors an
    # :class:`InferenceOutput` exposes. Assembly happens exactly once and the
    # concatenated tensor is the only storage: a per-variable or per-member view
    # is recovered by label slicing, which is a view rather than a copy.

    def _output_labels(self, query_names) -> List[Tuple[List[str], Variable]]:
        """``(labels, owning variable)`` chunks for the queried names, in order.

        A queried plate expands into one label per member, all in one chunk.
        That keeps the labels a strict *partition* of the assembled tensor's
        columns — asking for a plate and one of its members in the same query
        cannot emit those columns twice — and it gives every label an
        unambiguous concept type, since a plate of binary members is a run of
        binary columns rather than one wide categorical one.

        Nothing is lost by expanding: each label records its owning variable in
        the annotation's metadata, so the plate name still addresses the whole
        block (``out.probs['g']`` returns all of ``g``'s member columns as a
        view). Duplicates collapse to their first occurrence.

        The chunking is what lets assembly stay O(1) in member count for the
        common case: a chunk equal to ``variable.members`` (every whole-variable
        query — a non-plate, or a plate queried by its own name with no member
        already emitted) marks the factor's stacked output as usable as-is,
        with no per-member slicing and re-concatenation.
        """
        resolve = self.pgm.resolve
        seen: set = set()
        chunks: List[Tuple[List[str], Variable]] = []
        for name in query_names:
            var = resolve(name)
            labels = [
                label
                for label in (var.members if name == var.name else [name])
                if label not in seen
            ]
            if labels:
                seen.update(labels)
                chunks.append((labels, var))
        return chunks

    @staticmethod
    def _label_type(variable: Variable, width: int) -> str:
        """Concept type of one label, from its family and its column width.

        A continuous family is continuous whatever its width. A discrete one is
        ``'binary'`` when it occupies a single column and ``'categorical'``
        otherwise — the same convention the high level's concept annotations
        already use for these variables.
        """
        if not spec_for(variable.distribution).is_discrete:
            return "continuous"
        return "binary" if width == 1 else "categorical"

    def _annotate(
        self,
        pieces: List[torch.Tensor],
        labels: List[str],
        widths: List[int],
    ) -> AnnotatedTensor:
        """Concatenate ``pieces`` on the last axis under a label annotation.

        ``pieces`` need not be parallel to ``labels``: one piece may carry many
        labels (a whole plate's stacked output), as long as the label widths sum
        to the concatenated width.

        The ``Annotations`` object is cached on the engine, keyed by the
        ``(labels, widths)`` signature: for a fixed query only the tensor
        *values* change batch to batch, while building the annotation is
        O(total labels) Python — the dominant per-query cost for large plates.
        The cached instance is shared by every output with the same signature
        (its structural fields are write-once, so this is safe).
        """
        key = (tuple(labels), tuple(widths))
        annotation = self._annotation_cache.get(key)
        if annotation is None:
            resolve = self.pgm.resolve
            owners = [resolve(label) for label in labels]
            annotation = Annotations(
                labels=list(labels),
                cardinalities=list(widths),
                types=[
                    self._label_type(var, w) for var, w in zip(owners, widths)
                ],
                metadata={
                    label: {
                        "variable": var.name,
                        "variable_type": var.variable_type,
                        "distribution": var.distribution.__name__,
                    }
                    for label, var in zip(labels, owners)
                },
            )
            self._annotation_cache[key] = annotation
        data = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)
        return AnnotatedTensor(data, annotation, axis=-1)

    def _assemble_params(
        self,
        per_variable: Dict[str, ParamDict],
        query_names,
    ) -> Dict[str, AnnotatedTensor]:
        """Turn ``{variable: {quantity: tensor}}`` into ``{quantity: AnnotatedTensor}``.

        Only the queried names appear, in query order. A variable contributes to
        exactly the quantities its family produces, so a model mixing Bernoulli
        and Normal variables yields a ``'probs'``/``'logits'`` tensor over the
        former and a ``'loc'``/``'scale'`` tensor over the latter.

        An empty ``query_names`` means "everything computed" — the engines that
        report every traced site (the Pyro backend, and the guide side of
        variational inference) have no query to filter by.
        """
        query_names = list(query_names) or list(per_variable)
        pieces: Dict[str, List[torch.Tensor]] = {}
        labels: Dict[str, List[str]] = {}
        widths: Dict[str, List[int]] = {}

        def emit(quantity: str, tensor: torch.Tensor, chunk: List[str]) -> None:
            if tensor.dim() == 0:
                raise ValueError(
                    f"{self.name}: parameter {quantity!r} of {chunk[0]!r} is a "
                    "scalar, but every reported parameter must be shaped "
                    "(*leading, width) so it can live on the annotated axis. "
                    "Scalar knobs (a relaxation temperature, say) belong on "
                    "the engine, not in its output."
                )
            pieces.setdefault(quantity, []).append(tensor)
            labels.setdefault(quantity, []).extend(chunk)
            # A one-piece chunk keeps its full width; a whole plate's stacked
            # tensor splits evenly over its members (plate families are
            # one-scalar-per-element, so the division is exact).
            widths.setdefault(quantity, []).extend(
                [int(tensor.shape[-1]) // len(chunk)] * len(chunk)
            )

        for chunk, var in self._output_labels(query_names):
            params = per_variable.get(var.name)
            if params is None:
                continue  # fully observed, or not computed by this engine
            if chunk == var.members:
                # Whole variable (a non-plate, or a plate queried by name):
                # the factor's stacked output is used as-is — no per-member
                # slicing and re-concatenation.
                for quantity, tensor in params.items():
                    emit(quantity, tensor, chunk)
            else:
                for label in chunk:
                    for quantity, tensor in var.select(params, label).items():
                        emit(quantity, tensor, [label])
        return {
            quantity: self._annotate(tensors, labels[quantity], widths[quantity])
            for quantity, tensors in pieces.items()
        }

    def _assemble_samples(
        self,
        per_variable: Dict[str, torch.Tensor],
        query_names,
    ) -> Optional[AnnotatedTensor]:
        """Same as :meth:`_assemble_params` for realisations.

        Values are flattened from ``(*leading, *event)`` to ``(*leading, size)``
        so they share the annotated-axis layout of the parameters. Returns
        ``None`` when the engine drew no samples for any queried name.
        """
        pieces: List[torch.Tensor] = []
        labels: List[str] = []
        widths: List[int] = []
        for chunk, var in self._output_labels(query_names):
            value = per_variable.get(var.name)
            if value is None:
                continue
            flat = flatten_event(var, value)
            if chunk == var.members:
                # Whole variable: the stacked value is used as-is (see
                # :meth:`_assemble_params` for the width bookkeeping).
                pieces.append(flat)
                labels.extend(chunk)
                widths.extend([int(flat.shape[-1]) // len(chunk)] * len(chunk))
            else:
                for label in chunk:
                    piece = var.select_value(flat, label)
                    pieces.append(piece)
                    labels.append(label)
                    widths.append(int(piece.shape[-1]))
        return self._annotate(pieces, labels, widths) if pieces else None

    def __call__(
        self,
        query: Union[List[str], Dict[str, Optional[torch.Tensor]]],
        evidence: Dict[str, torch.Tensor],
    ) -> InferenceOutput:
        return self.query(query=query, evidence=evidence)

    def _format_repr(self, **fields) -> str:
        """Render ``EngineName(param=value, ...)`` for the given inference
        parameters.

        The wrapped :class:`ProbabilisticModel` is intentionally excluded — only
        the engine's own configuration is shown. ``nn.Module`` values are
        rendered by their class name and (non-string) callables by their
        ``__name__`` so the PGM is never recursively printed.
        """
        items = []
        for key, val in fields.items():
            if isinstance(val, nn.Module):
                rendered = type(val).__name__
            elif callable(val) and not isinstance(val, str):
                rendered = getattr(val, "__name__", repr(val))
            else:
                rendered = repr(val)
            items.append(f"{key}={rendered}")
        return f"{type(self).__name__}({', '.join(items)})"

    def __repr__(self) -> str:
        # Concrete engines override this to surface their own parameters; the
        # base fallback shows just the engine name (no parameters, no PGM).
        return self._format_repr()
