"""Backend-agnostic abstract class for inference engines."""

from __future__ import annotations

import inspect
import math
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..distributions import spec_for
from ..graph.probabilistic_model import ProbabilisticModel
from ..variable import Variable
from .utils import flatten_event, leading_shape, make_temperature_schedule
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
    initial_temperature, annealing, annealing_rate, final_temperature
        Schedule for the relaxed-discrete distributions' temperature; see
        :func:`~torch_concepts.nn.modules.mid.inference.utils.make_temperature_schedule`.
        Lives here rather than per-engine: annealing is a property of the run,
        not of a backend, and a training loop should be able to advance it
        without knowing which engine it holds (see :meth:`temperature_step`).
        Engines that never sample keep it and never read it.

    Warns
    -----
    UserWarning
        If any root factor's parametrization requires input arguments; such a
        root must be supplied as constant evidence on every ``query`` call.
    """
    
    name: str = "BaseInference"

    def __init__(
        self,
        pgm: ProbabilisticModel,
        initial_temperature: float = 1.0,
        annealing: Union[str, Callable[[int], float]] = "constant",
        annealing_rate: float = 0.0,
        final_temperature: float = 1e-6,
    ):
        super().__init__()
        # NOTE: nn.Module.__setattr__ auto-registers ``pgm`` as a submodule, so
        # the engine shares parameters with the original PGM (no copy).
        self.pgm = pgm

        # Retained for repr/introspection; the live schedule lives in ``_schedule``.
        self.initial_temperature = float(initial_temperature)
        self.annealing = annealing
        self.annealing_rate = float(annealing_rate)
        self.final_temperature = float(final_temperature)
        self._schedule = make_temperature_schedule(
            initial_temperature, annealing, annealing_rate, final_temperature
        )
        self._step: int = 0
        # A buffer, so it follows `.to(device)` and — being persistent — travels
        # in `state_dict`: evaluation after a reload must decode at the
        # temperature training reached, not at the initial one.
        self.register_buffer(
            "_temperature", torch.tensor(float(self._schedule(self._step)))
        )

        # Both keyed by a query signature and reused across queries: the label
        # chunks a query expands to (see _output_labels) and the Annotations
        # built from them (see _annotate). Only the tensor *values* change from
        # one query to the next, so all of this Python work is done once.
        self._label_cache: Dict[tuple, List[Tuple[List[str], Variable]]] = {}
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

    # ------------------------------------------------------------------
    # Relaxation temperature
    # ------------------------------------------------------------------
    # Shared by every engine, because every engine that realises a discrete
    # variable relaxes it, and the schedule is a property of the *run* rather
    # than of any one backend. Engines that never sample simply never read it.

    @property
    def temperature(self) -> torch.Tensor:
        """Current relaxation temperature of the discrete distributions."""
        return self._temperature

    def temperature_step(self) -> None:
        """Advance the schedule by one training step.

        A no-op outside training, so evaluation reads the temperature training
        reached rather than annealing past it — a model tested at a relaxation
        it never trained at samples codes its decoder has never seen.

        Updates the buffer **in place**, so callers must not run this between a
        forward pass and its ``backward``: the temperature is part of the graph
        autograd is about to walk. Once per optimiser step, after backward, is
        the contract (see
        :meth:`~torch_concepts.nn.modules.high.base.learner.BaseLearner.on_train_batch_end`).
        """
        if not self.training:
            return
        self._step += 1
        self._temperature.fill_(float(self._schedule(self._step)))

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
        """It returns the shape and size of a variable given its name.
        If member of a plate then returns the member shape and size; 
        otherwise returns the variable shape and size.
        """
        var = self.pgm.resolve(name)
        if name == var.name:
            return tuple(var.shape), var.size
        return (var.member_size,), var.member_size

    def _leading_shape(self, name: str, value: torch.Tensor) -> torch.Size:
        """Given value's shape, it identifies the 'operating' (event) dimensions
        of the variable and return the leading dimensions
        """
        event, size = self._event_of(name)
        return leading_shape(event, size, value, f"{self.name}: {name!r}")

    def _query_leading_shape(
        self,
        query: Dict[str, Optional[torch.Tensor]],
        evidence: Dict[str, torch.Tensor],
        default: Optional[int] = None,
    ) -> torch.Size:
        """It determines the single leading (batch-like) shape that applies to an entire query call,
        by picking one representative tensor.

        If there's any evidence, just take the first (name, value) pair from the dict
        and return its leading shape immediately. If there was no evidence,
        fall through to the query dict and look for the first entry with a non-None tensor value.
        If neither evidence nor query contains any actual tensor, fall back to
        ``default`` observations (``torch.Size([default])``), or to
        ``torch.Size([1])`` when no default is given.

        The ``default`` is how an unconditional draw asks for a batch: with no
        tensor anywhere there is nothing to read a batch size from, so a pure
        ancestral sample from the priors would otherwise always be a single
        observation.
        """
        for name, value in evidence.items():
            return self._leading_shape(name, value)
        for name, value in query.items():
            if value is not None:
                return self._leading_shape(name, value)
        return torch.Size([1 if default is None else int(default)])

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
        # TODO: should we move to the standard check used by torch? 
        # ``torch.broadcast_shapes(*leadings.values())``
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
        """Normalize query input to a dict mapping variable names to optional tensors.

        This operation is always performed since we can accept also list of varibles.
        """
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
        """It returns ``(labels, owning variable)`` chunks for the queried names,
        in order.

        Plate memebers expand to (member labels, plate variable) and
        non-plate variables expand to (variable labels, variable).

        A pure function of the query names and the (fixed) model structure, so
        the chunks are cached per query signature — expanding a large plate is
        O(members) Python that must not be repeated on every query. Callers
        only read the chunks (never mutate them), so sharing them is safe.
        """
        key = tuple(query_names)

        # If the chunk has already been computed, return it from the cache.
        chunks = self._label_cache.get(key)
        if chunks is not None:
            return chunks
        
        # If the chunk has not been computed, compute it and store it in the cache.
        resolve = self.pgm.resolve
        seen: set = set()
        chunks = []
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
        self._label_cache[key] = chunks
        return chunks

    @staticmethod
    def _label_type(variable: Variable, width: int) -> str:
        """Concept type of one label, inferred from its owning variable and the label's width."""
        if not spec_for(variable.distribution).is_discrete:
            return "continuous"
        return "binary" if width == 1 else "categorical"

    def _annotate(
        self,
        pieces: List[torch.Tensor],
        labels: List[str],
        widths: List[int],
        key: tuple,
    ) -> AnnotatedTensor:
        """Concatenate ``pieces`` on the last axis under a label annotation.

        ``pieces`` need not be parallel to ``labels``: one piece may carry many
        labels (a whole plate's stacked output), as long as the label widths sum
        to the concatenated width.

        The ``Annotations`` object is cached on the engine: for a fixed query
        only the tensor *values* change batch to batch, while building the
        annotation is O(total labels). 
        The cached instance is shared by every output with the
        same signature (its structural fields are write-once, so this is safe).

        ``key`` is that signature, built by the caller with one record per
        emitted chunk so it stays O(#queried names): deriving it from ``labels``
        here would put an O(total labels) tuple build and hash back on every
        query, cache hit included.
        """
        annotation = self._annotation_cache.get(key)
        if annotation is None:
            resolve = self.pgm.resolve
            owners = [resolve(label) for label in labels]
            # A variable spanning several labels (a plate) is registered as a
            # group so the whole plate is addressable by the variable name.
            grouped: Dict[str, List[str]] = {}
            for label, var in zip(labels, owners):
                grouped.setdefault(var.name, []).append(label)
            label_set = set(labels)
            groups = {o: m for o, m in grouped.items() if o not in label_set}
            annotation = Annotations(
                labels=list(labels),
                cardinalities=list(widths),
                types=[
                    self._label_type(var, w) for var, w in zip(owners, widths)
                ],
                groups=groups or None,
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
        query_key = tuple(query_names)
        pieces: Dict[str, List[torch.Tensor]] = {}
        labels: Dict[str, List[str]] = {}
        widths: Dict[str, List[int]] = {}
        sigs: Dict[str, List[tuple]] = {}

        def emit(quantity: str, tensor: torch.Tensor, chunk: List[str]) -> None:
            if tensor.dim() == 0:
                raise ValueError(
                    f"{self.name}: parameter {quantity!r} of {chunk[0]!r} is a "
                    "scalar, but every reported parameter must be shaped "
                    "(*leading, width) so it can live on the annotated axis. "
                    "Scalar knobs (a relaxation temperature, say) belong on "
                    "the engine, not in its output."
                )
            # A one-piece chunk keeps its full width; a whole plate's stacked
            # tensor splits evenly over its members (plate families are
            # one-scalar-per-element, so the division is exact).
            width = int(tensor.shape[-1]) // len(chunk)
            pieces.setdefault(quantity, []).append(tensor)
            labels.setdefault(quantity, []).extend(chunk)
            widths.setdefault(quantity, []).extend([width] * len(chunk))
            # One record per chunk, not per label: with ``query_key`` fixing the
            # chunks (see _output_labels), this pins down which of them actually
            # contributed and at what width — the whole annotation signature.
            sigs.setdefault(quantity, []).append((chunk[0], len(chunk), width))

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
            elif len(chunk) == 1:
                # A single member: a plain column slice, cheaper than a
                # one-column fancy-index gather.
                for quantity, tensor in var.select(params, chunk[0]).items():
                    emit(quantity, tensor, chunk)
            else:
                # A member subset: gather every column in one indexing op
                # rather than slicing and re-concatenating one member at a time.
                idx = var.get_slice(chunk)
                for quantity, tensor in params.items():
                    emit(quantity, tensor[..., idx], chunk)
        return {
            quantity: self._annotate(
                tensors, labels[quantity], widths[quantity],
                (query_key, quantity, tuple(sigs[quantity])),
            )
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
        sig: List[tuple] = []  # see :meth:`_assemble_params` for the signature
        for chunk, var in self._output_labels(query_names):
            value = per_variable.get(var.name)
            if value is None:
                continue
            flat = flatten_event(var, value)
            if chunk == var.members:
                # Whole variable: the stacked value is used as-is.
                piece = flat
            elif len(chunk) == 1:
                # A single member: a plain column slice.
                piece = var.select_value(flat, chunk[0])
            else:
                # A member subset: one gather over all its columns.
                piece = flat[..., var.get_slice(chunk)]
            # One piece per chunk; members share member_size so the per-label
            # width is uniform (see :meth:`_assemble_params`).
            width = int(piece.shape[-1]) // len(chunk)
            pieces.append(piece)
            labels.extend(chunk)
            widths.extend([width] * len(chunk))
            sig.append((chunk[0], len(chunk), width))
        if not pieces:
            return None
        return self._annotate(
            pieces, labels, widths, (tuple(query_names), None, tuple(sig))
        )

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
