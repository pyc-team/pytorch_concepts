from typing import Optional, Union, Sequence, Tuple, Dict
import torch
from torch_concepts import AnnotatedTensor, Annotations


def _merge_payload(name: str,
                   A: Optional[torch.Tensor], A_mask: torch.Tensor,
                   B: Optional[torch.Tensor], B_mask: torch.Tensor,
                   left_labels: Tuple[str, ...],
                   right_labels: Tuple[str, ...],
                   union_labels: Tuple[str, ...]) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    Merge two payloads along axis=1 with masks.
    Returns (merged_payload, merged_mask).
    """
    device = None
    if A is not None:
        device = A.device
    elif B is not None:
        device = B.device

    # conflict detection
    posL = {l: i for i, l in enumerate(left_labels)}
    posR = {l: i for i, l in enumerate(right_labels)}
    conflicts = [
        lab for lab in set(left_labels) & set(right_labels)
        if A_mask[posL.get(lab, 0)] and B_mask[posR.get(lab, 0)]
    ]
    if conflicts:
        raise ValueError(f"Join conflict on payload '{name}' for labels {conflicts}")

    # new mask for union
    U_mask = torch.zeros(len(union_labels), dtype=torch.bool, device=device)
    for lab in union_labels:
        if (lab in posL and A_mask[posL[lab]]) or (lab in posR and B_mask[posR[lab]]):
            U_mask[union_labels.index(lab)] = True

    # if neither side provides anything, done
    if not U_mask.any():
        return None, U_mask

    # choose template for shape/dtype
    src = A if A is not None else B
    Bsz = src.shape[0]
    if name == "concept_probs":
        out = torch.zeros(Bsz, len(union_labels), dtype=src.dtype, device=src.device)
    else:
        D = src.shape[2]
        out = torch.zeros(Bsz, len(union_labels), D, dtype=src.dtype, device=src.device)

    posU = {l: i for i, l in enumerate(union_labels)}

    # copy left side
    if A is not None:
        idx_union, idx_left = [], []
        for l in left_labels:
            if A_mask[posL[l]]:
                idx_union.append(posU[l])
                idx_left.append(posL[l])
        if idx_union:
            iu = torch.tensor(idx_union, device=src.device)
            il = torch.tensor(idx_left, device=src.device)
            out.index_copy_(1, iu, A.index_select(1, il))

    # copy right side (only where right mask=True)
    if B is not None:
        idx_union, idx_right = [], []
        for l in right_labels:
            if B_mask[posR[l]]:
                idx_union.append(posU[l])
                idx_right.append(posR[l])
        if idx_union:
            iu = torch.tensor(idx_union, device=src.device)
            ir = torch.tensor(idx_right, device=src.device)
            out.index_copy_(1, iu, B.index_select(1, ir))

    return out, U_mask


class ConceptTensor(torch.Tensor):
    """
    Tensor subclass with multiple concept-related payloads
    (embeddings, probabilities, residuals) and their annotations.
    """

    def __new__(
        cls,
        annotations: Annotations,
        concept_probs: Optional[torch.Tensor] = None,
        concept_embs: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None
    ):
        base = None
        if concept_embs is not None:
            base = concept_embs
        elif concept_probs is not None:
            base = concept_probs
        elif residual is not None:
            base = residual

        if base is None:
            obj = torch.Tensor.__new__(cls)
        else:
            obj = torch.Tensor._make_subclass(
                cls, base, require_grad=getattr(base, "requires_grad", False)
            )
        return obj

    def __init__(self,
                 annotations: Annotations,
                 concept_probs: Optional[torch.Tensor] = None,
                 concept_embs: Optional[torch.Tensor] = None,
                 residual: Optional[torch.Tensor] = None):
        super().__init__()
        self.annotations = annotations
        self.concept_embs = concept_embs
        self.concept_probs = concept_probs
        self.residual = residual

        if 1 not in annotations.annotated_axes:
            raise ValueError("Concept axis (1) must be annotated")

        C = len(annotations.get_axis_labels(1))

        def _check(name, t, min_ndim):
            if t is None:
                return
            if t.ndim < min_ndim:
                raise ValueError(f"Payload '{name}' must have at least {min_ndim} dims")
            if t.shape[1] != C:
                raise ValueError(f"Payload '{name}' columns ({t.size(1)}) must equal |annotations| ({C})")

        _check("concept_embs", concept_embs, 3)
        _check("concept_probs", concept_probs, 2)
        _check("residual", residual, 2)

        # automatically create presence masks
        self._mask = {}
        for name, payload in {
            "concept_embs": concept_embs,
            "concept_probs": concept_probs,
            "residual": residual,
        }.items():
            device = None if payload is None else payload.device
            self._mask[name] = torch.ones(C, dtype=torch.bool, device=device) if payload is not None else \
                torch.zeros(C, dtype=torch.bool)

    def mask(self, name: str) -> torch.Tensor:
        """Return boolean presence mask for payload."""
        return self._mask[name]

    # ---------- priority selection ----------
    def _select_tensor(self):
        for name in ("concept_embs", "concept_probs", "residual"):
            t = getattr(self, name, None)
            if t is not None:
                return t, name
        raise RuntimeError("No backing payload (all None).")

    @property
    def tensor(self):
        t, _ = self._select_tensor()
        return t

    # ---------- unwrap helper ----------
    @staticmethod
    def _materialise_if_nested(inner):
        if hasattr(inner, "is_nested") and getattr(inner, "is_nested"):
            if hasattr(inner, "concat_concepts"):
                return inner.concat_concepts()
        return inner

    @staticmethod
    def _unwrap(obj):
        if isinstance(obj, ConceptTensor):
            chosen = obj.tensor
            return ConceptTensor._materialise_if_nested(chosen)
        if isinstance(obj, (tuple, list)):
            return type(obj)(ConceptTensor._unwrap(x) for x in obj)
        if isinstance(obj, dict):
            return {k: ConceptTensor._unwrap(v) for k, v in obj.items()}
        return obj

    # ---------- torch op interception ----------
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*ConceptTensor._unwrap(args), **ConceptTensor._unwrap(kwargs))

    # ---------- convenience ----------
    @property
    def shape(self):
        inner = self.tensor
        if hasattr(inner, "is_nested") and getattr(inner, "is_nested"):
            return inner.concat_concepts().shape
        return inner.shape

    def _apply_to_all(self, method: str, *args, **kwargs):
        def maybe_apply(x):
            if x is None:
                return None
            fn = getattr(x, method, None)
            return fn(*args, **kwargs) if callable(fn) else x

        return ConceptTensor(
            concept_probs=maybe_apply(self.concept_probs),
            concept_embs=maybe_apply(self.concept_embs),
            residual=maybe_apply(self.residual),
        )

    def to(self, all: bool = False, *args, **kwargs):
        if all:
            return self._apply_to_all("to", *args, **kwargs)
        t, name = self._select_tensor()
        moved = getattr(t, "to", lambda *a, **k: t)(*args, **kwargs)
        return ConceptTensor(
            concept_probs=moved if name == "concept_probs" else self.concept_probs,
            concept_embs=moved if name == "concept_embs" else self.concept_embs,
            residual=moved if name == "residual" else self.residual,
        )

    def cpu(self, all=False):
        return self.to(all=all, device="cpu")

    def cuda(self, all=False):
        return self.to(all=all, device="cuda")

    def detach(self, all=False):
        if all:
            return self._apply_to_all("detach")
        t, name = self._select_tensor()
        det = getattr(t, "detach", lambda: t)()
        return ConceptTensor(
            concept_probs=det if name == "concept_probs" else self.concept_probs,
            concept_embs=det if name == "concept_embs" else self.concept_embs,
            residual=det if name == "residual" else self.residual,
        )

    def clone(self, all=False):
        if all:
            return self._apply_to_all("clone")
        t, name = self._select_tensor()
        cl = getattr(t, "clone", lambda: t)()
        return ConceptTensor(
            concept_probs=cl if name == "concept_probs" else self.concept_probs,
            concept_embs=cl if name == "concept_embs" else self.concept_embs,
            residual=cl if name == "residual" else self.residual,
        )

    # ---------- nice printing ----------
    def __repr__(self):
        try:
            active, which = self._select_tensor()
            shape = tuple(active.shape)
        except RuntimeError:
            which, shape = "none", None
        return (
            f"ConceptTensor(default={which}, "
            f"embs={self.concept_embs is not None}, "
            f"probs={self.concept_probs is not None}, "
            f"residual={self.residual is not None}, "
            f"shape={shape})"
        )

    def join(self, other: "ConceptTensor") -> "ConceptTensor":
        new_ann = self.annotations.join_union(other.annotations, axis=1)
        union_labels = new_ann.get_axis_labels(1)
        left_labels = self.annotations.get_axis_labels(1)
        right_labels = other.annotations.get_axis_labels(1)

        new_embs, embs_mask = _merge_payload("concept_embs",
                                             self.concept_embs, self._mask["concept_embs"],
                                             other.concept_embs, other._mask["concept_embs"],
                                             left_labels, right_labels, union_labels)

        new_probs, probs_mask = _merge_payload("concept_probs",
                                               self.concept_probs, self._mask["concept_probs"],
                                               other.concept_probs, other._mask["concept_probs"],
                                               left_labels, right_labels, union_labels)

        new_resid, resid_mask = _merge_payload("residual",
                                               self.residual, self._mask["residual"],
                                               other.residual, other._mask["residual"],
                                               left_labels, right_labels, union_labels)

        out = ConceptTensor(new_ann, new_probs, new_embs, new_resid)
        out._mask = {
            "concept_embs": embs_mask,
            "concept_probs": probs_mask,
            "residual": resid_mask,
        }
        return out

    def extract_by_annotation(self, labels: Sequence[str]) -> "ConceptTensor":
        labels = tuple(labels)
        new_ann = self.annotations.select(axis=1, keep_labels=labels)
        pos = {l: i for i, l in enumerate(self.annotations.get_axis_labels(1))}
        idx = torch.tensor([pos[l] for l in labels],
                           device=next((t.device for t in [self.concept_embs, self.concept_probs, self.residual] if
                                        t is not None), 'cpu'))

        def _slice(T):
            return None if T is None else T.index_select(1, idx)

        def _slice_mask(m):
            return m.index_select(0, idx)

        out = ConceptTensor(
            annotations=new_ann,
            concept_embs=_slice(self.concept_embs),
            concept_probs=_slice(self.concept_probs),
            residual=_slice(self.residual),
        )
        out._mask = {
            "concept_embs": _slice_mask(self._mask["concept_embs"]),
            "concept_probs": _slice_mask(self._mask["concept_probs"]),
            "residual": _slice_mask(self._mask["residual"]),
        }
        return out
