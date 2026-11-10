"""Drawing of a probabilistic graphical model to a figure file."""

from __future__ import annotations

import math
from itertools import combinations
from statistics import mean
from typing import Dict, Tuple

import networkx as nx
import torch.nn as nn

from ..factors.cpd import ParametricCPD
from ..factors.factor import ParametricFactor
from ..variable import ConceptVariable, EmbeddingVariable
from .probabilistic_model import ProbabilisticModel

# Node fills: 40% tints of a yellow and a blue (black names on them read at >= 12:1).
_FILLS = {ConceptVariable: "#f8d999", EmbeddingVariable: "#aac9ef"}
# One color per layer class, for its arcs and its name alike: hues darkened to >= 4.5:1 on
# white and pairwise distinct (OKLab delta E >= 17); the classes past the fourth share the gray.
_LAYER_COLORS = ["#c35502", "#db2783", "#008300", "#6250d6"]
_OTHER_LAYER_COLOR = "#52514e"
_NAME_SIZE, _LABEL_SIZE = 8, 7  # font sizes, in points
_GAP = 0.1  # inches between a node and its layer name
_SPACE = 0.3  # inches free between a layer name and the next node in a row
_ROW_GAP = 0.45  # inches free between two rows, for the arrowheads


def _module_name(module: nn.Module) -> str:
    """Class name of ``module``; a ``Sequential`` also lists the modules it chains."""
    if isinstance(module, nn.Sequential):
        return f"Sequential({', '.join(map(_module_name, module))})"
    return type(module).__name__


def _layer_name(factor: ParametricFactor) -> str:
    """Name of the layer parametrizing ``factor`` (see :func:`_module_name`), or one
    ``key: name`` line per module when it has a trunk or several parameters."""
    modules = dict(factor.parametrization)
    if factor.trunk is not None:
        modules = {"trunk": factor.trunk, **modules}
    names = {key: _module_name(module) for key, module in modules.items()}
    return next(iter(names.values())) if len(names) == 1 else "\n".join(f"{k}: {n}" for k, n in names.items())


def _bend(pos: Dict[str, Tuple[float, float]], radius: Dict[str, float], u: str, v: str) -> float:
    """``rad`` of an ``arc3`` edge from ``u`` to ``v`` that clears the nodes the straight edge
    would cross, passing them on its bending side (the left of a downward edge); 0 if none."""
    (xu, yu), (xv, yv) = pos[u], pos[v]
    dx, dy = xv - xu, yv - yu
    length = math.hypot(dx, dy)
    rad = 0.0
    for node, (x, y) in pos.items():
        t = ((x - xu) * dx + (y - yu) * dy) / length ** 2  # where along the edge the node lies
        side = ((x - xu) * dy - (y - yu) * dx) / length  # how far from it, positive on the bending side
        if node not in (u, v) and 0 < t < 1 and abs(side) < radius[node] + _GAP:
            # an arc3 departs from the straight edge by 2 t (1 - t) rad length at t
            rad = max(rad, (side + radius[node] + _SPACE / 2) / (2 * t * (1 - t) * length))
    return min(rad, 1.0)


def _text_size(text: str, size: float) -> Tuple[float, float]:
    """Width and height of ``text`` in inches, at font ``size`` (points)."""
    from matplotlib.textpath import TextPath

    lines = text.split("\n")
    width = max(TextPath((0, 0), line, size=size).get_extents().width for line in lines)
    return width / 72, 1.2 * size * len(lines) / 72


def plot_pgm(model: nn.Module, path: str = "pgm.pdf", layer_names: bool = True) -> None:
    """Save a drawing of the probabilistic graphical model of ``model`` to ``path``.

    Concept variables are drawn in yellow, embedding variables in blue. A CPD
    draws an arrow from each parent to its child; a potential links every pair of
    variables in its scope with an undirected edge. A plate is drawn as a single
    node, also when a factor uses only some of its members. The edges of a factor
    take the color of its layer class, shared by all the factors with a layer of
    that class.

    Directed models are laid out top to bottom by topological depth, with the
    edges that would cross a node curved around it, and each layer name right of
    the child of its CPD, level with its top, where its edges arrive: the names
    never overlap each other or the nodes. The others are laid out with a spring
    layout, each name at the center of its factor's variables.

    Requires ``matplotlib``.

    Parameters
    ----------
    model : nn.Module
        Anything holding its PGM in a ``pgm`` attribute (a high-level model, an
        inference engine), or a :class:`ProbabilisticModel` itself.
    path : str, optional
        Where to save the figure; the format follows the extension (``.pdf``,
        ``.png``, ``.svg``, ...). Default ``"pgm.pdf"``.
    layer_names : bool, optional
        Write once per factor, in the color of its edges, the class name of the
        layer parametrizing them (for a CPD, all the edges into its child); a
        ``Sequential`` lists the modules it chains. Default ``True``.

    Raises
    ------
    TypeError
        If ``model`` neither is nor holds a :class:`ProbabilisticModel`.

    Examples
    --------
    >>> from torch_concepts import Annotations
    >>> from torch_concepts.nn import ConceptBottleneckModel, plot_pgm
    >>> annotations = Annotations(labels=["c1", "c2", "task"], cardinalities=[1, 1, 1])
    >>> model = ConceptBottleneckModel(input_size=4, annotations=annotations, task_names=["task"])
    >>> plot_pgm(model, "cbm.pdf")  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt  # optional dependency: imported only when plotting

    pgm = getattr(model, "pgm", model)
    if not isinstance(pgm, ProbabilisticModel):
        raise TypeError(f"{type(model).__name__} has no probabilistic model to plot.")
    fig = _draw(pgm, layer_names)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _draw(pgm: ProbabilisticModel, layer_names: bool = True):
    """The figure of :func:`plot_pgm`, measured in inches: one data unit is one inch."""
    import matplotlib.pyplot as plt

    # factors with edges, as (layer name, child or None for a potential, edges); a root CPD
    # or a single-variable potential has no edge to draw
    factors = []
    for factor in pgm.factors.values():
        names = list(dict.fromkeys(v.plate.name for v in factor.scope))  # a plate member is drawn as its plate
        if isinstance(factor, ParametricCPD):  # scope = [child, *parents]
            child, edges = names[0], [(parent, names[0]) for parent in names[1:]]
        else:
            child, edges = None, list(combinations(names, 2))
        if edges:
            factors.append((_layer_name(factor), child, edges))
    graph = nx.DiGraph()
    graph.add_nodes_from(pgm.variables)
    graph.add_edges_from(edge for *_, edges in factors for edge in edges)

    colors = {}
    for name, *_ in factors:
        colors.setdefault(name, _LAYER_COLORS[len(colors)] if len(colors) < len(_LAYER_COLORS) else _OTHER_LAYER_COLOR)

    # a circle wide enough for its name; a layer name beside the top of the child of its CPD, so
    # it widens the row of the child, and rows tall enough for a node and half a layer name above
    radius = {node: max(0.25, _text_size(node, _NAME_SIZE)[0] / 2 + 0.08) for node in graph}
    label_sizes = [_text_size(name, _LABEL_SIZE) for name, *_ in factors] if layer_names else []
    beside = {child: _GAP + w for (_, child, _), (w, _) in zip(factors, label_sizes) if child}
    r = max(radius.values())
    dy = _ROW_GAP + 2 * r + max((h for _, h in label_sizes), default=0) / 2

    layered = pgm.is_directed
    if layered:
        # rows by topological depth, a root right above its first child rather than in the first
        # row (a shorter edge); each row ordered by the mean position of the parents (fewer
        # crossings), roots last, packed from the left with each layer name before the next node,
        # and centered on its nodes. A layer name is level with the top of its child, by the
        # arrowheads of its edges
        depth = {node: d for d, nodes in enumerate(nx.topological_generations(graph)) for node in nodes}
        for node in graph:
            if graph.in_degree(node) == 0 and graph.out_degree(node):
                depth[node] = min(depth[child] for child in graph.successors(node)) - 1
        pos = {}
        for d in sorted(set(depth.values())):
            nodes = sorted((node for node in graph if depth[node] == d),
                           key=lambda node: mean(pos[p][0] for p in graph.predecessors(node))
                           if graph.in_degree(node) else math.inf)
            xs, end = [], 0.0
            for node in nodes:
                xs.append(end + radius[node])
                end = xs[-1] + radius[node] + beside.get(node, 0) + _SPACE
            for node, x in zip(nodes, xs):
                pos[node] = (x - mean(xs), -d * dy)
        anchors = [(pos[child][0] + radius[child] + _GAP, pos[child][1] + radius[child], "left")
                   for _, child, _ in factors]
    else:
        # a layer name at the center of its factor's variables
        spread = 2 * r + _SPACE + max((w for w, _ in label_sizes), default=0.4)
        pos = nx.spring_layout(graph.to_undirected(), seed=0, scale=spread * max(1, len(graph) ** 0.5 / 2))
        anchors = [(*(mean(pos[n][i] for edge in edges for n in edge) for i in (0, 1)), "center")
                   for *_, edges in factors]

    xs, ys = [x for x, _ in pos.values()], [y for _, y in pos.values()]
    margin = r + _GAP
    fig = plt.figure(figsize=(max(xs) - min(xs) + 2 * margin, max(ys) - min(ys) + 2 * margin))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    node_size = [(2 * 72 * radius[node]) ** 2 for node in graph]  # marker area, in points squared
    fills = [next((fill for cls, fill in _FILLS.items() if isinstance(pgm.variables[node], cls)), "white")
             for node in graph]
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=node_size, node_color=fills, edgecolors="black",
                           linewidths=0.8)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=_NAME_SIZE)
    for (name, child, edges), (x, y, align) in zip(factors, anchors):
        for u, v in edges:
            # an edge that would cross a node bends around it: to the left of a downward edge,
            # away from the layer names
            nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=[(u, v)], node_size=node_size, width=1.2,
                                   edge_color=colors[name], arrows=True, arrowsize=12,
                                   arrowstyle="-|>" if child else "-",
                                   connectionstyle=f"arc3,rad={_bend(pos, radius, u, v):.3f}")
        if layer_names:
            ax.text(x, y, name, color=colors[name], fontsize=_LABEL_SIZE, ha=align, va="center",
                    gid="layer_name", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})

    # curves and names may reach past the axes: keep them, the tight bbox of the saved figure
    # takes them in
    for artist in ax.get_children():
        artist.set_clip_on(False)
    ax.axis("off")
    return fig
