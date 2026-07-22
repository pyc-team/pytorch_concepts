"""Benchmark: AnnotatedTensor label-based slicing overhead vs. raw torch indexing.

Sweeps the concept count N and times four access patterns:

  1. single      -- one concept,                e.g. t['c0']       vs data[:, 0]
  2. plate       -- a contiguous run via a registered plate name,
                    e.g. t['plate']              vs data[:, :k]
  3. consecutive -- the same contiguous run via an explicit label list,
                    e.g. t[['c0', ..., 'ck']]    vs data[:, :k]
  4. scattered   -- a non-consecutive run via an explicit label list,
                    e.g. t[['c0', 'c2', 'c4']]   vs data[:, [0, 2, 4]]

Each pattern is timed three ways, to separate *why* it costs what it costs:

  - by name     the real call, t[query]: looks the name(s) up AND wraps the result.
  - wrap only   the same result built from an already-resolved (selector,
                sub-annotation) pair -- same wrapping, but the name lookup
                itself is skipped, so this is the cost IF the lookup were free.
  - raw torch   the equivalent raw, unwrapped torch indexing op -- no
                AnnotatedTensor involved at all.

From these:
  - lookup overhead = by name / wrap only    -- cost of resolving the name(s)
                                                 alone, wrapping cost cancelled out.
  - wrap overhead   = wrap only / raw torch  -- fixed cost of wrapping a result
                                                 into an AnnotatedTensor, alone.

All concepts are binary (one column each) so a label maps 1:1 to a column
index, keeping the comparison purely about label resolution, not cardinality
bookkeeping.
"""
import time

import torch

from torch_concepts.annotations import Annotations
from torch_concepts.tensor import AnnotatedTensor

BATCH = 64
ITERS = 200
NS = [2, 10, 100, 1_000, 10_000, 100_000, 200_000]


def timeit(fn, iters=ITERS):
    fn()  # warm-up: pay any one-time cache-miss cost outside the timed loop
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) / iters * 1e6  # microseconds/call


def build(n):
    """n binary concepts 'c0'..'c{n-1}' over a (BATCH, n) tensor."""
    labels = [f"c{i}" for i in range(n)]
    ann = Annotations(labels=labels, cardinalities=[1] * n, types=["binary"] * n)
    data = torch.randn(BATCH, n)
    return labels, data, AnnotatedTensor(data, ann, axis=-1)


def wrap_only_fn(ann, data, query):
    """Same result as ``t[query]``, with the (selector, sub-annotation) already
    resolved -- isolates the fixed cost of wrapping from the cost of the lookup."""
    keys = (query,) if isinstance(query, str) else tuple(query)
    selector, sub_ann = ann.resolve(keys)
    idx = (Ellipsis, selector)
    return lambda: AnnotatedTensor(data[idx], sub_ann, axis=-1)


def run(n):
    labels, data, t = build(n)
    ann = t.annotation
    run_len = max(1, n // 10)

    plate_labels = labels[:run_len]
    t.register_plate_label("plate", plate_labels)
    scattered_labels = labels[: 2 * run_len : 2]
    scattered_idx = list(range(0, 2 * run_len, 2))

    # (query for t[...], equivalent raw torch index)
    queries = {
        "single": (labels[0], 0),
        "plate": ("plate", slice(0, run_len)),
        "consecutive": (plate_labels, slice(0, run_len)),
        "scattered": (scattered_labels, scattered_idx),
    }

    results = {}
    for pattern, (query, torch_idx) in queries.items():
        results[pattern] = (
            lambda q=query: t[q],
            wrap_only_fn(ann, data, query),
            lambda ti=torch_idx: data[:, ti],
        )
    return results


LEGEND = """\
  by name     t[query]                                     (lookup + wrap)
  wrap only   AnnotatedTensor(...) from an already-resolved (selector, sub-
              annotation) -- same wrapping, lookup skipped  (wrap only)
  raw torch   data[...]                                     (no wrapping)

  lookup overhead = by name / wrap only    -- cost of the name lookup, alone
  wrap overhead   = wrap only / raw torch  -- cost of wrapping the result, alone
"""


def main():
    print(LEGEND)
    header = (
        f"{'N':>8}  {'pattern':<12}{'by name':>10}{'wrap only':>11}{'raw torch':>11}"
        f"{'lookup ovh':>13}{'wrap ovh':>11}"
    )
    print(header)
    print("-" * len(header))
    for n in NS:
        for i, (pattern, (by_name_fn, wrap_only_fn_, raw_torch_fn)) in enumerate(run(n).items()):
            t_by_name = timeit(by_name_fn)
            t_wrap_only = timeit(wrap_only_fn_)
            t_raw_torch = timeit(raw_torch_fn)
            lookup_overhead = t_by_name / t_wrap_only if t_wrap_only > 0 else float("nan")
            wrap_overhead = t_wrap_only / t_raw_torch if t_raw_torch > 0 else float("nan")
            label = n if i == 0 else ""
            print(
                f"{label!s:>8}  {pattern:<12}{t_by_name:>8.2f}u{t_wrap_only:>9.2f}u{t_raw_torch:>9.2f}u"
                f"{lookup_overhead:>12.2f}x{wrap_overhead:>10.2f}x"
            )
        print()


if __name__ == "__main__":
    main()
