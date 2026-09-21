"""One-batch CausalCGM diagnostic against the original implementation.

Run from the repository root:

    python tests/test_cgm_consistency.py

The goal is narrow: initialize our model by replaying the original CGM random
initialization order, then compare the training computation block by block.
No weights are copied from the original model into ours. The optimizer is then
kept alive across two consecutive training steps and the computation is
decomposed again after each step.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = ROOT.parent / "CausalCGM"
LABELS = ["Shape", "Size", "PosY", "PosX", "Color", "Label"]
TASK = "Label"
BATCH_SIZE = 128
EMBEDDING_SIZE = 8
GAMMA = 1
COMPARISON_RESULTS = []
CURRENT_SECTION = "setup"


def reset_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def start_check(title: str) -> None:
    global CURRENT_SECTION
    CURRENT_SECTION = title
    print(f"\n{title}")


def status_label(ok: bool) -> str:
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower()
    if "utf" in encoding:
        return "🟢 PASSED" if ok else "🔴 NOT PASSED"
    color = "\033[32m" if ok else "\033[31m"
    label = "[PASSED]" if ok else "[NOT PASSED]"
    return f"{color}{label}\033[0m" if sys.stdout.isatty() else label


def compare(name, actual, expected, *, atol=1e-6, rtol=1e-5, strict=False):
    """Compare tensor values, printing shape only as debugging context."""
    actual = actual.detach().cpu()
    expected = expected.detach().cpu()
    same_shape = actual.shape == expected.shape
    if same_shape:
        max_abs = (actual - expected).abs().max().item() if actual.numel() else 0.0
        values_close = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    else:
        max_abs = float("inf")
        values_close = False
    ok = same_shape and values_close
    status = status_label(ok)
    COMPARISON_RESULTS.append(
        {
            "section": CURRENT_SECTION,
            "name": name,
            "ok": ok,
            "values_close": values_close,
            "shape_ok": same_shape,
            "max_abs": max_abs,
            "actual_shape": tuple(actual.shape),
            "expected_shape": tuple(expected.shape),
        }
    )
    print(
        f"{status} | {name:<38} | "
        f"values_allclose={values_close} | max_abs={max_abs:.8g} | "
        f"shape_ok={same_shape} {tuple(actual.shape)} vs {tuple(expected.shape)}"
    )
    if strict and not ok:
        raise AssertionError(f"{name} mismatch")
    return ok


def print_comparison_summary() -> None:
    total = len(COMPARISON_RESULTS)
    passed = [result for result in COMPARISON_RESULTS if result["ok"]]
    failed = [result for result in COMPARISON_RESULTS if not result["ok"]]

    print("\nSUMMARY")
    print(f"Total checks: {total}")
    print(f"PASSED: {len(passed)}")
    print(f"NOT PASSED: {len(failed)}")

    if failed:
        print("\nNOT PASSED details:")
        for result in failed:
            print(
                f"  - {result['section']} :: {result['name']} "
                f"(values_allclose={result['values_close']}, "
                f"shape_ok={result['shape_ok']}, "
                f"max_abs={result['max_abs']:.8g}, "
                f"shape={result['actual_shape']} vs {result['expected_shape']})"
            )
    else:
        print("\nNOT PASSED details: none")

    print("\nPASSED by section:")
    for section in dict.fromkeys(result["section"] for result in COMPARISON_RESULTS):
        section_passed = [
            result for result in passed if result["section"] == section
        ]
        section_total = [
            result for result in COMPARISON_RESULTS if result["section"] == section
        ]
        print(f"  - {section}: {len(section_passed)}/{len(section_total)}")


def load_original_modules():
    sys.path.insert(0, str(ORIGINAL_ROOT))
    import matplotlib.style as mpl_style

    if not hasattr(mpl_style, "core"):
        mpl_style.core = mpl_style
    if "causallearn.search.PermutationBased.GRaSP" not in sys.modules:
        causallearn = types.ModuleType("causallearn")
        search = types.ModuleType("causallearn.search")
        permutation = types.ModuleType("causallearn.search.PermutationBased")
        grasp_module = types.ModuleType("causallearn.search.PermutationBased.GRaSP")
        grasp_module.grasp = lambda *_, **__: None
        sys.modules.setdefault("causallearn", causallearn)
        sys.modules.setdefault("causallearn.search", search)
        sys.modules.setdefault("causallearn.search.PermutationBased", permutation)
        sys.modules.setdefault("causallearn.search.PermutationBased.GRaSP", grasp_module)

    from causalcgm.causalcgm import CausalCGM as OriginalCausalCGM
    from causalcgm.utils import conditional_entropy_dag

    return OriginalCausalCGM, conditional_entropy_dag


def load_dsprites_arrays():
    data_dir = ORIGINAL_ROOT / "datasets" / "dsprites"

    def load(name):
        value = torch.from_numpy(np.load(data_dir / f"{name}.npy")).float()
        return value[:, None] if value.ndim == 1 else value

    return load("train_features"), load("train_concepts"), load("train_tasks")


def first_training_batch(seed: int):
    x_train, c_train, y_train = load_dsprites_arrays()
    reset_all_seeds(seed)
    permutation = torch.randperm(len(x_train))
    x_train = x_train[permutation]
    c_train = c_train[permutation]
    y_train = y_train[permutation]
    split = int(0.8 * len(x_train))
    x_fit = x_train[:split]
    c_fit = c_train[:split]
    y_fit = y_train[:split]
    s_fit = torch.cat([c_fit, y_fit], dim=1)
    loader = DataLoader(
        TensorDataset(x_fit, c_fit, y_fit),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    x, c, y = next(iter(loader))
    return s_fit, x, torch.cat([c, y], dim=1)


def make_annotations():
    from torch_concepts import Annotations

    return Annotations(
        labels=LABELS,
        cardinalities=[1] * len(LABELS),
        types=["binary"] * len(LABELS),
    )


def make_models(seed: int, fit_symbols):
    OriginalCausalCGM, conditional_entropy_dag = load_original_modules()
    from torch_concepts.graph_generator import (
        GraphGeneratorLearnable,
        entropy_initialization,
        remove_weakest_cycles,
    )
    from torch_concepts.nn import CausalCGM, CGMTrainingLoss, WeightedConceptLoss

    input_size = int(
        np.load(ORIGINAL_ROOT / "datasets" / "dsprites" / "train_features.npy").shape[1]
    )

    reset_all_seeds(seed)
    old = OriginalCausalCGM(
        input_size, EMBEDDING_SIZE, 5, 1, EMBEDDING_SIZE, GAMMA,
        0, 0.0, probabilistic=False, no_out_task=True,
    )
    old_cov = torch.tensor(conditional_entropy_dag(fit_symbols)).float()
    old_cov[-1, :] = 0
    old_cov = torch.clamp(old_cov / old_cov.mean(), 0, 0.99)
    old.concept_embedder.eq_model.fc1.weight = nn.Parameter(
        old_cov.clone(), requires_grad=True,
    )

    graph_generator = GraphGeneratorLearnable(
        name="dagma_cgm",
        concept_names=LABELS,
        task_names=[TASK],
        threshold=0.02,
        refinement=remove_weakest_cycles,
        initialization=entropy_initialization(fit_symbols),
    )
    loss = CGMTrainingLoss(
        prediction_loss=WeightedConceptLoss(
            concept_weight=1.0,
            task_weight=1.0,
            task_names=[TASK],
            binary=nn.BCEWithLogitsLoss(),
        ),
        lambda_dag=3.0,
        lambda_cace=0.0,
    )
    new = CausalCGM(
        input_size=input_size,
        annotations=make_annotations(),
        task_names=TASK,
        embedding_size=EMBEDDING_SIZE,
        graph_generator=graph_generator,
        run_interventions=True,
        lightning=True,
        loss=loss,
        optim_class=torch.optim.AdamW,
        optim_kwargs={"lr": 0.01},
    )
    initialize_new_like_original_rng_order(new, seed, fit_symbols)
    return old, new, old_cov


def reset_linear(linear: nn.Linear) -> None:
    linear.reset_parameters()


def reset_mlp_two_linears(mlp) -> None:
    reset_linear(mlp.mlp[0].affinity)
    reset_linear(mlp.readout)


def consume_original_post_predictor_rng() -> None:
    # Original CausalCGM constructs concept_prob_predictor_post even though the
    # training path compared here does not use it. Consume the same randomness.
    dummy = nn.Sequential(
        nn.Linear(2 * EMBEDDING_SIZE, EMBEDDING_SIZE),
        nn.LeakyReLU(),
        nn.Linear(EMBEDDING_SIZE, 1),
    )
    del dummy


def initialize_new_like_original_rng_order(new_model, seed: int, fit_symbols) -> None:
    """Replay the original constructor's RNG order on our modules.

    This does not read weights from the original model. It resets our modules in
    the same order that the original constructor creates equivalent modules.
    """
    from torch_concepts.graph_generator import entropy_initialization

    reset_all_seeds(seed)

    shared = new_model._shared_encoder[1]
    reset_linear(shared.mlp[0].affinity)
    reset_linear(shared.mlp[1].affinity)
    reset_linear(shared.readout)

    # Original eq_model constructs fc1 and then fc2[0]. fc1 is later overwritten
    # by entropy initialization in run.py, but it still consumes RNG.
    reset_linear(new_model.graph_generator.fc1)
    reset_linear(new_model.structural_equations.shared_structural_equations["2"])

    for node in range(len(LABELS)):
        reset_mlp_two_linears(new_model.exogenous_cpds[node].parametrization["value"][0])

        # Original has one concept_prob_predictor reused for prior and posterior.
        # Our model has a copy/prior predictor plus a final equation. Initialize
        # both from the same RNG segment, then advance RNG as if the original
        # unused post predictor had been constructed.
        state_before_predictor = torch.random.get_rng_state()
        reset_mlp_two_linears(new_model.endogenous_copy_cpds[node].parametrization["logits"])
        state_after_predictor = torch.random.get_rng_state()
        torch.random.set_rng_state(state_before_predictor)
        reset_mlp_two_linears(
            new_model.structural_equations.concept_structural_equations[node]
        )
        torch.random.set_rng_state(state_after_predictor)
        consume_original_post_predictor_rng()

    entropy_initialization(fit_symbols)(new_model.graph_generator)


def check_graph_initialization(old, new, old_cov, strict=False):
    start_check("CHECK 0 - DAG initialization before training")
    compare("raw entropy matrix", new.graph_generator.fc1.weight, old.concept_embedder.eq_model.fc1.weight, strict=strict)
    compare("raw matrix vs original cov", old.concept_embedder.eq_model.fc1.weight, old_cov, strict=strict)
    compare("edge mask", new.graph_generator.edge_mask, old.concept_embedder.eq_model.mask, strict=strict)
    compare("materialized adjacency", new.graph_generator(), old.concept_embedder.eq_model.fc1_to_adj(), strict=strict)


def old_new_blocks(old, new, x, s):
    old_h = old.encoder(x)
    new_h = new._shared_encoder(x)

    old_contexts = [
        old.concept_embedder.concept_context_generators[i](old_h)
        for i in range(len(LABELS))
    ]
    new_contexts = [
        new.exogenous_cpds[i].parametrization["value"](new_h)
        for i in range(len(LABELS))
    ]

    old_prior_logits = torch.cat([
        old.concept_embedder.concept_prob_predictor[i](old_contexts[i])
        for i in range(len(LABELS))
    ], dim=1)
    new_prior_logits = torch.cat([
        new.endogenous_copy_cpds[i].parametrization["logits"](new_contexts[i])
        for i in range(len(LABELS))
    ], dim=1)

    old_mixed = torch.stack([
        old.concept_embedder._build_concept_embedding(
            old_contexts[i], s[:, i:i + 1],
        )
        for i in range(len(LABELS))
    ], dim=1)
    new_banks = [
        context.unflatten(-1, (2, EMBEDDING_SIZE))
        for context in new_contexts
    ]
    new_mixed = new.mixer(new_banks, list(s.split(1, dim=1)))

    old_adj = old.concept_embedder.eq_model.fc1_to_adj()
    new_adj = new.graph_generator()
    old_agg = torch.matmul(old_mixed.transpose(1, 2), old_adj).transpose(1, 2)
    new.graph_layer.clear()
    new_agg = new.graph_layer(new_mixed)

    old_shared = torch.nn.functional.leaky_relu(old_agg)
    old_shared = old.concept_embedder.eq_model.fc2[0](old_shared)
    old_shared = torch.nn.functional.leaky_relu(old_shared)
    new_shared = new.structural_equations.shared_activation(new_agg)
    new_shared = new.structural_equations.shared_structural_equations["2"](new_shared)
    new_shared = new.structural_equations.shared_activation(new_shared)

    old_final_logits = torch.cat([
        old.concept_embedder.concept_prob_predictor[i](old_shared[:, i])
        for i in range(len(LABELS))
    ], dim=1)
    new_final_logits = torch.cat([
        new.structural_equations.concept_structural_equations[i](new_shared[:, i])
        for i in range(len(LABELS))
    ], dim=1)

    return {
        "shared embedding": (new_h, old_h),
        "exogenous contexts": (torch.stack(new_contexts, dim=1), torch.stack(old_contexts, dim=1)),
        "copy/prior logits": (new_prior_logits, old_prior_logits),
        "copy/prior probs": (torch.sigmoid(new_prior_logits), torch.sigmoid(old_prior_logits)),
        "mixed embeddings": (new_mixed, old_mixed),
        "adjacency": (new_adj, old_adj),
        "graph aggregation": (new_agg, old_agg),
        "shared structural": (new_shared, old_shared),
        "final logits": (new_final_logits, old_final_logits),
        "final probs": (torch.sigmoid(new_final_logits), torch.sigmoid(old_final_logits)),
    }


def check_decomposed_layers(
    old,
    new,
    x,
    s,
    strict=False,
    section="CHECK 1 - decomposed one-batch training path",
):
    start_check(section)
    for name, (actual, expected) in old_new_blocks(old, new, x, s).items():
        compare(name, actual, expected, strict=strict)


def check_training_losses(
    old_losses,
    new_losses,
    strict=False,
    section="CHECK 2 - training losses",
):
    start_check(section)
    compare("loss prior", new_losses["prior"], old_losses["prior"], strict=strict)
    compare("loss posterior", new_losses["posterior"], old_losses["posterior"], strict=strict)
    compare("loss DAG", new_losses["dag"], old_losses["dag"], strict=strict)
    compare("loss CACE", new_losses["cace"], old_losses["cace"], strict=strict)
    compare("loss total", new_losses["total"], old_losses["total"], strict=strict)


def check_training_interventions(
    old_train,
    new_train,
    strict=False,
    section="CHECK 3 - training interventions",
):
    start_check(section)
    compare(
        "low intervention probs",
        new_train.params["low_logits"].tensor.sigmoid(),
        old_train[4],
        strict=strict,
    )
    compare(
        "high intervention probs",
        new_train.params["high_logits"].tensor.sigmoid(),
        old_train[5],
        strict=strict,
    )
    compare(
        "training intervention CACE",
        cace_value(new_train.params["high_logits"].tensor.sigmoid(),
                   new_train.params["low_logits"].tensor.sigmoid()),
        cace_value(old_train[5], old_train[4]),
        strict=strict,
    )


def cace_value(high, low):
    return torch.abs(high.mean(dim=0) - low.mean(dim=0)).norm()


def old_training_loss_terms(old, old_train, s):
    old_prior = old.loss(old_train[0], s[:, :-1]) + old.loss(
        old_train[1].squeeze(), s[:, -1].squeeze()
    )
    old_posterior = old.loss(old_train[2], s[:, :-1]) + old.loss(
        old_train[3].squeeze(), s[:, -1].squeeze()
    )
    old_dag = 3 * old.concept_embedder.eq_model.h_func()
    old_cace = cace_value(old_train[5], old_train[4])
    old_total = old_prior + old_posterior + old_dag + old.lambda_cace / (
        old_cace + 1e-6
    )
    return {
        "prior": old_prior,
        "posterior": old_posterior,
        "dag": old_dag,
        "cace": old_cace,
        "total": old_total,
    }


def new_training_loss_terms(new, new_train, s):
    new_terms = new.loss.breakdown(new_train, new.prepare_target(s))
    new_prior = new_terms["prior"]
    new_posterior = new_terms["posterior"]
    new_dag = new_terms["dagma"]
    new_cace = cace_value(
        new_train.params["high_logits"].tensor.sigmoid(),
        new_train.params["low_logits"].tensor.sigmoid(),
    )
    new_total = sum(new_terms.values())
    return {
        "prior": new_prior,
        "posterior": new_posterior,
        "dag": new_dag,
        "cace": new_cace,
        "total": new_total,
    }


def make_optimizers(old, new):
    return old.configure_optimizers(), new.configure_optimizers()["optimizer"]


def materialized_old_dag(old):
    old.concept_embedder.compute_parent_indices()
    return torch.as_tensor(old.concept_embedder.dag).float()


def eval_raw_adjacency(new):
    for factor in new.eval_pgm._factors.values():
        trunk = getattr(factor, "trunk", None)
        if trunk is None:
            continue
        for module in trunk.modules():
            adjacency = getattr(module, "fixed_adjacency", None)
            if adjacency is not None:
                return adjacency.float()
    raise RuntimeError("No fixed eval adjacency found in the materialized PGM.")


def binary_embedding(context, value):
    return context[:, :EMBEDDING_SIZE] * value + context[:, EMBEDDING_SIZE:] * (1 - value)


def eval_old_new_blocks(old, new, x):
    import networkx as nx

    old.concept_embedder.compute_parent_indices()
    dag = new.graph.data.float()
    raw_adjacency = eval_raw_adjacency(new)
    root = dag.sum(dim=0) == 0
    order = list(nx.topological_sort(nx.from_numpy_array(
        dag.detach().cpu().numpy(), create_using=nx.DiGraph,
    )))

    old_h = old.encoder(x)
    new_h = new._shared_encoder(x)
    old_exogenous = [
        old.concept_embedder.concept_context_generators[i](old_h)
        for i in range(len(LABELS))
    ]
    new_exogenous = [
        new.eval_pgm._factors[f"{label}__u"].parametrization["value"](new_h)
        for label in LABELS
    ]

    old_embeddings = torch.zeros(x.shape[0], len(LABELS), EMBEDDING_SIZE)
    new_embeddings = torch.zeros_like(old_embeddings)
    old_agg = [
        x.new_zeros(x.shape[0], EMBEDDING_SIZE)
        for _ in LABELS
    ]
    new_agg = [
        x.new_zeros(x.shape[0], EMBEDDING_SIZE)
        for _ in LABELS
    ]
    old_shared = [torch.zeros_like(old_exogenous[0]) for _ in LABELS]
    new_shared = [torch.zeros_like(new_exogenous[0]) for _ in LABELS]
    old_logits = [None for _ in LABELS]
    new_logits = [None for _ in LABELS]
    old_probs = [None for _ in LABELS]
    new_probs = [None for _ in LABELS]
    shared_layer = new.structural_equations.shared_structural_equations["2"]

    for node in order:
        label = LABELS[node]
        if root[node]:
            old_root_input = old_exogenous[node]
            new_root_input = new_exogenous[node]
            old_shared[node] = old_root_input
            new_shared[node] = new_root_input
            old_logits[node] = old.concept_embedder.concept_prob_predictor[node](
                old_root_input
            )
            new_logits[node] = new.eval_pgm._factors[label](
                {f"{label}__u": new_root_input}
            )["logits"]
        else:
            old_aggregated = torch.matmul(
                old_embeddings.transpose(1, 2),
                old.concept_embedder.eq_model.fc1_to_adj(),
            ).transpose(1, 2)[:, node]
            new_aggregated = new.eval_pgm._factors[label].trunk[1](
                new_embeddings
            )[:, node]
            old_agg[node] = old_aggregated
            new_agg[node] = new_aggregated
            old_predictor_input = torch.nn.functional.leaky_relu(old_aggregated)
            old_predictor_input = old.concept_embedder.eq_model.fc2[0](
                old_predictor_input
            )
            old_predictor_input = torch.nn.functional.leaky_relu(
                old_predictor_input
            )
            new_structural_input = new.structural_equations.shared_activation(
                new_aggregated
            )
            new_structural_input = shared_layer(new_structural_input)
            new_structural_input = new.structural_equations.shared_activation(
                new_structural_input
            )
            old_shared[node] = old_predictor_input
            new_shared[node] = new_structural_input
            old_logits[node] = old.concept_embedder.concept_prob_predictor[node](
                old_predictor_input
            )
            new_logits[node] = (
                new.structural_equations.concept_structural_equations[node](
                    new_structural_input
                )
            )
        old_probs[node] = torch.sigmoid(old_logits[node])
        new_probs[node] = torch.sigmoid(new_logits[node])
        old_embeddings[:, node] = binary_embedding(
            old_exogenous[node], old_probs[node]
        )
        new_embeddings[:, node] = binary_embedding(
            new_exogenous[node], new_probs[node]
        )

    return {
        "shared embedding": (new_h, old_h),
        "exogenous": (torch.stack(new_exogenous, dim=1), torch.stack(old_exogenous, dim=1)),
        "raw adjacency": (raw_adjacency, old.concept_embedder.eq_model.fc1_to_adj()),
        "dag": (dag, torch.as_tensor(old.concept_embedder.dag).float()),
        "root mask": (root.float(), (torch.as_tensor(old.concept_embedder.dag).float().sum(dim=0) == 0).float()),
        "graph aggregation": (torch.stack(new_agg, dim=1), torch.stack(old_agg, dim=1)),
        "shared structural": (torch.stack(new_shared, dim=1), torch.stack(old_shared, dim=1)),
        "endogenous logits": (torch.cat(new_logits, dim=1), torch.cat(old_logits, dim=1)),
        "endogenous": (torch.cat(new_probs, dim=1), torch.cat(old_probs, dim=1)),
        "endogenous embeddings": (new_embeddings, old_embeddings),
    }


def check_materialized_eval_blocks(old, new, x, strict=False):
    start_check("CHECK 6 - materialized eval blocks")
    for name, (actual, expected) in eval_old_new_blocks(old, new, x).items():
        compare(f"eval {name}", actual, expected, strict=strict)


def check_materialized_dag_and_eval(seed, old, new, x, strict=False):
    start_check("CHECK 5 - materialized/binarized DAG and eval forward")
    old.eval()
    new.eval()
    with torch.no_grad():
        old_eval = old(x)
        new_eval_output = new(input=x)
        new_eval = new_eval_output.params["logits"].tensor.sigmoid()

    old_raw_adjacency = old.concept_embedder.eq_model.fc1_to_adj()
    new_raw_adjacency = eval_raw_adjacency(new)
    old_dag = materialized_old_dag(old)
    new_dag = new.graph.data.float()
    old_roots = (old_dag.sum(dim=0) == 0).float()
    new_roots = (new_dag.sum(dim=0) == 0).float()
    compare("eval raw adjacency", new_raw_adjacency, old_raw_adjacency, strict=strict)
    compare("materialized weighted DAG", new_dag, old_dag, strict=strict)
    compare("materialized binary DAG", (new_dag > 0).float(), (old_dag > 0).float(), strict=strict)
    compare("eval root mask", new_roots, old_roots, strict=strict)

    compare("eval forward after DAG", new_eval, old_eval, strict=strict)
    check_materialized_eval_blocks(old, new, x, strict=strict)


def new_intervention_forward(new, x, evidence):
    with torch.no_grad():
        output = new(
            input=x,
            query={label: None for label in LABELS if label not in evidence},
            evidence=evidence,
        )
        return torch.cat([
            evidence[label]
            if label in evidence
            else output.params[label]["logits"].sigmoid()
            for label in LABELS
        ], dim=1)


def paper_graph_info(adjacency):
    import networkx as nx

    dag = (adjacency > 0.1).float().cpu().numpy()
    graph = nx.from_numpy_array(dag, create_using=nx.DiGraph)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Expected the materialized CGM graph to be a DAG.")
    reachability = dag.copy()
    for source in range(dag.shape[0]):
        for target_index in range(dag.shape[1]):
            if source != target_index and nx.has_path(
                graph, source, target_index,
            ):
                reachability[source, target_index] = 1
    order = np.flip(np.argsort(reachability.sum(axis=1))).tolist()
    return (
        torch.as_tensor(dag).float(),
        torch.as_tensor(reachability).float(),
        order,
    )


def check_interventions_from_root(old, new, x, s, strict=False):
    start_check("CHECK 7 - interventions from roots")
    old.eval()
    new.eval()

    old_dag, old_reachability, old_order = paper_graph_info(
        materialized_old_dag(old)
    )
    new_dag, new_reachability, order = paper_graph_info(new.graph.data.float())
    compare("intervention DAG", new_dag, old_dag, strict=strict)
    compare(
        "intervention reachability",
        new_reachability,
        old_reachability,
        strict=strict,
    )
    compare(
        "intervention order",
        torch.as_tensor(order),
        torch.as_tensor(old_order),
        strict=strict,
    )
    for length in range(1, len(order) + 1):
        nodes = order[:length]
        labels = [LABELS[node] for node in nodes]
        evidence = {
            label: s[:, node:node + 1]
            for node, label in zip(nodes, labels)
        }
        with torch.no_grad():
            old_out = old(
                x,
                c=s,
                intervention_idxs=nodes,
                train=False,
            )
        new_out = new_intervention_forward(new, x, evidence)
        compare(
            f"do prefix {' -> '.join(labels)}",
            new_out,
            old_out,
            strict=strict,
        )


def pns_pairs_to_bounds(matrix):
    n_symbols = len(LABELS)
    lower = np.full((n_symbols, n_symbols), np.nan)
    upper = np.full((n_symbols, n_symbols), np.nan)
    for source, row in enumerate(matrix):
        for target, value in enumerate(row):
            if isinstance(value, tuple) or isinstance(value, list):
                lower[source, target] = value[0]
                upper[source, target] = value[1]
    return torch.as_tensor(lower).float(), torch.as_tensor(upper).float()


def new_compute_pns_matrix(new, x, reachability):
    lower = np.full((len(LABELS), len(LABELS)), np.nan)
    upper = np.full_like(lower, np.nan)
    with torch.no_grad():
        for source, label in enumerate(LABELS):
            zero_evidence = {
                label: torch.zeros(x.shape[0], 1, device=x.device)
            }
            one_evidence = {
                label: torch.ones(x.shape[0], 1, device=x.device)
            }
            predicted_zero = (
                new_intervention_forward(new, x, zero_evidence) > 0.5
            ).float()
            predicted_one = (
                new_intervention_forward(new, x, one_evidence) > 0.5
            ).float()
            probability_zero = predicted_zero.mean(dim=0).cpu().numpy()
            probability_one = predicted_one.mean(dim=0).cpu().numpy()
            for target, edge in enumerate(reachability[source]):
                if edge != 0:
                    lower[source, target] = max(
                        0.0,
                        probability_one[target] - probability_zero[target],
                    )
                    upper[source, target] = min(
                        probability_one[target],
                        1.0 - probability_zero[target],
                    )
    return (
        torch.as_tensor(np.round(lower, 2)).float(),
        torch.as_tensor(np.round(upper, 2)).float(),
    )


def check_pns_calculation(old, new, x, strict=False):
    start_check("CHECK 8 - PNS calculation")
    old.eval()
    new.eval()
    from causalcgm.utils import compute_pns_matrix as old_compute_pns_matrix

    old_dag, old_reachability, _ = paper_graph_info(materialized_old_dag(old))
    new_dag, new_reachability, _ = paper_graph_info(new.graph.data.float())
    compare("PNS DAG", new_dag, old_dag, strict=strict)
    compare("PNS reachability", new_reachability, old_reachability, strict=strict)

    old_pns = old_compute_pns_matrix(
        x,
        old,
        old_dag.detach().cpu().numpy().copy(),
    )
    old_lower, old_upper = pns_pairs_to_bounds(old_pns)
    new_lower, new_upper = new_compute_pns_matrix(new, x, new_reachability)
    compare("PNS lower finite mask", torch.isfinite(new_lower).float(), torch.isfinite(old_lower).float(), strict=strict)
    compare("PNS lower bound", torch.nan_to_num(new_lower, nan=-1), torch.nan_to_num(old_lower, nan=-1), strict=strict)
    compare("PNS upper finite mask", torch.isfinite(new_upper).float(), torch.isfinite(old_upper).float(), strict=strict)
    compare("PNS upper bound", torch.nan_to_num(new_upper, nan=-1), torch.nan_to_num(old_upper, nan=-1), strict=strict)


def run(seed: int, strict: bool) -> None:
    COMPARISON_RESULTS.clear()
    fit_symbols, x, s = first_training_batch(seed)
    old, new, old_cov = make_models(seed, fit_symbols)
    print(f"Batch x={tuple(x.shape)} target={tuple(s.shape)}")
    try:
        check_graph_initialization(old, new, old_cov, strict=strict)
        check_decomposed_layers(
            old,
            new,
            x,
            s,
            strict=strict,
            section="CHECK 1 - decomposed layers after initialization",
        )
        old_optimizer, new_optimizer = make_optimizers(old, new)
        for step in (1, 2):
            step_seed = seed + step
            print(f"\nTRAIN STEP {step} - one AdamW optimizer update")
            old_optimizer.zero_grad(set_to_none=True)
            new_optimizer.zero_grad(set_to_none=True)
            old.train()
            new.train()

            reset_all_seeds(step_seed)
            old_train = old(
                x, c=s, intervention_idxs=torch.arange(s.shape[1]), train=True,
            )
            old_losses = old_training_loss_terms(old, old_train, s)

            reset_all_seeds(step_seed)
            new_train = new(input=x, target=new.prepare_target(s))
            new_losses = new_training_loss_terms(new, new_train, s)

            old_losses["total"].backward()
            new_losses["total"].backward()
            old_optimizer.step()
            new_optimizer.step()
            print(
                f"completed AdamW step {step} with aligned intervention seed "
                f"{step_seed}."
            )

            check_decomposed_layers(
                old,
                new,
                x,
                s,
                strict=strict,
                section=(
                    f"CHECK {1 + step}A - decomposed path after "
                    f"training step {step}"
                ),
            )
            check_training_interventions(
                old_train,
                new_train,
                strict=strict,
                section=(
                    f"CHECK {1 + step}B - training interventions during "
                    f"training step {step}"
                ),
            )
            check_training_losses(
                old_losses,
                new_losses,
                strict=strict,
                section=(
                    f"CHECK {1 + step}C - training losses used for "
                    f"training step {step}"
                ),
            )
        check_materialized_dag_and_eval(seed, old, new, x, strict=strict)
        check_interventions_from_root(old, new, x, s, strict=strict)
        check_pns_calculation(old, new, x, strict=strict)
    finally:
        print_comparison_summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    ok = False
    try:
        if args.verbose:
            run(args.seed, args.strict)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                run(args.seed, args.strict)
            print(".", end="")
        ok = True
    except Exception:
        if not args.verbose:
            print("E", end="")
        raise
    finally:
        elapsed = time.perf_counter() - start
        if not args.verbose:
            print()
            print("-" * 70)
            print(f"Ran 1 test in {elapsed:.3f}s")
            print()
            if ok:
                print("OK")


if __name__ == "__main__":
    main()
