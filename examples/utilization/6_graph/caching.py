"""
Example: Explicit Graph Precomputation (data side)

A causal graph can be computed before training a graph-aware concept model:

-- ``precompute_graph`` first shows that an unrefined GES graph can fail DAG
  validation, then uses an LLM to orient its ambiguous edges and caches the
  resulting fixed ``ConceptGraph`` on disk.

This is an explicit preprocessing step and happens before ``setup()``. During
training the graph is fixed; the model's ``MLP`` operates on the vectors
already provided by the Asia dataset.

Flow:
1. Generate samples from the Asia Bayesian network.
2. Run GES without refinement and display the error if its CPDAG is not a DAG.
3. Repeat GES with an LLM refinement that orients ambiguous edges.
4. Reuse the refined graph from memory without rerunning GES or the LLM.
5. Train ``CausallyReliableConceptBottleneckModel`` using the fixed graph.

Unrefined graphs are cached on disk; callable refinements use the in-memory cache. Pass ``force=True`` to recompute it.

Optional dependency: ``pip install causal-learn``.
The refinement also requires ``litellm`` and an API key.

The same graph API exists one level down on the dataset itself:
``dm.dataset.precompute_graph(generator, cache=True, force=False)``.
"""
import matplotlib.style as mpl_style
import torch

if not hasattr(mpl_style, "core"):
    mpl_style.core = mpl_style

from torch_concepts.graph_generator import compose_refinements, dfs_remove_cycles, refine_llm
from torch_concepts.llm_backends import LiteLLMBackend
from torch_concepts.env import get_env
import time
from pathlib import Path

from pytorch_lightning import Trainer
from torchmetrics.classification import BinaryAccuracy

from torch_concepts import seed_everything
from torch_concepts.data import BnLearnDataModule
from torch_concepts.graph_generator import GraphGeneratorFixed
from torch_concepts.nn import (
    CausallyReliableConceptBottleneckModel,
    ConceptLoss,
    ConceptMetrics,
    MLP,
)
from torch_concepts.nn.modules.mid.inference.torch.deterministic import (
    DeterministicInference,
)


LLM_MODEL = get_env("GRAPH_LLM_MODEL", "groq/openai/gpt-oss-20b")
LLM_API_KEY = get_env("GROQ_API_KEY")
DOMAIN = "medical diagnosis"
OUTPUT_DIR = Path("output")

DATASET_LABEL_DESCRIPTIONS = {
    "asia": "Whether the patient has recently travelled to Asia.",
    "smoke": "Whether the patient smokes.",
    "lung": "Whether the patient has lung cancer.",
    "tub": "Whether the patient has tuberculosis.",
    "bronc": "Whether the patient has bronchitis.",
    "either": "Whether tuberculosis or lung cancer is present.",
    "xray": "Whether the chest X-ray is abnormal.",
    "dysp": "Whether the patient experiences dyspnea.",
}

# Explicit descriptions override the dataset descriptions only for these names.
NEW_LABEL_DESCRIPTIONS = {
    "smoke": "Patient has a sustained history of tobacco use.",
    "dysp": "Patient reports clinically significant shortness of breath.",
}


def main():
    seed_everything(42)

    # 1. Generate samples from the Asia Bayesian network.
    dm = BnLearnDataModule(
        name="asia", n_gen=2_000, batch_size=128, seed=42,
        label_descriptions=DATASET_LABEL_DESCRIPTIONS,
    )

    # 2. GES commonly returns a partially directed graph. DAG validation is on
    # by default, so ambiguous reciprocal edges produce an actionable error.
    try:
        dm.precompute_graph(GraphGeneratorFixed(name="ges"))
    except ValueError as error:
        print("Expected non-DAG error:")
        print(error)

    # Materialize the unrefined CPDAG without DAG validation so it can be
    # inspected and plotted. This pass is intentionally not cached.
    raw_generator = GraphGeneratorFixed(name="ges", require_dag=False)
    dm.precompute_graph(raw_generator, cache=False, force=True)
    raw_plot = dm.graph.plot(OUTPUT_DIR / "ges_unrefined", title="GES (unrefined)")
    print(f"Unrefined graph plot: {raw_plot}")

    if not LLM_API_KEY:
        raise RuntimeError(
            "Set GROQ_API_KEY in .env to run the GES + LLM refinement."
        )

    backend = LiteLLMBackend(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
    )
    refinement = compose_refinements(
        refine_llm(
            llm_backend=backend,
            domain=DOMAIN,
            concept_descriptions={
                **DATASET_LABEL_DESCRIPTIONS,
                **NEW_LABEL_DESCRIPTIONS,
            },
            repeats=1,
        ),
        dfs_remove_cycles,
    )

    # if new descriptions are provided, they override the dataset descriptions for the
    # concepts for which they are specified.
    generator = GraphGeneratorFixed(
        name="ges",
        refinement=refinement,
    )

    # 3. Recompute GES and ask the LLM to orient its ambiguous edges. The
    # refined DAG is cached in memory.
    t0 = time.perf_counter()
    dm.precompute_graph(generator, cache=True, force=True)
    print(f"First precompute_graph call:  {time.perf_counter() - t0:.2f}s")

    # 4. Load the same refined graph without rerunning GES or the LLM.
    t0 = time.perf_counter()
    dm.precompute_graph(generator, cache=True)
    print(f"Second precompute_graph call: {time.perf_counter() - t0:.2f}s (cache hit)")
    print("Precomputed Asia graph:\n", dm.graph.to_pandas())
    refined_plot = dm.graph.plot(OUTPUT_DIR / "ges_refined", title="GES + LLM refinement")
    print(f"Refined graph plot: {refined_plot}")

    # 5. The causally reliable CBM consumes the already materialized graph.
    model = CausallyReliableConceptBottleneckModel(
        input_size=dm.n_features[-1],
        annotations=dm.annotations,
        graph=dm.graph,
        backbone=MLP(dm.n_features[-1], 128),
        latent_size=128,
        embedding_size=8,
        hypernet_hidden_size=8,
        lightning=True,
        train_inference=DeterministicInference,
        loss=ConceptLoss(binary=torch.nn.BCEWithLogitsLoss()),
        metrics=ConceptMetrics(
            annotations=dm.annotations,
            summary=True,
            per_concept=True,
            binary={"accuracy": BinaryAccuracy()},
        ),
        optim_class=torch.optim.AdamW,
        optim_kwargs={"lr": 0.01},
    )

    trainer = Trainer(
        max_epochs=20,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
