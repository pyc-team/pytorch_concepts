"""
Example: Graph Generation from a BNLearn DataModule

This example shows five ways to obtain a concept graph:

1. ``ground-truth`` uses the graph stored in the dataset, when available.
2. ``causallearn`` discovers a causal graph with PC or GES.
3. ``llm`` asks an LLM to causally reason about every pair of concepts.
4. ``causallearn+llm`` discovers a graph from data with a CausalLearn algorithm and asks an LLM to orient
   its ambiguous edges.
5. ``wanda`` learns a differentiable graph with gradient descent.

Every generator is selected and run through ``DataModule.precompute_graph``.
With ``use_as_gt=True``, the resulting graph is exposed as ``DataModule.graph``.
With ``use_as_gt=False``, the generator is retained for end-to-end training.

The LLM-based strategies use Groq through LiteLLM. Set ``LLM_API_KEY`` below
and enable the corresponding flags to execute them.

Optional dependencies:

- ``pip install causal-learn`` for CausalLearn.
- ``pip install litellm`` for the LLM-based strategies.
- ``pip install pydot matplotlib`` to save the generated graphs.
- The Graphviz system executable (``dot``) is also required.
"""

from pathlib import Path

from torch_concepts import seed_everything
from torch_concepts.data import BnLearnDataModule


# Configuration shared by the LLM-based strategies.
RUN_LLM_EXAMPLE = False
RUN_HYBRID_EXAMPLE = True
LLM_MODEL = "groq/openai/gpt-oss-20b"
LLM_API_KEY = ""  # Paste your Groq API key here.
DOMAIN = "medical diagnosis"
EXPERIMENT_NAME = "bnlearn_graph_generation"
ASIA_LABEL_DESCRIPTIONS = {
    "asia": "Whether the patient has recently visited Asia.",
    "tub": "Whether the patient has tuberculosis.",
    "smoke": "Whether the patient is a smoker.",
    "lung": "Whether the patient has lung cancer.",
    "bronc": "Whether the patient has bronchitis.",
    "either": "Whether the patient has either tuberculosis or lung cancer.",
    "xray": "Whether the patient has an abnormal chest X-ray result.",
    "dysp": "Whether the patient has dyspnea (shortness of breath).",
}



def main():
    seed_everything(42)

    api_key = LLM_API_KEY

    # Keep each experiment's generated files in a separate output directory.
    outputs_dir = Path(__file__).resolve().parents[3] / "outputs"
    plots_dir = outputs_dir / EXPERIMENT_NAME / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # The DataModule creates samples from the built-in Asia Bayesian network.
    n_samples = 10000
    batch_size = 2048
    datamodule = BnLearnDataModule(
        seed=42,
        name="asia",
        n_gen=n_samples,
        batch_size=batch_size,
        val_size=0.1,
        test_size=0.2,
        label_descriptions=ASIA_LABEL_DESCRIPTIONS,
    )

    dataset = datamodule.dataset

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------
    # Every generation mode goes through the DataModule. The generator is
    # selected internally from ``name`` and ``source``.
    datamodule.precompute_graph(
        name="ground_truth",
        source="GroundTruth",
    )
    ground_truth_graph = datamodule.graph
    print("\nGround-truth graph")
    print(ground_truth_graph.to_pandas())
    ground_truth_graph.plot(
        plots_dir / "ground_truth",
        title="Ground truth",
    )

    # ------------------------------------------------------------------
    # CausalLearn: score-based method (GES)
    # ------------------------------------------------------------------
    # Generate a causal graph using the GES algorithm from CausalLearn.
    datamodule.precompute_graph(name="ges", source="Causallearn")
    ges_graph = datamodule.graph
    print("\nCausalLearn graph: GES (score-based)")
    print(ges_graph.to_pandas())
    ges_graph.plot(
        plots_dir / "causallearn_ges",
        title="CausalLearn: GES (score-based)",
    )

    # ------------------------------------------------------------------
    # CausalLearn: constraint-based method (PC)
    # ------------------------------------------------------------------
    # Generate a causal graph using the PC algorithm from CausalLearn.
    datamodule.precompute_graph(name="pc", source="Causallearn")
    pc_graph = datamodule.graph
    print("\nCausalLearn graph: PC (constraint-based)")
    print(pc_graph.to_pandas())
    pc_graph.plot(
        plots_dir / "causallearn_pc",
        title="CausalLearn: PC (constraint-based)",
    )

    if RUN_LLM_EXAMPLE:
        if not api_key:
            raise RuntimeError(
                "Set LLM_API_KEY in this file to run the LLM-based examples."
            )

        # ------------------------------------------------------------------
        # LLM
        # ------------------------------------------------------------------
        datamodule.precompute_graph(
            name=LLM_MODEL,
            source="LLM",
            api_key=api_key,
            domain=DOMAIN,
            use_rag=False,
        )
        llm_graph = datamodule.graph
        print("\nLLM graph")
        print(llm_graph.to_pandas())
        llm_graph.plot(
            plots_dir / "llm",
            title="LLM",
        )

    if RUN_HYBRID_EXAMPLE:
        if not api_key:
            raise RuntimeError(
                "Set LLM_API_KEY in this file to run the LLM-based examples."
            )
        # ------------------------------------------------------------------
        # CausalLearn + LLM
        # ------------------------------------------------------------------
        datamodule.precompute_graph(
            name="ges",
            source="Causallearn",
            refinement={
                "name": LLM_MODEL,
                "source": "LLM",
                "api_key": api_key,
                "domain": DOMAIN,
                "use_rag": False,
            },
        )
        hybrid_graph = datamodule.graph
        print("\nCausalLearn + LLM graph")
        print(hybrid_graph.to_pandas())
        hybrid_graph.plot(
            plots_dir / "causallearn_llm",
            title="CausalLearn + LLM",
        )

    # ------------------------------------------------------------------
    # WANDA: differentiable graph generation
    # ------------------------------------------------------------------
    # Learnable generators are trained end-to-end with the model. This
    # standalone call only materializes the generator's current parameters.
    datamodule.precompute_graph(
        name="wanda",
        source="WANDA",
        use_as_gt=True,
        concept_names=list(dataset.concept_names),
        threshold_init=0.5,
    )
    wanda_graph = datamodule.graph
    print("\nWANDA graph (current parameter snapshot)")
    print(wanda_graph.to_pandas())

    wanda_graph.plot(
        plots_dir / "wanda",
        title="WANDA",
    )

    # Graph precomputation is complete; setup can now create the data splits.
    datamodule.setup("fit")
    print(f"\nPlots saved in: {plots_dir}")


if __name__ == "__main__":
    main()
