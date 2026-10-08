"""
Example: Using CausallyReliableConceptBottleneckModel
"""

import os

from pathlib import Path
from joblib import parallel_backend

import matplotlib.style as mpl_style

if not hasattr(mpl_style, "core"):
    mpl_style.core = mpl_style

from dotenv import load_dotenv

from torch_concepts.graphs import GraphGeneratorStatic, dfs_remove_cycles, refine_llm
from torch_concepts.llm_backends import LiteLLMBackend
import torch
from pytorch_lightning import Trainer
import torchmetrics

from torch_concepts import seed_everything

from torch_concepts.nn import CausallyReliableConceptBottleneckModel, MLP
from torch_concepts.nn.modules.loss import ConceptLoss
from torch_concepts.nn.modules.metrics import ConceptMetrics
from torch_concepts.data import BnLearnDataModule
from torch_concepts.nn.modules.mid.inference.torch.deterministic import DeterministicInference


ASIA_LABEL_DESCRIPTIONS = {
    "asia": "Recent travel to Asia; an exposure risk factor that can increase the chance of tuberculosis.",
    "smoke": "Smoking status; a risk factor that can increase the chance of lung cancer and bronchitis.",
    "lung": "Lung cancer diagnosis; a disease that can make either true and can contribute to dyspnea.",
    "tub": "Tuberculosis diagnosis; a disease that can make either true and can contribute to dyspnea.",
    "bronc": "Bronchitis diagnosis; a respiratory disease influenced by smoking and able to contribute to dyspnea.",
    "either": "A deterministic logical indicator equal to tub OR lung; it is caused by tuberculosis or lung cancer and can explain an abnormal chest X-ray.",
    "xray": "Abnormal chest X-ray result; an observation influenced by either tuberculosis or lung cancer being present.",
    "dysp": "Dyspnea, or shortness of breath; a symptom influenced by bronchitis and by either tuberculosis or lung cancer.",
}


def main():

    repo_root = Path(__file__).resolve().parents[3]
    load_dotenv(repo_root / "torch_concepts" / ".env")
    seed_everything(42)

    PLOTS_DIR = Path(__file__).resolve().parents[3] / "outputs" / "causally_reliable" / "plots"
    LLM_MODEL = "groq/openai/gpt-oss-20b"
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY in the environment or .env before running this example.")

    # Generate toy data
    print("=" * 60)
    print("Step 1: Generate Asia datamodule")
    print("=" * 60)

    n_samples = 2000
    batch_size = 64
    # BIFReader requests all cores; load in-process to avoid worker crashes.
    datamodule = BnLearnDataModule(seed=42,
                                    name='asia',
                                    root='data/asia_causally_reliable',
                                    n_gen=n_samples,
                                    batch_size=batch_size,
                                    label_descriptions=ASIA_LABEL_DESCRIPTIONS)

    # Setup LLM backend
    backend = LiteLLMBackend(
        model=LLM_MODEL,
        api_key=api_key,
        temperature=0,
        max_tokens=200,
        retry_on_rate_limit=True,
        max_rate_limit_wait=120.0,
    )

    # precompute graph with GES + LLM
    print("=" * 60)
    print("Step 2: Precompute graph with GES + LLM")
    print("=" * 60)
    # setup('fit') is required before precompute_graph to use only training data.
    datamodule.setup('fit')
    datamodule.precompute_graph(GraphGeneratorStatic(
        name="ges",
        source="Causallearn",
        refinement=[
            refine_llm(
                llm_backend=backend,
                domain="medical diagnosis",
                concept_descriptions=ASIA_LABEL_DESCRIPTIONS,
                repeats=1,
            ),
            dfs_remove_cycles,
        ],
    ))
    graph_ges_llm = datamodule.dataset.graph
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    graph_ges_llm.plot(PLOTS_DIR / "ges_llm", title="GES + LLM")

    annotations = datamodule.annotations
    concept_names = list(ASIA_LABEL_DESCRIPTIONS.keys())[:-1]  # all except the last one ("dysp")
    task_names = ["dysp"]

    n_features = datamodule.n_features[-1]
    n_concepts = len(concept_names)
    n_tasks = len(task_names)

    print(f"Input features: {n_features}")
    print(f"Concepts: {n_concepts} - {concept_names}")
    print(f"Tasks: {n_tasks} - {task_names}")
    print(f"Training samples: {datamodule.n_samples}")

    # Init model
    print("\n" + "=" * 60)
    print("Step 3: Initialize a CausallyReliableConceptBottleneckModel")
    print("=" * 60)

    # Define loss function
    loss_fn = ConceptLoss(
        binary = torch.nn.BCEWithLogitsLoss()
    )

    metrics = ConceptMetrics(
        annotations=annotations,
        summary=True,
        per_concept=True,
        binary = {'accuracy': torchmetrics.classification.BinaryAccuracy()}
    )

    # Initialize the model
    model = CausallyReliableConceptBottleneckModel(
        input_size=n_features,
        annotations=annotations,
        graph=graph_ges_llm,
        embedding_size=8,
        hypernet_hidden_size=8,
        backbone=MLP(input_size=n_features, hidden_size=16, n_layers=1),
        latent_size=16,
        lightning=True,
        train_inference=DeterministicInference,
        loss=loss_fn,
        metrics=metrics,
        optim_class=torch.optim.AdamW,
        optim_kwargs={'lr': 0.01}
    )

    print(f"Model created successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Encoder output features: {model.latent_size}")

    # Step 4: Training loop with lightning
    print("\n" + "=" * 60)
    print("Step 4: Training loop with lightning")
    print("=" * 60)

    trainer = Trainer(max_epochs=20)

    model.train()
    trainer.fit(model, datamodule=datamodule)
    print("Training completed successfully!")

    # Evaluate
    print("\n" + "=" * 60)
    print("Step 5: Test the model")
    print("=" * 60)
    trainer.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()
