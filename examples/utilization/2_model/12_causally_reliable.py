"""
Example: Using CausallyReliableConceptBottleneckModel
"""

import os
import re
import time

from pathlib import Path

import matplotlib.style as mpl_style

if not hasattr(mpl_style, "core"):
    mpl_style.core = mpl_style

from torch_concepts.graph_generator import compose_refinements, dfs_remove_cycles, refine_llm
from torch_concepts.llm_backends import LiteLLMBackend
import torch
from pytorch_lightning import Trainer
import torchmetrics

from torch_concepts import seed_everything

from torch_concepts.nn import CausallyReliableConceptBottleneckModel, MLP
from torch_concepts.nn.modules.loss import ConceptLoss
from torch_concepts.nn.modules.metrics import ConceptMetrics
from torchmetrics.classification import BinaryAccuracy
from torch_concepts.data import BnLearnDataModule
from torch_concepts.nn.modules.mid.inference.torch.deterministic import DeterministicInference
from torch_concepts.graph_generator import GraphGeneratorFixed


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

REFINEMENT_LABEL_DESCRIPTIONS = {
    "asia": "Recent travel to Asia; an upstream exposure risk factor for tuberculosis, not a symptom or disease.",
    "tub": "Active tuberculosis infection; a disease that is downstream of Asia travel risk and upstream of either/dyspnea.",
}

TASK_NAMES = ["dysp"]
CONCEPT_NAMES = [
    "asia",
    "smoke",
    "lung",
    "tub",
    "bronc",
    "either",
    "xray",
]
SUPERVISED_NAMES = CONCEPT_NAMES + TASK_NAMES


class RateLimitedBackend:
    """Small experiment-local pacer for LLM rate limits."""

    def __init__(
        self,
        backend,
        min_interval=2.5,
        max_retries=3,
        rate_limit_wait=65.0,
    ):
        self.backend = backend
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.rate_limit_wait = rate_limit_wait
        self._last_call = 0.0

    def __getattr__(self, name):
        return getattr(self.backend, name)

    @staticmethod
    def _is_rate_limit(error):
        text = str(error).lower()
        return "ratelimit" in text or "rate limit" in text or "429" in text

    def _pace(self):
        elapsed = time.monotonic() - self._last_call
        wait = self.min_interval - elapsed
        if wait > 0:
            time.sleep(wait)

    def _wait_seconds(self, error, attempt):
        match = re.search(r"try again in\s+(\d+(?:\.\d+)?)s", str(error), re.I)
        if match:
            return max(float(match.group(1)) + 1.0, self.rate_limit_wait)
        return self.rate_limit_wait * attempt

    def __call__(self, *args, **kwargs):
        for attempt in range(1, self.max_retries + 1):
            try:
                self._pace()
                response = self.backend(*args, **kwargs)
                self._last_call = time.monotonic()
                return response
            except Exception as error:
                self._last_call = time.monotonic()
                if not self._is_rate_limit(error) or attempt == self.max_retries:
                    raise
                wait = self._wait_seconds(error, attempt)
                print(
                    f"Rate limit from LLM backend; pausing {wait:.1f}s "
                    f"before retry {attempt + 1}/{self.max_retries}."
                )
                time.sleep(wait)

def main():

    seed_everything(42)

    PLOTS_DIR = Path(__file__).resolve().parents[3] / "outputs" / "causally_reliable" / "plots"
    LLM_MODEL = "groq/openai/gpt-oss-20b"
    api_key = os.environ.get("GROQ_API_KEY", "")
    
    # Generate toy data
    print("=" * 60)
    print("Step 1: Generate Asia dataset")
    print("=" * 60)
    
    n_samples = 10000
    batch_size = 2048
    datamodule = BnLearnDataModule(seed=42,
                                   name='asia', 
                                   root='data/asia_causally_reliable_no_task_input',
                                   batch_size=batch_size,
                                   val_size=0.1,
                                   test_size=0.2,
                                   concept_subset=SUPERVISED_NAMES,
                                   label_descriptions=ASIA_LABEL_DESCRIPTIONS)

    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY in .env before running this example.")
    backend = RateLimitedBackend(
        LiteLLMBackend(
            model=LLM_MODEL,
            api_key=api_key,
        )
    )


    # GROUND - TRUTH GRAPH
    datamodule.precompute_graph(GraphGeneratorFixed(
        name="ground_truth",
        source="GroundTruth",
    ))
    graph_gt = datamodule.graph
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    graph_gt.plot(PLOTS_DIR / "ground_truth", title="Ground Truth")

    # LLM
    datamodule.precompute_graph(GraphGeneratorFixed(
        name=LLM_MODEL,
        source="LLM",
        api_key=api_key,
        domain="medical diagnosis",  # it uses the original concept descriptions
        use_rag=False,
        llm_backend=backend,
        refinement=dfs_remove_cycles,
    ))
    graph_llm = datamodule.graph
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    graph_llm.plot(PLOTS_DIR / "llm", title="LLM")

    # PC + LLM
    datamodule.precompute_graph(GraphGeneratorFixed(
        name="pc",
        source="Causallearn",
        refinement=compose_refinements(
            refine_llm(
                llm_backend=backend,
                domain="medical diagnosis",
                concept_descriptions=REFINEMENT_LABEL_DESCRIPTIONS, # it only refines the specified concepts; for the others it uses the original descriptions
                repeats=1,
            ),
            dfs_remove_cycles,
        ),
    
    ))
    graph_pc_llm = datamodule.graph
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    graph_pc_llm.plot(PLOTS_DIR / "pc_llm", title="PC + LLM")

    # GES + LLM
    datamodule.precompute_graph(GraphGeneratorFixed(
        name="ges",
        source="Causallearn",
        refinement=compose_refinements(
            refine_llm(
                llm_backend=backend,
                domain="medical diagnosis",
                concept_descriptions=REFINEMENT_LABEL_DESCRIPTIONS,
                repeats=1,
            ),
            dfs_remove_cycles,
        ),
    ))
    graph_ges_llm = datamodule.graph
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    graph_ges_llm.plot(PLOTS_DIR / "ges_llm", title="GES + LLM")

    annotations = datamodule.annotations
    concept_names = CONCEPT_NAMES
    task_names = TASK_NAMES
    query_names = concept_names + task_names

    n_features = datamodule.n_features[-1]
    n_concepts = len(concept_names)
    n_tasks = len(task_names)

    print(f"Input features: {n_features}")
    print(f"Concepts: {n_concepts} - {concept_names}")
    print(f"Tasks: {n_tasks} - {task_names}")
    print(f"Training samples: {datamodule.n_samples}")

    # Init model
    print("\n" + "=" * 60)
    print("Step 2: Initialize CausallyReliableConceptBottleneckModel")
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


    # Initialize the CBM
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


    # Test forward pass
    print("\n" + "=" * 60)
    print("Step 3: Test forward pass")
    print("=" * 60)
    
    x_batch = datamodule.input_data[:batch_size]
    
    # Forward pass
    print(f"Query variables: {query_names}")
    
    device = next(model.parameters()).device
    with torch.no_grad():
        concepts = model(input=x_batch.to(device), query=query_names)

    print(f"Input shape: {x_batch.shape}")
    print(f"Output logits shape: {concepts.logits.shape}")
    print(f"Expected output dim: {n_concepts + n_tasks}")


    # Test lightning training
    print("\n" + "=" * 60)
    print("Step 4: Training loop with lightning")
    print("=" * 60)

    trainer = Trainer(max_epochs=200)

    model.train()
    trainer.fit(model, datamodule=datamodule)
    print("Training completed successfully!")

    # Evaluate
    print("\n" + "=" * 60)
    print("Step 5: Test with internally-tracked metrics")
    print("=" * 60)
    trainer.test(datamodule=datamodule)

    print("\n" + "=" * 60)
    print("Step 6: Test with a different set of metrics")
    print("=" * 60)
    eval_metrics = ConceptMetrics(
        annotations=annotations,
        summary=True,
        per_concept=True,
        binary={
            'accuracy': torchmetrics.classification.BinaryAccuracy()
        }
    )
    model.eval()
    datamodule.setup('test')

    test_idxs = datamodule.testset.indices
    x_test = datamodule.input_data[test_idxs]
    c_test = datamodule.concepts[test_idxs]
    
    with torch.no_grad():
        out = model(input=x_test, query=query_names)

    eval_metrics.update(out.logits, c_test.int())
    print(f"Evaluation results with custom metrics: {eval_metrics.compute()}")


if __name__ == "__main__":
    main()



