"""
Example: ConceptBottleneckModel with custom `variable_distributions`

This mirrors example 5 (manual PyTorch training of a CBM on the all-binary
``asia`` dataset) but, instead of letting the model fill in default
distributions, passes an explicit ``variable_distributions`` dict to the
constructor: a ``{concept_name: distribution_class}`` mapping that overrides the
distribution used for each concept (here ``RelaxedBernoulli`` for every binary
concept). The dict must cover every concept on the annotation axis.
"""

import torch
from torch import nn
from torch.distributions import RelaxedBernoulli

from torch_concepts import seed_everything
from torch_concepts.nn import ConceptBottleneckModel, MLP
from torch_concepts.data import BnLearnDataset

from torchmetrics.classification import BinaryAccuracy


def main():

    seed_everything(42)

    # ------------------------------------------------------------------
    # Step 1: dataset (asia is all-binary)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Initialize the dataset")
    print("=" * 60)

    dataset = BnLearnDataset(name="asia", n_gen=2000, seed=42)
    annotations = dataset.annotations
    n_features = dataset.n_features[-1]

    task_names = ["dysp"]
    concept_names = [n for n in dataset.concept_names if n not in task_names]
    query = concept_names + task_names

    x_train = dataset.input_data
    c_train = dataset.concepts[:, :len(concept_names)]
    y_train = dataset.concepts[:, len(concept_names):]

    # ------------------------------------------------------------------
    # Step 2: model with an explicit per-concept distribution dict
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Initialize ConceptBottleneckModel")
    print("=" * 60)

    # Override the model's per-type distribution policy: model every binary
    # concept with RelaxedBernoulli.
    variable_distributions = {'binary': RelaxedBernoulli}
    print(f"variable_distributions (per-type override): {variable_distributions}")

    model = ConceptBottleneckModel(
        input_size=n_features,
        annotations=annotations,
        task_names=task_names,
        variable_distributions=variable_distributions,
        backbone=MLP(input_size=n_features, hidden_size=16, n_layers=1),
        latent_size=16,  # output size of the backbone
    )
    print(f"Model: {type(model).__name__} | latent_size: {model.latent_size}")
    print(f"binary distribution (model-owned): {model.variable_distributions['binary'].__name__}")
    print(f"dysp distribution from pgm variable: {model.pgm.variables['tasks'].distribution.__name__}")

    # ------------------------------------------------------------------
    # Step 3: training loop with torch loss
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Training loop")
    print("=" * 60)

    n_epochs = 500
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    target = torch.cat([c_train, y_train], dim=1).float()

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Forward pass - query all variables (concepts + tasks).
        out = model(query=query, input=x_train)
        logits = torch.cat([out.params[name]["logits"] for name in query], dim=1)

        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}: loss {loss:.4f}")

    # ------------------------------------------------------------------
    # Step 4: evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Evaluation")
    print("=" * 60)

    concept_acc_fn = BinaryAccuracy()
    task_acc_fn = BinaryAccuracy()

    model.eval()
    with torch.no_grad():
        out = model(query=query, input=x_train)
        c_pred = torch.cat([out.params[name]["logits"] for name in concept_names], dim=1)
        y_pred = torch.cat([out.params[name]["logits"] for name in task_names], dim=1)

        print(f"Concept accuracy: {concept_acc_fn(c_pred, c_train.int()).item():.4f}")
        print(f"Task accuracy:    {task_acc_fn(y_pred, y_train.int()).item():.4f}")


if __name__ == "__main__":
    main()
