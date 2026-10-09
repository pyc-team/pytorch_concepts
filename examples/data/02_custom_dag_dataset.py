"""
A Custom Dataset from a Causal Graph
====================================

``ToyDAGDataset`` samples a dataset from a Bayesian network you write: a list
of variables with their number of states, the edges of a graph, and a
probability table for every variable. The concepts are the sampled states;
the inputs are embeddings of them, learned by an autoencoder.

Here a car starts depending on its battery (dead, low or full: a categorical
concept) and on whether the engine works (a binary concept)::

    battery ----+
                +--> starts
    engine_ok --+

Binary variables list their states as [no, yes], so that 1 means yes.
"""

# %%
from torch_concepts import seed_everything
from torch_concepts.data import ToyDAGDataModule, ToyDAGDataset
from torch_concepts.env import DATA_ROOT

seed_everything(42)

# %%
# The network
# -----------
# A root's table gives the probability of each state; a child's table gives,
# for every combination of its parents' states, the probability of each of its
# own states.
network = dict(
    variables=["battery", "engine_ok", "starts"],
    cardinalities={
        "battery": 3,  # dead, low, full
        "engine_ok": 2,  # no, yes
        "starts": 2,  # no, yes
    },
    dag=[("battery", "starts"), ("engine_ok", "starts")],
    conditional_probs={
        "battery": [0.1, 0.2, 0.7],
        "engine_ok": [0.1, 0.9],
        "starts": {
            "battery=0,engine_ok=0": [1.00, 0.00],
            "battery=0,engine_ok=1": [0.99, 0.01],
            "battery=1,engine_ok=0": [0.99, 0.01],
            "battery=1,engine_ok=1": [0.40, 0.60],
            "battery=2,engine_ok=0": [0.98, 0.02],
            "battery=2,engine_ok=1": [0.05, 0.95],
        },
    },
)

# %%
# The dataset
# -----------
# Cached under ``root``, keyed by ``n_gen`` and ``seed`` only: after editing
# the tables, use a new ``root`` (or delete the old one).
dataset = ToyDAGDataset(**network, root=str(DATA_ROOT / "car_starts"), n_gen=5000)
print(f"inputs:        {tuple(dataset.input_data.shape)}")
print(f"concepts:      {dataset.concept_names}")
print(f"cardinalities: {dataset.annotations.cardinalities}")

concepts = dataset.concepts
working = concepts["engine_ok"].flatten() == 1
for state, battery in enumerate(["dead", "low", "full"]):
    rows = working & (concepts["battery"].flatten() == state)
    p_starts = concepts["starts"][rows].float().mean()
    print(f"P(starts | battery {battery}, engine ok) = {p_starts:.2f}")

# %%
# The datamodule
# --------------
# The same arguments give a datamodule with train/validation/test splits.
datamodule = ToyDAGDataModule(
    **network,
    root=str(DATA_ROOT / "car_starts"),
    n_gen=5000,
    batch_size=128,
)
datamodule.setup()
batch = next(iter(datamodule.train_dataloader()))
x, c = batch["inputs"]["x"], batch["concepts"]["c"]
print(f"batch inputs:   {tuple(x.shape)}")
print(f"batch concepts: {tuple(c.shape)}")
