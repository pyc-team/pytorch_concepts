"""
Dynamic Concept Bottleneck Model
================================

A concept bottleneck model over time. An encoder ``E`` reads the health of a
machine at the last step of a window of sensor readings, a learned transition
``f`` moves the health forward one step at a time, and an alarm head ``g``
reads the alarm from the health at every step::

    health_0 ~ Normal(E(x), sigma)
    health_t ~ Normal(f(health_{t-1}), sigma),   t = 1, ..., H
    alarm_t  ~ Bernoulli(sigmoid(g(health_t))),  t = 0, ..., H

The model is a dynamic Bayesian network (DBN): the CPDs of step 0, plus the
CPDs of a step given the step before, the same at every step (a two-slice
temporal Bayes net, 2TBN). ``UnrolledInference`` unrolls the DBN over the
horizon, with one copy of each variable per step (``health_0``,
``health_1``, ...), and runs a static engine on the unrolled network: a
forward engine to train and forecast, and ``RejectionSampling`` to condition
on a previous step.

Data: ``WearDataset``, a fleet of machines with two components whose health is
0 when new and drops at every step; the alarm fires when the mean health drops
below a threshold. The readings depend on the health through operating
conditions that drift over time. The design follows N-CMAPSS.

References: Murphy, "Dynamic Bayesian Networks: Representation, Inference and
Learning", PhD thesis, UC Berkeley, 2002; Koh et al., "Concept Bottleneck
Models", ICML 2020; Arias Chao et al., "Aircraft Engine Run-to-Failure Dataset
under Real Flight Conditions for Prognostics and Diagnostics", Data 2021.
"""

# %%
import torch
from torch.distributions import Bernoulli, Normal

from torch_concepts import ConceptVariable, EmbeddingVariable, seed_everything
from torch_concepts.data import WearDataset
from torch_concepts.data.splitters import InductiveSplitter
from torch_concepts.distributions import Delta
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import (
    ConceptLoss,
    DeterministicInference,
    DynamicBayesianNetwork,
    FixedPrior,
    GlobalScale,
    LinearConceptToConcept,
    MLPConceptToConcept,
    MLPEmbeddingToConcept,
    ParametricCPD,
    RejectionSampling,
    Sequential,
    UnrolledInference,
)

seed_everything(42)

# %%
# Data
# ----
# Each machine runs for 30 steps, enough for almost all of them to reach the
# alarm. A sample is a window of ``W + H`` steps of one machine: ``D``
# channels, the health of its two components and the alarm. A window starts
# at every step, so the samples cover every stage of wear. The windows of a
# machine overlap, so the split is inductive, by machine, and 500 machines are
# only for testing.
W, H, D = 8, 12, 16  # steps read, steps forecast, channels per step
dataset = WearDataset(
    n_gen=1500,
    n_steps=30,
    n_sensors=D - 2,  # besides the two operating conditions
    window=W + H,
    stride=1,
    root=str(DATA_ROOT / "wear"),
)
x = dataset.input_data[:, :W]  # (N, W, D)
c = dataset.concepts[:, W - 1:][["health_A", "health_B"]]  # (N, H + 1, 2)
y = dataset.concepts[:, W - 1:][["alarm"]]  # (N, H + 1, 1)
splitter = InductiveSplitter(val_size=0, test_size=500, seed=42)
splitter.fit(dataset)
train, test = splitter.train_idxs, splitter.test_idxs
x_train, x_test = x[train], x[test]
c_train, c_test = c[train], c[test]
y_train, y_test = y[train], y[test]

# %%
# Model
# -----
# The model reads the first ``W`` steps, ``x``, and forecasts the health ``c``
# and the alarm ``y`` from the last of them to ``H`` steps after it.
#
# Each variable is declared once; the DBN makes one copy per step. ``initial``
# holds the CPDs of step 0 and ``transition`` those of every later step. In a
# transition CPD, ``health.previous()`` is the health at step ``t - 1``; the
# other parents are at step ``t``. The alarm CPD is the same object in both
# lists, so every step uses one alarm head. ``GlobalScale`` with
# ``learnable=False`` keeps ``sigma`` fixed, and the window is always observed,
# so its prior is a placeholder.
sensors = EmbeddingVariable("sensors", distribution=Delta, shape=(W, D))
health = ConceptVariable(
    "health",
    members=["health_A", "health_B"],
    distribution=Normal,
)
alarm = ConceptVariable("alarm", distribution=Bernoulli)

encoder = torch.nn.Sequential(
    torch.nn.Flatten(-2),
    MLPEmbeddingToConcept(
        in_embeddings=W * D,
        out_concepts=2,
        hidden_size=64,
        activation="gelu",
    ),
)
noise = Sequential(GlobalScale(2, init=0.1, learnable=False), torch.nn.Softplus())
alarm_head = LinearConceptToConcept(in_concepts=2, out_concepts=1)
alarm_cpd = ParametricCPD(
    alarm,
    parents=[health],
    parametrization={"logits": alarm_head},
)
dynamics = MLPConceptToConcept(
    in_concepts=2,
    out_concepts=2,
    hidden_size=32,
    activation="gelu",
)

model = DynamicBayesianNetwork(
    variables=[sensors, health, alarm],
    initial=[
        ParametricCPD(sensors, parametrization=FixedPrior(torch.zeros(W, D))),
        ParametricCPD(
            health,
            parents=[sensors],
            parametrization={"loc": encoder, "scale": noise},
        ),
        alarm_cpd,
    ],
    transition=[
        ParametricCPD(
            health,
            parents=[health.previous()],
            parametrization={"loc": dynamics, "scale": noise},
        ),
        alarm_cpd,
    ],
)
engine = UnrolledInference(model, DeterministicInference, p_int=1.0)

# %%
# Training
# --------
# A name in the query covers all the steps of its variable: ``c`` is
# ``(batch, H + 1, 2)``, and so is the predicted health. The sensors exist only
# at step 0, so their tensor has no time axis. With ``p_int=1`` the children of
# the health read the true health, while the query reports the prediction
# (teacher forcing).
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = ConceptLoss(
    binary=torch.nn.BCEWithLogitsLoss(),
    continuous=torch.nn.MSELoss(),
)
for epoch in range(25):
    for batch in torch.randperm(len(x_train)).split(100):  # shuffled minibatches
        x, c, y = x_train[batch], c_train[batch], y_train[batch]
        optimizer.zero_grad()
        out = engine.query(
            {"health": c, "alarm": None},
            evidence={"sensors": x},
            horizon=H,
        )
        loss = loss_fn(out, c) + loss_fn(out, y)
        loss.backward()
        optimizer.step()

# %%
# Evaluation
# ----------
# The forecast from the sensors alone, on the test windows. The error grows
# with the horizon, since the transition reads only the health of the step
# before and the load of each machine is not one of the concepts. The second
# row is an intervention on the health, with the true health given at ``t = 4``
# by the key ``health_4``, from which the forecast and the alarm go on (the
# row still reports the prediction at ``t = 4``).
queries = {  # label: query
    "sensors only": {"health": None, "alarm": None},
    "true health at t = 4": {"health": None, "health_4": c_test[:, 4], "alarm": None},
}
print(f"{'':22s} health RMSE at t = 0, 4, 8, 12   alarm accuracy at t = 0, 4, 8, 12")
with torch.no_grad():
    for label, query in queries.items():
        out = engine.query(query, evidence={"sensors": x_test}, horizon=H)
        rmse = ((out.loc["health"] - c_test) ** 2).mean((0, 2)).sqrt()[::4]
        correct = (out.logits["alarm"] > 0).float() == y_test
        accuracy = correct.float().mean((0, 2))[::4]
        print(f"{label:22s} {rmse.numpy().round(3)}   {accuracy.numpy().round(3)}")

# %%
# Evidence at a previous step
# ---------------------------
# Both engines get the alarm at step 4 as evidence and return the probability
# of the alarm at step ``H``. ``DeterministicInference`` propagates the ``loc``
# of the health and ignores its noise. As a forward engine it reads evidence
# as an intervention, which reaches only the descendants of the alarm, and the
# alarm has none. ``RejectionSampling`` draws 2000 trajectories per window and
# keeps those that match the evidence. The health is noisy, so observing the
# alarm at step 4 changes the belief about the alarm at step ``H``. The key
# ``alarm_4`` is the alarm at step 4 only; the windows are those whose
# forecast leaves the alarm at step ``H`` uncertain.
with torch.no_grad():
    forecast = engine.query([f"alarm_{H}"], evidence={"sensors": x_test}, horizon=H)
p_last = torch.sigmoid(forecast.logits["alarm"]).flatten()
x_open = x_test[(p_last > 0.2) & (p_last < 0.8)]
ones, zeros = torch.ones(len(x_open), 1), torch.zeros(len(x_open), 1)
sampler = UnrolledInference(model, RejectionSampling, n_samples=2000)
evidence = {  # label: evidence besides the sensors
    "none": {},
    "alarm_4 = 1": {"alarm_4": ones},
    "alarm_4 = 0": {"alarm_4": zeros},
}
print(f"\nP(alarm_{H} = 1), mean over the {len(x_open)} uncertain test windows")
print(f"{'evidence':14s} {'DeterministicInference':>24s} {'RejectionSampling':>20s}")
with torch.no_grad():
    for label, given in evidence.items():
        given = {"sensors": x_open, **given}
        logits = engine.query([f"alarm_{H}"], given, horizon=H).logits["alarm"]
        sampled = sampler.query({f"alarm_{H}": ones}, given, horizon=H).probabilities
        forward = torch.sigmoid(logits)
        print(f"{label:14s} {forward.mean():24.3f} {sampled.mean():20.3f}")
