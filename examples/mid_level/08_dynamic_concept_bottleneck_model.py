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
temporal Bayes net, 2TBN). ``DynamicInference`` unrolls the DBN over the
horizon, with one copy of each variable per step (``health_0``,
``health_1``, ...), and runs a static engine on the unrolled network: a
forward engine to train and forecast, and ``RejectionSampling`` to condition
on a later step.

Data: ``WearDataset``, a fleet of machines with two components whose health is
0 when new and drops at every step; the alarm fires when the mean health drops
below a threshold. As in N-CMAPSS, the readings depend on the health through
operating conditions that drift over time.

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
from torch_concepts.distributions import Delta
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import (
    DeterministicInference,
    DynamicBayesianNetwork,
    DynamicInference,
    FixedPrior,
    GlobalScale,
    LinearConceptToConcept,
    MLPConceptToConcept,
    MLPEmbeddingToConcept,
    ParametricCPD,
    RejectionSampling,
    Sequential,
)

seed_everything(42)

# %%
# Data
# ----
# Each machine runs for 30 steps, enough for almost all of them to reach the
# alarm. A sample is a window of ``W + H`` steps of one machine: ``D``
# channels, the health of its two components and the alarm. A window starts
# every 2 steps, so the samples cover every stage of wear. The model reads the
# first ``W`` steps, ``x``, and forecasts the health ``c`` and the alarm ``y``
# from the last of them to ``H`` steps after it. The windows of a machine
# overlap, so the split is by machine: ``dataset.series`` is the machine of
# each window, and the last 500 machines are only for testing.
W, H = 8, 12
dataset = WearDataset(
    n_gen=1500, n_steps=30, window=W + H, stride=2, root=str(DATA_ROOT / "wear")
)
x = dataset.input_data[:, :W]
c = dataset.concepts[:, W - 1:][["health_A", "health_B"]]
y = dataset.concepts[:, W - 1:][["alarm"]]
train, test = dataset.series < 1000, dataset.series >= 1000
x_train, x_test = x[train], x[test]
c_train, c_test = c[train], c[test]
y_train, y_test = y[train], y[test]
D = dataset.input_data.shape[-1]

# %%
# Variables
# ---------
# Each variable is declared once; the DBN makes one copy per step.
sensors = EmbeddingVariable("sensors", distribution=Delta, shape=(W, D))
health = ConceptVariable(
    "health", members=["health_A", "health_B"], distribution=Normal
)
alarm = ConceptVariable("alarm", distribution=Bernoulli)

# %%
# Model
# -----
# ``initial`` holds the CPDs of step 0 and ``transition`` those of every later
# step. In a transition CPD, ``health.previous()`` is the health at step
# ``t - 1``; the other parents are at step ``t``. The alarm CPD is the same
# object in both lists, so every step uses one alarm head. ``GlobalScale`` with
# ``learnable=False`` keeps ``sigma`` fixed, and the window is always observed,
# so its prior is a placeholder.
encoder = torch.nn.Sequential(
    torch.nn.Flatten(-2),
    MLPEmbeddingToConcept(
        in_embeddings=W * D, out_concepts=2, hidden_size=64, activation="gelu"
    ),
)
noise = Sequential(GlobalScale(2, init=0.1, learnable=False), torch.nn.Softplus())
alarm_head = LinearConceptToConcept(in_concepts=2, out_concepts=1)
alarm_cpd = ParametricCPD(
    alarm, parents=[health], parametrization={"logits": alarm_head}
)
dynamics = MLPConceptToConcept(
    in_concepts=2, out_concepts=2, hidden_size=32, activation="gelu"
)
model = DynamicBayesianNetwork(
    variables=[sensors, health, alarm],
    initial=[
        ParametricCPD(sensors, parametrization=FixedPrior(torch.zeros(W, D))),
        ParametricCPD(
            health, parents=[sensors], parametrization={"loc": encoder, "scale": noise}
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
engine = DynamicInference(model, DeterministicInference, p_int=1.0)

# %%
# Training
# --------
# A name in the query covers all the steps of its variable: ``c`` is
# ``(batch, H + 1, 2)``, and so is the predicted health. The sensors exist only
# at step 0, so their tensor has no time axis. With ``p_int=1`` the children of
# the health read the true health, while the query reports the prediction
# (teacher forcing).
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
health_loss_fn = torch.nn.MSELoss()
alarm_loss_fn = torch.nn.BCEWithLogitsLoss()
for epoch in range(25):
    for batch in torch.randperm(len(x_train)).split(100):  # shuffled minibatches
        x, c, y = x_train[batch], c_train[batch], y_train[batch]
        optimizer.zero_grad()
        out = engine.query({"health": c, "alarm": None}, evidence={"sensors": x}, horizon=H)
        health_loss = health_loss_fn(out.loc["health"], c)
        alarm_loss = alarm_loss_fn(out.logits["alarm"], y)
        loss = health_loss + alarm_loss
        loss.backward()
        optimizer.step()

# %%
# Evaluation
# ----------
# The forecast from the sensors alone, on the test windows. The error grows
# with the horizon: the transition reads only the health of the step before,
# and each machine has a load that no concept records.
with torch.no_grad():
    out = engine.query(["health", "alarm"], evidence={"sensors": x_test}, horizon=H)
rmse = ((out.loc["health"] - c_test) ** 2).mean((0, 2)).sqrt()
accuracy = ((out.logits["alarm"] > 0).float() == y_test).float().mean((0, 2))
print(f"health RMSE at t = 0, 4, 8, 12:     {rmse[::4].numpy().round(3)}")
print(f"alarm accuracy at t = 0, 4, 8, 12:  {accuracy[::4].numpy().round(3)}")

# %%
# Evidence at a later step
# ------------------------
# ``DeterministicInference`` propagates the ``loc`` of the health and ignores
# its noise. ``RejectionSampling`` draws 2000 trajectories per window and keeps
# those that match the evidence. The health is noisy, so observing the alarm at
# step ``H`` changes the belief about the alarm at step 4 (a *smoothing*
# query). The key ``("alarm", H)`` is the alarm at step ``H`` only; the windows
# are those whose forecast leaves that alarm uncertain.
p_last = torch.sigmoid(out.logits["alarm"][:, H]).flatten()
x_open = x_test[(p_last > 0.2) & (p_last < 0.8)]
ones, zeros = torch.ones(len(x_open), 1), torch.zeros(len(x_open), 1)
sampler = DynamicInference(model, RejectionSampling, n_samples=2000)
evidence = {
    "P(alarm_4 = 1 | x)": {"sensors": x_open},
    "P(alarm_4 = 1 | x, alarm_H = 1)": {"sensors": x_open, ("alarm", H): ones},
    "P(alarm_4 = 1 | x, alarm_H = 0)": {"sensors": x_open, ("alarm", H): zeros},
}
print(f"\nmean over the {len(x_open)} test windows with 0.2 < P(alarm_H = 1) < 0.8:")
for label, given in evidence.items():
    p = sampler.query({("alarm", 4): ones}, evidence=given, horizon=H).probabilities
    print(f"  {label:32s} {p.mean():.3f}")
