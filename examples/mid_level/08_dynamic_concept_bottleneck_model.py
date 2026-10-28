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

The model is a dynamic Bayesian network (DBN): the CPDs of step 0, and the
CPDs of a step ``t >= 1`` given step ``t - 1``, the same at every step (a
two-slice temporal Bayes net, 2TBN). ``DynamicInference`` unrolls it over the
horizon, one copy of each variable per step (``health_0``, ``health_1``, ...),
and runs a static engine on that network: a forward engine to train and
forecast, or an engine that conditions on the evidence, which a query about
an earlier step given a later one needs. The last section runs the model one
step at a time instead, by hand.

Data: ``WearDataset``, a fleet of machines with two components whose health,
0 when new, drops at every step; the alarm fires when the mean health drops
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
# One machine per sample: a window of ``W`` steps of ``D`` channels, the two
# operating conditions and then the sensor readings, and the health of its two
# components and the alarm at the window's last step and at the ``H`` steps
# after it, ``(n, H + 1, 3)``.
dataset = WearDataset(n_gen=1500, root=str(DATA_ROOT / "wear"))
W, H, D = dataset.window, dataset.horizon, dataset.input_data.shape[-1]
x = dataset.input_data
c = dataset.concepts[["health_A", "health_B"]]
y = dataset.concepts[["alarm"]]
x_train, x_test = x[:1000], x[1000:]
c_train, c_test = c[:1000], c[1000:]
y_train, y_test = y[:1000], y[1000:]

# %%
# Variables
# ---------
# Every variable once, whatever the steps at which it exists.
sensors = EmbeddingVariable("sensors", distribution=Delta, shape=(W, D))
health = ConceptVariable(
    "health", members=["health_A", "health_B"], distribution=Normal
)
alarm = ConceptVariable("alarm", distribution=Bernoulli)

# %%
# Model
# -----
# ``initial`` holds the CPDs of step 0 and ``transition`` those of a step
# ``t >= 1``, where ``health.previous()`` is the health at step ``t - 1`` and
# every other parent is at step ``t``. The health is read at the step before,
# so it is the *interface*: what each step passes to the next. The alarm CPD
# is one object in both lists, so every step uses the same alarm head.
# ``GlobalScale`` with ``learnable=False`` keeps ``sigma`` fixed. The window is
# always observed, so its prior is a placeholder.
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
# A name in a query covers every step at which the variable exists, and its
# tensor stacks those steps before the event: ``c_train`` is ``(n, H + 1, 2)``,
# and so is the predicted health. The sensors exist at step 0 only, so their
# tensor has no time axis. With ``p_int=1`` the true health replaces the
# prediction before the children of the health read it, and the query still
# reports the prediction (teacher forcing).
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
health_loss_fn = torch.nn.MSELoss()
alarm_loss_fn = torch.nn.BCEWithLogitsLoss()
for epoch in range(1000):
    optimizer.zero_grad()
    out = engine.query(
        {"health": c_train, "alarm": None}, evidence={"sensors": x_train}, horizon=H
    )
    health_loss = health_loss_fn(out.loc["health"], c_train)
    alarm_loss = alarm_loss_fn(out.logits["alarm"], y_train)
    loss = health_loss + alarm_loss
    loss.backward()
    optimizer.step()

# %%
# Evaluation
# ----------
# The forecast from the window alone. The error grows with the horizon: the
# transition reads only the health of the step before, and each machine also
# has a load, which the sensors show but no concept records.
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
# its noise. ``RejectionSampling`` samples the health: for each window it draws
# 2000 trajectories, keeps those that match the evidence, and returns the
# fraction of them with an alarm at ``t = 4``. Since the health is noisy,
# observing the alarm at the last step changes the belief about an earlier
# step. Such a *smoothing* query needs an engine that conditions on the
# evidence. The key ``("alarm", H)`` is the alarm at step ``H`` only. The
# windows are those whose forecast leaves that alarm uncertain.
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

# %%
# Step by step, by hand
# ---------------------
# As a ``BayesianNetwork``, ``model`` is the network of steps 0 and 1, with
# names per step (``health_0``, ``health_1``). Given the health at step 0, a
# forward engine stops there, so a query of step 1 runs one transition from
# that health: the model runs one step at a time, with no horizon fixed in
# advance, and the carried health can be detached to truncate the gradient.
# The engines take plain tensors as evidence, so ``.tensor`` drops the labels
# of the carried health. The forecast is the one of ``engine`` above.
step = DeterministicInference(model)
with torch.no_grad():
    out_0 = step.query(["health_0"], evidence={"sensors_0": x_test})
    health = [out_0.loc["health_0"].tensor]
    for t in range(1, H + 1):
        out_t = step.query(["health_1"], evidence={"health_0": health[-1]})
        health.append(out_t.loc["health_1"].tensor)
same = torch.allclose(torch.stack(health, 1), out.loc["health"].tensor)
print(f"\nstep by step, by hand: same forecast as unrolled: {same}")
