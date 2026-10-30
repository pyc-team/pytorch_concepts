"""
Dynamic Concept Bottleneck Model with a Residual
================================================

A concept bottleneck model whose concepts are a *state* that moves forward in
time. A window of sensors gives the concepts at its last step and a hidden
state. Every step then moves the concepts with a *mechanism* ``f``, fitted
once from the data, plus a learned *residual* ``r``, and an alarm head reads
the alarm from the concepts::

    c_0 = E_c(x),   h_0 = E_h(x)
    c_t = f(c_{t-1}) + r_t,   r_t = MLP([c_{t-1}, h_{t-1}])
    h_t = A [c_{t-1}, c_t, h_{t-1}] + d,   alarm_t ~ Bernoulli(sigmoid(g(c_t)))

with ``c`` the health, ``h`` the hidden state, ``E_c`` and ``E_h`` the two
encoders, ``A`` and ``d`` the hidden update and ``g`` the alarm head. The
hidden state carries what the concepts miss and acts only through ``r``, so
``do(r = 0)`` gives back the mechanism alone, and every forecast and every
alarm passes through the concepts. Every equation is a CPD of a dynamic
Bayesian network; ``08_dynamic_concept_bottleneck_model.py`` introduces the
classes.

Data: ``WearDataset``, as in ``08_dynamic_concept_bottleneck_model.py``. A worn
component wears faster, which the mechanism models. Each machine also has a
load that no concept records, so the hidden state has to estimate it. It reads
the load from how fast the health drops along the window, so the window has
several steps and both encoders are small recurrent networks (GRU).

References: Murphy, "Dynamic Bayesian Networks: Representation, Inference and
Learning", PhD thesis, UC Berkeley, 2002; Arias Chao et al., "Aircraft Engine
Run-to-Failure Dataset under Real Flight Conditions for Prognostics and
Diagnostics", Data 2021 (N-CMAPSS).
"""

# %%
import torch
from torch.distributions import Bernoulli

from torch_concepts import ConceptVariable, EmbeddingVariable, seed_everything
from torch_concepts.data import WearDataset
from torch_concepts.distributions import Delta
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import (
    MLP,
    CallableConceptToConcept,
    DeterministicInference,
    DynamicBayesianNetwork,
    DynamicInference,
    FixedPrior,
    LinearConceptToConcept,
    ParametricCPD,
)

seed_everything(42)

# %%
# Data
# ----
# As in ``08_dynamic_concept_bottleneck_model.py``: windows of ``W + H`` steps,
# split by machine.
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
# All the variables are deterministic (``Delta``) but the alarm. The mechanism
# ``f`` and the residual ``r`` have one entry per component, like the health.
sensors = EmbeddingVariable("sensors", distribution=Delta, shape=(W, D))
health = ConceptVariable("health", members=["health_A", "health_B"], distribution=Delta)
hidden = EmbeddingVariable("hidden", distribution=Delta, size=8)
alarm = ConceptVariable("alarm", distribution=Bernoulli)
f = ConceptVariable("f", distribution=Delta, size=2)
r = ConceptVariable("r", distribution=Delta, size=2)

# %%
# Model
# -----
# The mechanism lowers the health at every step by ``a * relu(-c) + b``, so a
# worn component wears faster. Its rates are fitted once per component, by
# least squares on the first forecast step of every training window, and then
# stay fixed. The fit does not see the load, so it finds the average law of the
# fleet: the data follows ``a = dataset.alpha`` and ``b = 0``, plus the load of
# each machine.
#
# In a transition CPD, ``health.previous()`` and ``hidden.previous()`` are the
# health and the hidden state at step ``t - 1``. ``f`` and ``r`` have a CPD
# only in ``transition``, so they exist from step 1 on, and the health of those
# steps is their sum. The alarm CPD is the same object in both lists.
c_now, c_next = c_train[:, 0].T, c_train[:, 1].T
design = torch.stack([torch.relu(-c_now), torch.ones_like(c_now)], -1)
a, b = torch.linalg.lstsq(design, (c_now - c_next).unsqueeze(-1)).solution[..., 0].T
print(
    f"fitted a = {a.numpy().round(4)}, b = {b.numpy().round(4)}; the data's law: "
    f"a = {dataset.alpha.numpy()}, b = 0, plus each machine's load"
)


class Encoder(torch.nn.Module):
    def __init__(self, out):
        super().__init__()
        self.gru = torch.nn.GRU(D, 32, batch_first=True)
        self.head = torch.nn.Linear(32, out)

    def forward(self, window):
        return self.head(self.gru(window)[0][..., -1, :])  # the last step's state


wear = CallableConceptToConcept(lambda c: c - a * torch.relu(-c) - b, use_bias=False)
alarm_head = LinearConceptToConcept(in_concepts=2, out_concepts=1)
alarm_cpd = ParametricCPD(
    alarm, parents=[health], parametrization={"logits": alarm_head}
)
model = DynamicBayesianNetwork(
    variables=[sensors, health, hidden, alarm, f, r],
    initial=[
        ParametricCPD(sensors, parametrization=FixedPrior(torch.zeros(W, D))),
        ParametricCPD(health, parents=[sensors], parametrization=Encoder(2)),
        ParametricCPD(hidden, parents=[sensors], parametrization=Encoder(8)),
        alarm_cpd,
    ],
    transition=[
        ParametricCPD(f, parents=[health.previous()], parametrization=wear),
        ParametricCPD(
            r,
            parents=[health.previous(), hidden.previous()],
            parametrization=MLP(2 + 8, 64, 2, activation="gelu"),
        ),
        ParametricCPD(
            health,
            parents=[f, r],
            parametrization=torch.nn.Identity(),
            aggregate=lambda parents: sum(parents.values()),
        ),
        ParametricCPD(
            hidden,
            parents=[health.previous(), health, hidden.previous()],
            parametrization=torch.nn.Linear(2 + 2 + 8, 8),
        ),
        alarm_cpd,
    ],
)
engine = DynamicInference(model, DeterministicInference, p_int=1.0)

# %%
# Training
# --------
# Two queries per batch. The first uses teacher forcing, as in
# ``08_dynamic_concept_bottleneck_model.py``: the concept encoder fits step 0
# and the alarm head reads true concepts. With the true health at two
# consecutive steps, the hidden update can read the wear of each step, and so
# the load, which a forecast never has. The second query gives the true health
# at step 0 only, with the key ``("health", 0)``, so the hidden state learns
# from its own forecasts, as at test time.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
health_loss_fn = torch.nn.MSELoss()
alarm_loss_fn = torch.nn.BCEWithLogitsLoss()
for epoch in range(50):
    for batch in torch.randperm(len(x_train)).split(100):  # shuffled minibatches
        x, c, y = x_train[batch], c_train[batch], y_train[batch]
        optimizer.zero_grad()
        teacher = engine.query({"health": c, "alarm": None}, evidence={"sensors": x}, horizon=H)
        free = engine.query(
            {"health": None, ("health", 0): c[:, 0]}, evidence={"sensors": x}, horizon=H
        )
        loss = (
            health_loss_fn(teacher.value["health"], c)
            + alarm_loss_fn(teacher.logits["alarm"], y)
            + health_loss_fn(free.value["health"], c)
        )
        loss.backward()
        optimizer.step()

# %%
# Evaluation
# ----------
# Three forecasts on the test windows: the full model; the mechanism alone,
# with ``r = 0`` at every step (``r`` exists at steps 1 to ``H``, so its
# evidence has ``H`` steps); and the full model with the true health given at
# ``t = 4``, from which the forecast and the alarm go on (the row still reports
# the prediction at ``t = 4``). The residual adds the load of each machine,
# which the fitted mechanism does not know.
queries = {  # label: (query, evidence)
    "full model": ({"health": None, "alarm": None}, {"sensors": x_test}),
    "mechanism alone, do(r = 0)": (
        {"health": None, "alarm": None},
        {"sensors": x_test, "r": torch.zeros(len(x_test), H, 2)},
    ),
    "true health given at t = 4": (
        {"health": None, ("health", 4): c_test[:, 4], "alarm": None},
        {"sensors": x_test},
    ),
}
print(f"\n{'':28s} health RMSE at t = 0, 4, 8, 12   alarm accuracy at t = 0, 4, 8, 12")
with torch.no_grad():
    for label, (query, evidence) in queries.items():
        out = engine.query(query, evidence, horizon=H)
        rmse = ((out.value["health"] - c_test) ** 2).mean((0, 2)).sqrt()[::4]
        correct = (out.logits["alarm"] > 0).float() == y_test
        accuracy = correct.float().mean((0, 2))[::4]
        print(f"{label:28s} {rmse.numpy().round(3)}   {accuracy.numpy().round(3)}")
