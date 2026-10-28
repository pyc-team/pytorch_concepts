"""
Dynamic Concept Bottleneck Model with a Residual
================================================

A concept bottleneck model whose concepts are a *state* that moves forward in
time. A window of sensors gives the concepts at its last step and a hidden
state; then every step moves the concepts with a *mechanism* ``f``, fitted once
from the data, plus a learned *residual* ``r``, and an alarm head reads the
alarm from the concepts::

    c_0 = E_c(x),   h_0 = E_h(x)
    c_t = f(c_{t-1}) + r_t,   r_t = G([c_{t-1}, h_{t-1}])
    h_t = A [c_{t-1}, c_t, h_{t-1}] + d,   alarm_t ~ Bernoulli(sigmoid(g(c_t)))

with ``c`` the health, ``h`` the hidden state, ``E_c`` and ``E_h`` the two
encoders, ``G`` the MLP of the residual, ``A`` and ``d`` the hidden update
and ``g`` the alarm head. The hidden state carries what the concepts miss,
and acts only through ``r``: ``do(r = 0)`` gives back the mechanism alone,
and every forecast and every alarm passes through the concepts. Every
equation is a CPD of a dynamic Bayesian network, all of them deterministic
but the alarm's; ``08_dynamic_concept_bottleneck_model.py`` introduces the
classes.

Data: ``WearDataset``, as in ``08_dynamic_concept_bottleneck_model.py``. A worn
component wears faster, which the mechanism models; each machine also has a
load that the sensors show but no concept records, which the hidden state has
to estimate. The readings depend on the health through operating conditions
that drift over time, so both encoders are small recurrent networks (GRU).

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
# As in ``08_dynamic_concept_bottleneck_model.py``.
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
# All deterministic (``Delta``) but the alarm. The mechanism ``f`` and the
# residual ``r`` have the width of the health, one entry per component.
sensors = EmbeddingVariable("sensors", distribution=Delta, shape=(W, D))
health = ConceptVariable("health", members=["health_A", "health_B"], distribution=Delta)
hidden = EmbeddingVariable("hidden", distribution=Delta, size=8)
alarm = ConceptVariable("alarm", distribution=Bernoulli)
f = ConceptVariable("f", distribution=Delta, size=2)
r = ConceptVariable("r", distribution=Delta, size=2)

# %%
# Model
# -----
# The mechanism lowers the health at every step by ``a * relu(-c) + b``: a
# worn component wears faster. Its rates are fitted once per component, by
# least squares on one transition of every training machine, and then stay
# fixed; both come out positive, as wear should be. The fit does not observe
# the load, so it finds the average law of the fleet; the data follows
# ``a = dataset.alpha`` and ``b = 0``, plus the load of each machine.
#
# In a transition CPD, ``health.previous()`` and ``hidden.previous()`` are the
# health and the hidden state at step ``t - 1``: they form the *interface*,
# what each step passes to the next. ``f`` and ``r`` have a CPD in
# ``transition`` only, so they exist from step 1 on, and the health of those
# steps is their sum. The alarm CPD is one object in both lists.
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
# Two queries per epoch. The first is teacher forcing: with
# ``p_int=1`` the true health replaces the prediction at every step before its
# children read it, and the query still reports the prediction, so the concept
# encoder fits step 0 and the alarm head reads true concepts. From two
# consecutive true healths, however, the hidden update can read how much a
# step wore the machine, and so its load, which a forecast never has. The
# second query gives the true health at step 0 only, with the key
# ``("health", 0)``, and the hidden pathway learns from its own forecasts, as
# at test time.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
health_loss_fn = torch.nn.MSELoss()
alarm_loss_fn = torch.nn.BCEWithLogitsLoss()
for epoch in range(600):
    optimizer.zero_grad()
    teacher = engine.query(
        {"health": c_train, "alarm": None}, evidence={"sensors": x_train}, horizon=H
    )
    free = engine.query(
        {"health": None, ("health", 0): c_train[:, 0]},
        evidence={"sensors": x_train},
        horizon=H,
    )
    loss = (
        health_loss_fn(teacher.value["health"], c_train)
        + alarm_loss_fn(teacher.logits["alarm"], y_train)
        + health_loss_fn(free.value["health"], c_train)
    )
    loss.backward()
    optimizer.step()

# %%
# Evaluation
# ----------
# Three forecasts of the test machines: the full model; the mechanism alone,
# with the residual clamped to zero at every step (``r`` exists at steps 1
# to ``H``, so its evidence stacks ``H`` steps); and the full model with the
# true health given at ``t = 4``, from which the forecast and the alarm
# continue (at ``t = 4`` the query still reports the prediction). The
# residual adds each machine's load, which the fitted mechanism does not know.
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
