.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle


Mid-Level API
=============

The Mid-Level API lets you describe a model as a **probabilistic graphical model**: a set of
random variables connected by factors, queried through an inference engine. It is the right
entry point if you think in terms of probabilistic or causal models.

.. warning::

   This API is still under development and interfaces might change in future releases.

.. image:: /_static/img/api_levels/mid_level.png
   :alt: Overview of the PyC Mid-Level API
   :align: center
   :width: 100%

|

A mid-level model is assembled from four building blocks:

- **Variables** — the random variables (concepts, embeddings) that make up the model.
- **Factors** — conditional distributions wiring each variable to its parents.
- **ProbabilisticModel** — the container that collects variables and factors into a graph.
- **Inference** — engines that answer queries over the model.

Each block is described below, followed by an example of how to use it. The running example
builds a Concept Bottleneck Model ``input → latent → concepts → task`` as a probabilistic model.


Variables
---------

A :class:`~torch_concepts.Variable` is a random variable in the model. Two concrete kinds are
provided:

- :class:`~torch_concepts.ConceptVariable` — an interpretable, named variable (a concept or a
  task). A single call can declare several concepts at once.
- :class:`~torch_concepts.EmbeddingVariable` — a vector-valued node (e.g. the raw input or a
  latent representation), typically given a :class:`~torch_concepts.distributions.Delta`
  distribution.

A variable is defined by its name, its ``distribution`` type, and its ``size`` (the number of
states / dimensions). Edges of the graph are declared later, on the factors.

**Example.**

.. code-block:: python

   import torch
   from torch.distributions import Bernoulli, OneHotCategorical

   from torch_concepts import EmbeddingVariable, ConceptVariable
   from torch_concepts.distributions import Delta

   input_var = EmbeddingVariable("input", distribution=Delta, size=64)
   latent_var = EmbeddingVariable("latent", distribution=Delta, size=10)
   concepts = ConceptVariable(["c1", "c2"], distribution=Bernoulli)        # two binary concepts
   task = ConceptVariable("xor", distribution=OneHotCategorical, size=2)   # one categorical task


Factors
-------

A factor turns the values of a variable's parents into the parameters of that variable's
distribution. :class:`~torch_concepts.nn.ParametricFactor` is the abstract base class; the one
you will use is :class:`~torch_concepts.nn.ParametricCPD`, a **conditional probability
distribution** ``p(variable | parents)`` parameterised by a |pyc_logo| PyC or |pytorch_logo|
PyTorch module.

Each ``ParametricCPD`` declares its ``parents``, which is exactly how the graph structure is
defined. Parent-less (root) variables use a prior such as
:class:`~torch_concepts.nn.LearnablePrior`.

**Example.**

.. code-block:: python

   from torch_concepts.nn import (
       ParametricCPD, LearnablePrior, Sequential,
       LinearEmbeddingToConcept, LinearConceptToConcept,
   )

   # Root prior over the input embedding (no parents)
   input_cpd = ParametricCPD(input_var, parametrization=LearnablePrior(64), parents=[])

   # A plain torch backbone: input -> latent
   backbone = ParametricCPD(
       latent_var,
       parametrization=torch.nn.Sequential(torch.nn.Linear(64, 10), torch.nn.LeakyReLU()),
       parents=[input_var],
   )

   # Concept encoder: latent -> concept logits
   c_encoder = ParametricCPD(
       concepts,
       parametrization={'logits': Sequential(LinearEmbeddingToConcept(in_embeddings=10, out_concepts=1))},
       parents=[latent_var],
   )

   # Task predictor: concepts -> task logits
   y_predictor = ParametricCPD(
       task,
       parametrization={'logits': Sequential(LinearConceptToConcept(in_concepts=2, out_concepts=2))},
       parents=[*concepts],
   )


ProbabilisticModel
------------------

:class:`~torch_concepts.nn.ProbabilisticModel` is the abstract base for probabilistic graphical
models. The concrete class you instantiate is :class:`~torch_concepts.nn.BayesianNetwork`, a
directed model that wires a list of variables to a list of factors (one factor per variable).

Because ``concepts`` and ``c_encoder`` each declare several entries, they are spread with ``*``
when collected.

**Example.**

.. code-block:: python

   from torch_concepts.nn import BayesianNetwork

   model = BayesianNetwork(
       variables=[input_var, latent_var, *concepts, task],
       factors=[input_cpd, backbone, *c_encoder, y_predictor],
   )


Inference
---------

An inference engine answers queries of the form *"give me these variables, given this
evidence"* via ``engine.query(query, evidence)``. |pyc_logo| PyC ships several engines:

- :class:`~torch_concepts.nn.DeterministicInference` — propagates distribution parameters in
  topological order (a standard deep-learning forward pass). Use this for training and
  point predictions.
- :class:`~torch_concepts.nn.AncestralSamplingInference` — draws a (reparameterised) sample per
  variable in topological order.
- :class:`~torch_concepts.nn.ForwardInference`, :class:`~torch_concepts.nn.IndependentInference`,
  :class:`~torch_concepts.nn.RejectionSampling`, :class:`~torch_concepts.nn.ImportanceSampling`,
  and the Pyro-backed :class:`~torch_concepts.nn.VariationalInference` provide further
  probabilistic alternatives.

``query`` accepts a **list** of variable names (predict them) or a **dict** mapping names to
observed values (clamp them as evidence, e.g. for teacher forcing during training). The result
exposes per-variable distribution parameters in ``out.params[name]``.

**Example.**

.. code-block:: python

   from torch_concepts.nn import DeterministicInference

   inference = DeterministicInference(model, activate_before_propagation=True)

   x = torch.randn(16, 64)
   out = inference.query(query=["c1", "c2", "xor"], evidence={'input': x})

   c1_logits = out.params['c1']['logits']    # (16, 1)
   xor_logits = out.params['xor']['logits']  # (16, 2)


Putting It Together
-------------------

The blocks above form a complete, trainable Concept Bottleneck Model. During training we pass
the **observed** concept/task values as the query dict so they are clamped as evidence:

.. code-block:: python

   import torch
   from torch.distributions import Bernoulli, OneHotCategorical

   from torch_concepts import seed_everything, EmbeddingVariable, ConceptVariable
   from torch_concepts.distributions import Delta
   from torch_concepts.data import ToyDataset
   from torch_concepts.nn import (
       LinearEmbeddingToConcept, LinearConceptToConcept, ParametricCPD,
       BayesianNetwork, DeterministicInference, LearnablePrior, Sequential,
   )

   seed_everything(42)

   dataset = ToyDataset(dataset='xor', seed=42, n_gen=1000)
   x_train = dataset.input_data
   concept_idx = list(dataset.graph.edge_index[0].unique().numpy())
   task_idx = list(dataset.graph.edge_index[1].unique().numpy())
   c_train = dataset.concepts[:, concept_idx]
   y_train = dataset.concepts[:, task_idx]
   y_train = torch.cat([y_train, 1 - y_train], dim=1)

   # Variables
   input_var = EmbeddingVariable("input", distribution=Delta, size=x_train.shape[1])
   latent_var = EmbeddingVariable("latent", distribution=Delta, size=10)
   concepts = ConceptVariable(["c1", "c2"], distribution=Bernoulli)
   task = ConceptVariable("xor", distribution=OneHotCategorical, size=2)

   # Factors
   input_cpd = ParametricCPD(input_var, parametrization=LearnablePrior(x_train.shape[1]), parents=[])
   backbone = ParametricCPD(
       latent_var,
       parametrization=torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], 10), torch.nn.LeakyReLU()),
       parents=[input_var],
   )
   c_encoder = ParametricCPD(
       concepts,
       parametrization={'logits': Sequential(LinearEmbeddingToConcept(in_embeddings=10, out_concepts=1))},
       parents=[latent_var],
   )
   y_predictor = ParametricCPD(
       task,
       parametrization={'logits': Sequential(LinearConceptToConcept(in_concepts=2, out_concepts=2))},
       parents=[*concepts],
   )

   # Model + inference
   model = BayesianNetwork(
       variables=[input_var, latent_var, *concepts, task],
       factors=[input_cpd, backbone, *c_encoder, y_predictor],
   )
   inference = DeterministicInference(model, activate_before_propagation=True)

   evidence = {'input': x_train}
   query = {"c1": c_train[:, 0], "c2": c_train[:, 1], "xor": y_train}

   optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
   loss_fn = torch.nn.BCEWithLogitsLoss()
   model.train()
   for epoch in range(500):
       optimizer.zero_grad()
       pred = inference.query(query=query, evidence=evidence)
       c_pred = torch.cat([pred.params['c1']['logits'], pred.params['c2']['logits']], dim=1)
       y_pred = pred.params['xor']['logits']
       loss = loss_fn(c_pred, c_train) + 0.5 * loss_fn(y_pred, y_train)
       loss.backward()
       optimizer.step()


Causal Models (SEMs)
--------------------

The same primitives express **Structural Equation Models**: each variable's factor is a
structural equation over its parents, and :class:`~torch_concepts.nn.AncestralSamplingInference`
draws a realisation of every node in topological order. Deterministic mechanisms are convenient
to write with :class:`~torch_concepts.nn.CallableConceptToConcept`.

.. code-block:: python

   import torch
   from torch.distributions import Bernoulli

   from torch_concepts import seed_everything, EmbeddingVariable, ConceptVariable
   from torch_concepts.distributions import Delta
   from torch_concepts.nn import (
       ParametricCPD, BayesianNetwork, AncestralSamplingInference,
       CallableConceptToConcept, LearnablePrior,
   )

   seed_everything(42)

   # genotype -> smoking, genotype & smoking -> tar -> cancer
   input_var = EmbeddingVariable("input", distribution=Delta, size=1)
   genotype = ConceptVariable("genotype", distribution=Bernoulli)
   smoking = ConceptVariable("smoking", distribution=Bernoulli)
   tar = ConceptVariable("tar", distribution=Bernoulli)
   cancer = ConceptVariable("cancer", distribution=Bernoulli)

   factors = [
       ParametricCPD(input_var, parametrization=LearnablePrior(1), parents=[]),
       ParametricCPD(genotype,
                     parametrization=torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Sigmoid()),
                     parents=[input_var]),
       ParametricCPD(smoking,
                     parametrization=CallableConceptToConcept(lambda g: (g > 0.5).float(), use_bias=False),
                     parents=[genotype]),
       ParametricCPD(tar,
                     parametrization=CallableConceptToConcept(
                         lambda gs: torch.logical_or(gs[:, 0] > 0.5, gs[:, 1] > 0.5).float().unsqueeze(-1),
                         use_bias=False),
                     parents=[genotype, smoking]),
       ParametricCPD(cancer,
                     parametrization=CallableConceptToConcept(lambda t: t, use_bias=False),
                     parents=[tar]),
   ]

   sem = BayesianNetwork(variables=[input_var, genotype, smoking, tar, cancer], factors=factors)
   inference = AncestralSamplingInference(sem)

   evidence = {'input': torch.randn((1000, 1))}
   results = inference.query(["genotype", "smoking", "tar", "cancer"], evidence=evidence)
   print("P(cancer=1) ≈", results.samples["cancer"].mean().item())

.. note::

   Do-interventions and causal-effect estimation (e.g. the CACE score) for the mid-level
   ``BayesianNetwork`` are on the roadmap. For interventions today, see the
   :doc:`Low-Level API <using_low_level>` ``intervention`` context manager.


Next Steps
----------

- Browse the full :doc:`Mid-Level API reference </modules/mid_level_api>`.
- Drop down to the :doc:`Low-Level API <using_low_level>` to customise the layers behind each factor.
- Move up to the :doc:`High-Level API <using_high_level>` for the same models, pre-assembled.
