.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle

.. |pl_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/lightning.svg
    :width: 20px
    :align: middle


Out-of-the-box Models
=====================

The High-Level API gives you concept-based models out-of-the-box. 
It is the right entry point if you just want a model that works without assembling it yourself.

.. image:: /_static/img/api_levels/high_level.png
   :alt: Overview of the PyC High-Level API
   :align: center
   :width: 100%

|

Working at this level involves three building blocks:

- **Models** — ready-to-use interpretable concept-based models.
- **Losses & Metrics** — objectives and metrics with automatic type-aware routing.
- **Training** — a manual |pyc_logo| PyTorch loop, or automatic |pl_logo| Lightning training.

All of them are configured with :class:`~torch_concepts.Annotations`, which describe your
concepts (see :doc:`Annotations <using_low_level>`).

Expand each block below for an explanation and an example.


.. dropdown:: Models
    :icon: rocket

    |pyc_logo| PyC provides ready-to-use interpretable models. Each wraps a mid-level
    probabilistic model with a backbone; more models will be added over time.

    - :class:`~torch_concepts.nn.ConceptBottleneckModel` — the standard CBM.
    - :class:`~torch_concepts.nn.ConceptEmbeddingModel` — expressive CBM with concept embeddings (CEM).
    - :class:`~torch_concepts.nn.GraphConceptBottleneckModel` — graph-structured CBM.
    - :class:`~torch_concepts.nn.CausallyReliableConceptBottleneckModel` — causally reliable CBM variant.
    - :class:`~torch_concepts.nn.BlackBox` — non-interpretable baseline for comparison.

    All models take an ``input_size``, ``annotations``, and model-specific parameters.
    A forward pass returns a :class:`~torch_concepts.nn.ModelOutput` — a structured object
    whose ``params`` dict maps each queried variable name to its distribution parameters
    (e.g. ``{'logits': ...}`` for binary/categorical, ``{'loc': ..., 'scale': ...}`` for
    Normal). A ``query`` list controls which variables are computed.

    .. code-block:: python

       import torch
       import torch_concepts as pyc
       from torch_concepts.nn import ConceptBottleneckModel, MLP

       annotations = pyc.Annotations(
           labels=["smoking", "genotype", "tar"],
           cardinalities=[1, 3, 1],
           types=["binary", "categorical", "continuous"],
       )
       n_features = 64

       model = ConceptBottleneckModel(
           input_size=n_features,
           annotations=annotations,
           task_names=['cancer'],
           backbone=MLP(input_size=n_features, hidden_size=128, n_layers=1),
           latent_size=128,
       )

       x = torch.randn(16, n_features)
       out = model(input=x, query=['smoking', 'genotype', 'tar', 'cancer'])
       smoking_logits = out.params['smoking']['logits']    # (16, 1)
       genotype_logits = out.params['genotype']['logits']  # (16, 3)
       cancer_logits  = out.params['cancer']['logits']     # (16, 1)


.. dropdown:: Losses & Metrics
    :icon: flame

    :class:`~torch_concepts.nn.ConceptLoss` and :class:`~torch_concepts.nn.ConceptMetrics` are
    **type-aware**: they automatically route each concept to the right objective/metric based on
    its type (binary, categorical, continuous).

    **Basic usage.** Pass one loss per type:

    .. code-block:: python

       import torch
       from torch_concepts.nn import ConceptLoss

       loss = ConceptLoss(
           annotations=annotations,
           binary=torch.nn.BCEWithLogitsLoss(),
           categorical=torch.nn.CrossEntropyLoss(),
           continuous=torch.nn.MSELoss(),
       )

    **Composing losses.** Pass a list of terms per type and optional per-term weights.
    Terms are summed with those weights. Here binary concepts are supervised with BCE
    and additionally regularised with an L1 penalty at weight 0.01:

    .. code-block:: python

       from torch_concepts.nn import ConceptLoss, L1LogitRegularizer

       loss = ConceptLoss(
           annotations=annotations,
           binary=[torch.nn.BCEWithLogitsLoss(), L1LogitRegularizer(scale=1.0)],
           binary_weights=[1.0, 0.01],
           categorical=torch.nn.CrossEntropyLoss(),
           continuous=torch.nn.MSELoss(),
       )

    **Weighting concepts vs tasks differently.** :class:`~torch_concepts.nn.WeightedConceptLoss`
    splits the loss into a concept term and a task term, each with its own scalar weight:

    .. code-block:: python

       from torch_concepts.nn import WeightedConceptLoss

       loss = WeightedConceptLoss(
           annotations=annotations,
           concept_weight=0.5,
           task_weight=1.0,
           task_names=['cancer'],
           binary=torch.nn.BCEWithLogitsLoss(),
           categorical=torch.nn.CrossEntropyLoss(),
           continuous=torch.nn.MSELoss(),
       )

    **Metrics.** :class:`~torch_concepts.nn.ConceptMetrics` follows the same type-aware pattern.
    Each type accepts a ``dict`` of ``name → torchmetrics.Metric`` — any
    `torchmetrics <https://torchmetrics.readthedocs.io>`_ metric works (or custom metrics can be designed). 
    For categorical concepts, pass a ``(MetricClass, kwargs)`` tuple so the right ``num_classes`` is
    inferred automatically from the annotations:

    .. code-block:: python

       from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, MulticlassAccuracy
       from torch_concepts.nn import ConceptMetrics

       metrics = ConceptMetrics(
           annotations=annotations,
           binary={
               'accuracy': BinaryAccuracy(),
               'auroc':    BinaryAUROC(),
           },
           categorical={
               'accuracy': (MulticlassAccuracy, {'average': 'micro'}),
           },
           per_concept=True,   # a MetricCollection per individual concept
           summary=True,       # also a MetricCollection per type (SUMMARY-binary_*, ...)
       )


.. dropdown:: Training
    :icon: workflow

    High-level models support two training modes.

    **Manual PyTorch.** Instantiate the model without Lightning and write your own loop. Querying
    returns logits you can feed to any loss:

    .. code-block:: python

       import torch
       from torch_concepts.nn import ConceptBottleneckModel, MLP

       model = ConceptBottleneckModel(
           input_size=n_features,
           annotations=annotations,
           task_names=['xor'],
           backbone=MLP(input_size=n_features, hidden_size=128, n_layers=1),
           latent_size=128,
       )

       query = ['c1', 'c2', 'xor']
       optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
       loss_fn = torch.nn.BCEWithLogitsLoss()

       model.train()
       for epoch in range(500):
           optimizer.zero_grad()
           out = model(input=x_train, query=query)
           logits = torch.cat([out.params[name]['logits'] for name in query], dim=1)
           loss = loss_fn(logits, target)
           loss.backward()
           optimizer.step()

    **PyTorch Lightning.** Pass ``lightning=True`` together with a ``loss`` (either a standard
    |pytorch_logo| PyTorch loss or a |pyc_logo| :class:`~torch_concepts.nn.ConceptLoss`),
    optional ``metrics``, and an optimizer; then hand the model and a datamodule to a ``Trainer``:

    .. code-block:: python

       from pytorch_lightning import Trainer

       model = ConceptBottleneckModel(
           input_size=n_features,
           annotations=annotations,
           task_names=['xor'],
           backbone=MLP(input_size=n_features, hidden_size=128, n_layers=1),
           latent_size=128,
           lightning=True,
           loss=loss,
           metrics=metrics,
           optim_class=torch.optim.AdamW,
           optim_kwargs={'lr': 0.02},
       )

       trainer = Trainer(max_epochs=100)
       trainer.fit(model, datamodule=datamodule)


.. dropdown:: Putting It Together: Concept Bottleneck Model
    :icon: package

    A complete, end-to-end Lightning pipeline on a toy dataset:

    .. code-block:: python

       import torch
       from pytorch_lightning import Trainer

       from torch_concepts import seed_everything
       from torch_concepts.nn import ConceptBottleneckModel, ConceptLoss, MLP
       from torch_concepts.data import ToyDataset
       from torch_concepts.data.base.datamodule import ConceptDataModule

       seed_everything(42)

       # Data
       dataset = ToyDataset(dataset='xor', seed=42, n_gen=10000)
       datamodule = ConceptDataModule(dataset=dataset, batch_size=2048,
                                      val_size=0.1, test_size=0.2, seed=42)
       annotations = dataset.annotations
       n_features = dataset.input_data.shape[1]

       # Type-aware loss
       loss = ConceptLoss(
           annotations=annotations,
           binary=torch.nn.BCEWithLogitsLoss(),
           categorical=torch.nn.CrossEntropyLoss(),
       )

       # Model (Lightning mode)
       model = ConceptBottleneckModel(
           input_size=n_features,
           annotations=annotations,
           task_names=['xor'],
           backbone=MLP(input_size=n_features, hidden_size=128, n_layers=1),
           latent_size=128,
           lightning=True,
           loss=loss,
           optim_class=torch.optim.AdamW,
           optim_kwargs={'lr': 0.02},
       )

       # Train
       trainer = Trainer(max_epochs=100)
       trainer.fit(model, datamodule=datamodule)


Next Steps
----------

- Browse the full :doc:`High-Level API reference </modules/high_level_api>`.
- Customise the :doc:`Interpretable Probabilistic Models <using_mid_level>` behind these models.
- Run experiments without code using :doc:`Conceptarium <using_conceptarium>`.
