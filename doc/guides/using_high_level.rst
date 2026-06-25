.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pl_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/lightning.svg
    :width: 20px
    :align: middle


Out-of-the-box Models
=====================

The High-Level API gives you state-of-the-art concept-based models out of the box. You
describe your concepts once, pick a model, and train it — manually or in one line with
|pl_logo| PyTorch Lightning. It is the right entry point if you just want to use a model
without assembling it yourself.

.. image:: /_static/img/api_levels/high_level.png
   :alt: Overview of the PyC High-Level API
   :align: center
   :width: 100%

|

Working at this level involves three building blocks:

- **Models** — ready-to-use concept-based models such as the Concept Bottleneck Model.
- **Losses & Metrics** — type-aware objectives and metrics that route automatically.
- **Training** — a manual |pyc_logo| PyTorch loop, or automatic |pl_logo| Lightning training.

All of them are configured with :class:`~torch_concepts.Annotations`, which describe your
concepts and tasks (see :doc:`Annotations <using_low_level>`). Most datasets expose them
directly as ``dataset.annotations``.

Expand each block below for an explanation and an example.


.. dropdown:: Models
    :icon: rocket

    |pyc_logo| PyC provides ready-to-use models that are instantiated with minimal configuration.
    The available models include:

    - :class:`~torch_concepts.nn.ConceptBottleneckModel` — the standard CBM.
    - :class:`~torch_concepts.nn.ConceptEmbeddingModel` — an expressive CBM with concept embeddings.
    - :class:`~torch_concepts.nn.GraphConceptBottleneckModel` and
      :class:`~torch_concepts.nn.CausallyReliableConceptBottleneckModel` — graph-structured variants.
    - :class:`~torch_concepts.nn.BlackBox` — a non-interpretable baseline for comparison.

    A model is created from an ``input_size``, the ``annotations``, and the ``task_names``; the
    remaining concepts are treated as intermediate concepts. You then query it for any subset of
    variables. The output exposes per-variable logits in ``out.params[name]['logits']``.

    .. code-block:: python

       import torch
       from torch_concepts.nn import ConceptBottleneckModel, MLP

       model = ConceptBottleneckModel(
           input_size=n_features,
           annotations=annotations,
           task_names=['xor'],
           backbone=MLP(input_size=n_features, hidden_size=128, n_layers=1),
           latent_size=128,   # output size of the backbone
       )

       query = ['c1', 'c2', 'xor']
       with torch.no_grad():
           out = model(input=x, query=query)
       xor_logits = out.params['xor']['logits']


.. dropdown:: Losses & Metrics
    :icon: flame

    :class:`~torch_concepts.nn.ConceptLoss` and :class:`~torch_concepts.nn.ConceptMetrics` are
    **type-aware**: given the annotations, they automatically route each concept to the right
    objective/metric based on whether it is binary, categorical, or continuous — so a single object
    handles mixed concept spaces.

    .. code-block:: python

       import torch
       from torchmetrics.classification import BinaryAccuracy
       from torch_concepts.nn import ConceptLoss, ConceptMetrics

       loss = ConceptLoss(
           annotations=annotations,
           binary=torch.nn.BCEWithLogitsLoss(),
           categorical=torch.nn.CrossEntropyLoss(),
       )

       metrics = ConceptMetrics(
           annotations=annotations,
           binary={'accuracy': BinaryAccuracy()},
           summary=True,       # aggregate per type
           per_concept=True,   # also track each concept individually
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

    **PyTorch Lightning.** Pass ``lightning=True`` together with a ``loss``, optional ``metrics``,
    and an optimizer; then hand the model and a datamodule to a ``Trainer``:

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


.. dropdown:: Putting It Together
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
- Customise the probabilistic model behind these models with the :doc:`Mid-Level API <using_mid_level>`.
- Run experiments without code using :doc:`Conceptarium <using_conceptarium>`.
