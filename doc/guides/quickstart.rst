Quickstart
==========

This page trains a first interpretable model in a few minutes. It assumes |pyc_logo| PyC is
installed with the ``data`` extra (see :doc:`installation`).

.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle


Concepts and tasks
------------------

A **concept** is a property of the input that a person can name and check: the digit in an
image, or its color. A **task** is what the model must finally predict, for example whether
the digit is even. A **concept bottleneck model** (CBM) predicts the concepts from the input,
and then the task from the concepts alone. This makes the model inspectable (which concepts
did it see?) and correctable: an expert can fix a wrong concept at test time, and the task
prediction follows. Fixing a concept in this way is called an **intervention**.

PyC describes the concepts of a dataset with :class:`~torch_concepts.Annotations`: their names,
their types (binary or categorical) and their number of classes. Datasets provide them, and
models are built from them.


1. Load the data
----------------

Color-MNIST tints MNIST digits red or green. Its concepts are ``digit`` (10 classes) and
``color`` (2 classes); its task, ``parity`` (1 if the digit is even), is stored as one more
concept. MNIST (~60 MB) is downloaded on first use.

.. code-block:: python

   import torch
   from pytorch_lightning import Trainer

   from torch_concepts import seed_everything
   from torch_concepts.data import ColorMNISTDataModule
   from torch_concepts.env import DATA_ROOT
   from torch_concepts.nn import (
       MLP,
       ConceptBottleneckModel,
       ConceptLoss,
       GroundTruthIntervention,
       UniformPolicy,
       intervention,
   )

   seed_everything(42)
   datamodule = ColorMNISTDataModule(
       root=str(DATA_ROOT / "colormnist"),
       max_samples=10000,
       batch_size=256,
   )
   print(datamodule.annotations.labels)  # ['parity', 'color', 'digit']


2. Build the model
------------------

A ready-made model takes the input size, the annotations, the names of the tasks among them,
and a backbone that maps an input to a latent representation. With ``lightning=True`` it is
also a PyTorch Lightning module, so it needs a loss and an optimizer.
:class:`~torch_concepts.nn.ConceptLoss` applies one loss per concept type.

.. code-block:: python

   model = ConceptBottleneckModel(
       input_size=datamodule.n_features,  # (3, 28, 28) images
       annotations=datamodule.annotations,
       task_names=["parity"],
       backbone=torch.nn.Sequential(torch.nn.Flatten(), MLP(3 * 28 * 28, 128)),
       latent_size=128,
       lightning=True,
       loss=ConceptLoss(
           binary=torch.nn.BCEWithLogitsLoss(),
           categorical=torch.nn.CrossEntropyLoss(),
       ),
       optim_class=torch.optim.AdamW,
       optim_kwargs={"lr": 1e-3},
   )


3. Train
--------

.. code-block:: python

   trainer = Trainer(max_epochs=10)
   trainer.fit(model, datamodule=datamodule)


4. Predict the concepts and the task
------------------------------------

``query`` names the variables to compute, and the output holds their logits by name. The
labels hold one column per concept: 0 or 1 for a binary concept, the class index for a
categorical one.

.. code-block:: python

   test = datamodule.testset.indices
   x_test = datamodule.dataset.input_data[test]
   c_test = datamodule.dataset.concepts[test]

   model.eval()
   with torch.no_grad():
       out = model(query=["digit", "color", "parity"], input=x_test)

   predicted_digit = out.logits["digit"].argmax(-1, keepdim=True)
   predicted_parity = (out.logits["parity"] > 0).float()
   digit_accuracy = (predicted_digit == c_test["digit"]).float().mean()
   parity_accuracy = (predicted_parity == c_test["parity"]).float().mean()
   print(f"digit accuracy {digit_accuracy:.2f}, parity accuracy {parity_accuracy:.2f}")


5. Intervene on a concept
-------------------------

The task is predicted from the concepts alone, so revealing the true digit should fix the
parity mistakes. :func:`~torch_concepts.nn.intervention` replaces the predicted digit with
the true one in every forward pass inside the ``with`` block. The digit is predicted as
logits, so the true class becomes a large logit.

.. code-block:: python

   one_hot = torch.nn.functional.one_hot(c_test["digit"].flatten().long(), 10)
   true_digit = GroundTruthIntervention(torch.logit(one_hot.float(), eps=1e-6))

   with torch.no_grad(), intervention(model, true_digit, UniformPolicy(), ["digit"]):
       out = model(query=["parity"], input=x_test)
   predicted_parity = (out.logits["parity"] > 0).float()
   parity_accuracy = (predicted_parity == c_test["parity"]).float().mean()
   print(f"parity accuracy with the true digit {parity_accuracy:.2f}")

The output should be close to::

   digit accuracy 0.92, parity accuracy 0.96
   parity accuracy with the true digit 1.00

Every parity mistake came from a misread digit: once the digit is right, so is the task.


Where next
----------

- The :doc:`User Guide <using>` explains the three API levels: layers, probabilistic models
  and ready-made models.
- :doc:`Datasets <using_data>` covers loading data, what a batch contains and where data is
  stored.
- The :doc:`Examples </examples>` show each feature in a short script.
