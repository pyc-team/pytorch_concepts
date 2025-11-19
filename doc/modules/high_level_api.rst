High-level API
==============

High-level APIs allow you to instantiate and use out-of-the-box state-of-the-art models with 1 line of code.


Documentation
----------------

.. toctree::
   :maxdepth: 1

   nn.base.high
   nn.models.high


Design principles
-----------------


Annotations
^^^^^^^^^^^

Annotations are used to define the structure of high-level models directly from data. For instance, we can define a concept annotation as:

.. code-block:: python

   labels = ["c1", "c2", "c3"]
   cardinalities = [2, 1, 3]
   metadata = {
         'c1': {'distribution': torch.distributions.RelaxedOneHotCategorical},
         'c2': {'distribution': torch.distributions.RelaxedBernoulli},
         'c3': {'distribution': torch.distributions.RelaxedOneHotCategorical},
     }
   annotations = pyc.Annotations({1: pyc.AxisAnnotation(labels=labels,
                                                        cardinalities=cardinalities,
                                                        metadata=metadata)})

Out-of-the-box Models
^^^^^^^^^^^^^^^^^^^^^^

We can instantiate out-of-the-box high-level models using annotations. For instance, we can instantiate a Concept Bottleneck Model as:

.. code-block:: python

   model = pyc.nn.CBM(
       task_names=['c3'],
       inference=pyc.nn.DeterministicInference,
       input_size=64,
       annotations=annotations,
       encoder_kwargs={'hidden_size': 16,
                       'n_layers': 1,
                       'activation': 'leaky_relu',
                       'dropout': 0.}
   )
