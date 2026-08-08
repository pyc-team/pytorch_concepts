.. |pyc_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg
   :width: 20px
   :align: middle

.. |pytorch_logo| image:: https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg
   :width: 20px
   :align: middle


Contributing a New Model
========================

This guide explains how to add a new high-level concept-based model to |pyc_logo| PyC.
All high-level models extend :class:`~torch_concepts.nn.modules.high.base.bipartite.BipartiteModel`,
which wires together the backbone, the probabilistic graphical model, and the inference
engine for you. Your job as a contributor is to write a single ``__init__`` that
assembles the Bayesian network specific to your model. Everything else — the forward
pass, Lightning training, concept queries — is inherited.


.. dropdown:: Prerequisites
   :icon: checklist

   Before starting, make sure you are comfortable with:

   - **Python and PyTorch** — the guide assumes you can write a custom ``nn.Module``.
   - **The PyC low-level API** — specifically :class:`~torch_concepts.nn.LinearEmbeddingToConcept`
     and :class:`~torch_concepts.nn.LinearConceptToConcept`. Skim
     :doc:`Using the Low-Level API <using_low_level>` if you haven't.
   - **The PyC mid-level API** — specifically
     :class:`~torch_concepts.nn.BayesianNetwork`,
     :class:`~torch_concepts.nn.ParametricCPD`,
     :class:`~torch_concepts.nn.ConceptVariable`, and
     :class:`~torch_concepts.nn.EmbeddingVariable`. Skim
     :doc:`Using the Mid-Level API <using_mid_level>` if you haven't.
   - **The |pyc_logo| PyC development setup** — see :doc:`Contributing <contributing>` for
     how to install the library in editable mode and run the tests.

   You do **not** need to understand the inference engine internals — the base class
   handles them.


.. dropdown:: Model Structure
   :icon: code

   Every high-level bipartite model extends
   :class:`~torch_concepts.nn.modules.high.base.bipartite.BipartiteModel`.
   The class hierarchy is::

       BipartiteModel  (BipartiteMixin + DirectedGraphModel)
       └── YourModel

   You only implement ``__init__``. The required steps are:

   1. Call ``super().__init__(input_size, annotations, task_names, lightning, **kwargs)``
      to let the base class set up the backbone and annotations.
   2. Build ``self.pgm`` — a :class:`~torch_concepts.nn.BayesianNetwork` whose
      variables and factors describe your model.
   3. Call ``self.setup_inference(inference, inference_kwargs, train_inference, train_inference_kwargs)``
      to wire the inference engine.

   **What** ``super().__init__`` **gives you**

   After the ``super().__init__`` call returns, these attributes are available:

   - ``self.input_size`` — raw input dimension.
   - ``self.latent_size`` — backbone output dimension (equals ``input_size`` when no
     backbone is provided).
   - ``self.backbone`` — the backbone ``nn.Module`` (default: ``MLP(input_size, latent_size)``).
   - ``self.concept_annotations`` — the :class:`~torch_concepts.Annotations` object.
   - ``self.intermediate_concept_names`` — concept labels that are *not* task labels.
   - ``self.task_names`` — task label names.
   - ``self.distribution_of(name)`` — returns the distribution class for a concept.
   - ``self.dist_kwargs_of(name)`` — returns distribution kwargs for a concept.
   - ``self.build_concept_variables(names, plate_name)`` — the shared factory every
     shipped model uses to turn a list of concept names into ``ConceptVariable`` objects:
     homogeneous concepts (same type *and* cardinality) are grouped into the minimum
     number of plates — even a lone concept becomes a single-member plate — while a
     heterogeneous level splits into one plate per ``(type, cardinality)`` family.
     Pass ``plate=False`` to the model constructor to instead get one individual
     variable per concept.
   - ``self.build_concept_embedding_variables(names, embedding_size, plate_name)`` —
     the matching factory for per-concept state embeddings. Called with the *same*
     ``names`` as ``build_concept_variables``, it returns a list that lines up
     element-by-element: group *i*'s embedding matrix feeds group *i*'s concepts.
     A concept contributes one embedding row per state (its cardinality) — except
     a **binary** concept, which contributes **two** (:math:`w^+`, :math:`w^-`)
     even though it is a single Bernoulli, matching the Concept Embedding Model's
     positive/negative context pair (Espinosa Zarlenga et al., NeurIPS 2022).

   **Building the PGM**

   Every PGM starts with the same ``input → latent`` block. This is **not** inherited —
   every shipped model (CBM, CEM) defines its own copy — so write it once in your class::

       def _input_latent_block(self):
           input_var = EmbeddingVariable("input", distribution=Delta, shape=self.input_size)
           latent_var = EmbeddingVariable("latent", distribution=Delta, size=self.latent_size)
           input_cpd = ParametricCPD(
               input_var, parents=[], parametrization=LearnablePrior(input_var.shape),
           )
           latent_cpd = ParametricCPD(
               latent_var, parents=[input_var], parametrization=self.backbone,
           )
           return input_var, latent_var, input_cpd, latent_cpd

   This returns four objects:

   - ``EmbeddingVariable("input", distribution=Delta, shape=self.input_size)`` — receives
     raw data as evidence.
   - ``EmbeddingVariable("latent", distribution=Delta, size=self.latent_size)`` — the
     backbone output.
   - A :class:`~torch_concepts.nn.ParametricCPD` for ``input`` (a :class:`~torch_concepts.nn.LearnablePrior`).
   - A :class:`~torch_concepts.nn.ParametricCPD` for ``latent | input`` (``self.backbone``).

   After that, build your concept and task variables with the shared factories above
   and wire their CPDs — this single method (conventionally named ``_build_model``,
   called once from ``__init__``) is the entire "build" step; there is no separate
   constructor/lazy-build phase.

   **Complete worked example**

   The model below is a minimal but real bipartite model that gives each concept a
   per-state embedding and predicts tasks by mixing concept activations with those
   embeddings (the CEM predictor head). It mirrors
   :class:`~torch_concepts.nn.ConceptEmbeddingModel`'s own ``_build_model`` — study
   ``torch_concepts/nn/modules/high/models/cem.py`` directly for the shipped version —
   and, unlike a version that hand-builds a single ``ConceptVariable`` for every
   concept, it handles a **heterogeneous** annotation (mixed types/cardinalities)
   correctly because it goes through the shared plate factories.  It is self-contained
   and can serve as a copy-paste starting point.

   .. code-block:: python

      # torch_concepts/nn/modules/high/models/my_model.py
      from typing import List, Optional, Union

      import torch
      import torch.nn as nn
      from torch.distributions import Bernoulli, OneHotCategorical, Normal

      from torch_concepts.annotations import Annotations
      from torch_concepts.distributions import Delta
      from torch_concepts.nn.modules.low.dense_layers import LinearEmbeddingEncoder
      from torch_concepts.nn.modules.low.encoders.linear import LinearEmbeddingToConcept
      from torch_concepts.nn.modules.low.predictors.mix import MixConceptEmbeddingToConcept
      from torch_concepts.nn.modules.low.priors import LearnablePrior
      from torch_concepts.nn.modules.low.sequential import Sequential
      from torch_concepts.nn.modules.mid.inference.base import BaseInference
      from torch_concepts.nn.modules.mid.inference.torch.deterministic import DeterministicInference
      from torch_concepts.nn.modules.mid.graph.bayesian_network import BayesianNetwork
      from torch_concepts.nn.modules.mid.factors.cpd import ParametricCPD
      from torch_concepts.nn.modules.mid.variable import EmbeddingVariable
      from torch_concepts.nn.modules.mid.distributions import DEFAULT_DIST_KWARGS
      from torch_concepts.nn.modules.high.base.bipartite import BipartiteModel


      class MyConceptModel(BipartiteModel):
          """Bipartite concept model: per-concept embeddings, mix-embedding task predictor.

          Each concept gets a per-state embedding, decoded from the latent
          representation. Tasks are predicted by mixing concept activations with
          those embeddings, following the CEM predictor head.

          Parameters
          ----------
          input_size : int
              Dimensionality of raw input features.
          annotations : Annotations
              Concept annotations (labels, cardinalities, types).
          task_names : list of str
              Subset of annotation labels used as task variables.
          embedding_size : int, default 8
              Width of each per-state concept embedding.
          inference : BaseInference, optional
              Evaluation inference engine class.
          inference_kwargs : dict, optional
              Keyword arguments for the evaluation engine.
          train_inference : BaseInference, optional
              Training inference engine class (defaults to ``inference``).
          train_inference_kwargs : dict, optional
              Keyword arguments for the training engine.
          lightning : bool, default False
              Set True to enable PyTorch Lightning training.
          plate : bool or None, default None
              Per-level plate preference. ``None``/``True`` group homogeneous
              concepts into the minimum number of plates; ``False`` uses one
              individual variable per concept.
          **kwargs
              Forwarded to :class:`BaseModel` (e.g. ``backbone``, ``latent_size``).
          """

          supported_concept_types = frozenset({"binary", "categorical", "continuous"})
          param_for_discrete_var = "logits"

          variable_distributions = {
              'binary': Bernoulli,
              'categorical': OneHotCategorical,
              'continuous': Normal,
          }
          variable_dist_kwargs = dict(DEFAULT_DIST_KWARGS)

          def __init__(
              self,
              input_size: int,
              annotations: Annotations,
              task_names: Union[List[str], str],
              embedding_size: int = 8,
              inference: Optional[BaseInference] = DeterministicInference,
              inference_kwargs: Optional[dict] = None,
              train_inference: Optional[BaseInference] = None,
              train_inference_kwargs: Optional[dict] = None,
              lightning: bool = False,
              plate: Optional[bool] = None,
              **kwargs,
          ):
              # Step 1 — base class sets up backbone, annotations, sizes, and the
              # bipartite concept -> task graph.
              super().__init__(
                  input_size=input_size,
                  annotations=annotations,
                  task_names=task_names,
                  lightning=lightning,
                  plate=plate,
                  **kwargs,
              )
              self.embedding_size = embedding_size

              # Step 2 — the single builder that assembles the Bayesian network.
              self.pgm = self._build_model()

              # Step 3 — wire the inference engines.
              self.setup_inference(
                  inference,
                  inference_kwargs,
                  train_inference,
                  train_inference_kwargs,
              )

          def _input_latent_block(self):
              """Raw input -> latent block (copy-pasted per model, see above)."""
              input_var = EmbeddingVariable("input", distribution=Delta, shape=self.input_size)
              latent_var = EmbeddingVariable("latent", distribution=Delta, size=self.latent_size)
              input_cpd = ParametricCPD(
                  input_var, parents=[], parametrization=LearnablePrior(input_var.shape),
              )
              latent_cpd = ParametricCPD(
                  latent_var, parents=[input_var], parametrization=self.backbone,
              )
              return input_var, latent_var, input_cpd, latent_cpd

          def _build_model(self) -> BayesianNetwork:
              """Assemble the PGM: input -> latent -> embeddings -> concepts -> tasks.

              Concepts and their state embeddings share the same minimum-plate
              grouping, so the two lists line up element-by-element: group i's
              embedding matrix produces group i's concept scores.
              """
              input_var, latent_var, input_cpd, latent_cpd = self._input_latent_block()

              concepts = self.build_concept_variables(self.intermediate_concept_names, plate_name="concepts")
              embeddings = self.build_concept_embedding_variables(
                  self.intermediate_concept_names, self.embedding_size, plate_name="embeddings"
              )
              tasks = self.build_concept_variables(self.task_names, plate_name="tasks")

              # latent -> embeddings: one batched encoder per group.
              emb_cpd = ParametricCPD(
                  variable=embeddings,
                  parents=[latent_var],
                  parametrization=[
                      {"value": LinearEmbeddingEncoder(
                          in_features=self.latent_size,
                          out_features=self.embedding_size,
                          n_embeddings=e.shape[0],
                      )}
                      for e in embeddings
                  ],
              )

              # embeddings -> concepts: one score per state embedding, per group.
              concept_cpd = [
                  ParametricCPD(
                      variable=cvar,
                      parents=[evar],
                      parametrization=self._flexible_parametrization(
                          variable=cvar,
                          first=Sequential(
                              LinearEmbeddingToConcept(in_embeddings=self.embedding_size, out_concepts=1),
                              nn.Flatten(start_dim=-2),
                          ),
                          second=concept_head(),
                      ),
                  )
                  for cvar, evar in zip(concepts, embeddings)
              ]

              # concepts + embeddings -> tasks: mix each concept's activation with its
              # embedding. Concepts concatenate along the last axis, embeddings along
              # the second-to-last (state) axis, so a custom `aggregate` keeps the two
              # apart before the mixer sees them.
              ordered_names = [m for cvar in concepts for m in cvar.members]
              mix_axis = self.concept_annotations.subset(ordered_names)

              def mix_parents(concepts, embeddings):
                  return {
                      "concepts": torch.cat(list(concepts.values()), dim=-1),
                      "embeddings": torch.cat(list(embeddings.values()), dim=-2),
                  }

              task_cpd = ParametricCPD(
                  variable=tasks,
                  parents=[*concepts, *embeddings],
                  parametrization=[
                      self._flexible_parametrization(
                          variable=tvar,
                          first=MixConceptEmbeddingToConcept(
                              in_concepts=mix_axis,
                              in_embeddings=self.embedding_size,
                              out_concepts=tvar.size,
                          ),
                          second=task_head(tvar),
                      )
                      for tvar in tasks
                  ],
                  aggregate=mix_parents,
              )

              return BayesianNetwork(
                  variables=[input_var, latent_var, *embeddings, *concepts, *tasks],
                  factors=[input_cpd, latent_cpd, *emb_cpd, *concept_cpd, *task_cpd],
              )

   **Instantiating and calling the model**

   .. code-block:: python

      import torch
      import torch_concepts as pyc
      from torch_concepts.nn.modules.high.models.my_model import MyConceptModel

      annotations = pyc.Annotations(
          labels=["smoking", "genotype", "tar", "cancer"],
          cardinalities=[1, 3, 1, 1],
          types=["binary", "categorical", "continuous", "binary"],
      )
      n_features = 64

      model = MyConceptModel(
          input_size=n_features,
          annotations=annotations,
          task_names=["cancer"],
          embedding_size=8,
      )

      x = torch.randn(16, n_features)
      out = model(input=x, query=["smoking", "genotype", "tar", "cancer"])

      # out.params maps each queried name to its distribution parameters.
      smoking_logits = out.params["smoking"]["logits"]   # (16, 1)
      genotype_logits = out.params["genotype"]["logits"] # (16, 3)
      tar_params      = out.params["tar"]                # {'loc': ..., 'scale': ...}
      cancer_logits   = out.params["cancer"]["logits"]   # (16, 1)


.. dropdown:: Configuring Distributions
   :icon: gear

   Each model class carries a ``variable_distributions`` class attribute — a dict that
   maps a concept type string (``'binary'``, ``'categorical'``, ``'continuous'``) to a
   ``torch.distributions`` class. The base class resolves the distribution for each
   concept at construction time using its type from the :class:`~torch_concepts.Annotations`.

   **Default distributions** (same as CBM and CEM):

   .. code-block:: python

      variable_distributions = {
          'binary':      Bernoulli,
          'categorical': OneHotCategorical,
          'continuous':  Normal,
      }

   **Override per type at the class level** — place this in your model class to use
   relaxed distributions during training:

   .. code-block:: python

      from torch.distributions import RelaxedBernoulli, RelaxedOneHotCategorical

      class MyConceptModel(BipartiteModel):
          variable_distributions = {
              'binary':      RelaxedBernoulli,
              'categorical': RelaxedOneHotCategorical,
              'continuous':  Normal,
          }
          variable_dist_kwargs = {
              RelaxedBernoulli:         {'temperature': 0.5},
              RelaxedOneHotCategorical: {'temperature': 0.5},
          }

   **Override per instance** — the user can pass ``variable_distributions`` when they
   construct your model. The instance dict is merged on top of the class dict, so only
   the keys the user specifies change:

   .. code-block:: python

      model = MyConceptModel(
          input_size=64,
          annotations=annotations,
          task_names=["cancer"],
          variable_distributions={"binary": RelaxedBernoulli},
          variable_dist_kwargs={RelaxedBernoulli: {"temperature": 0.7}},
      )

   Both the class-level and instance-level overrides work through the base class
   — you do not need to do anything special in your ``__init__`` to support them.


.. dropdown:: Registering the Model
   :icon: package

   Once your model works, register it in four places.

   **1. Module file**

   Create ``torch_concepts/nn/modules/high/models/my_model.py`` with your class.

   **2. Public API export**

   Add two lines to ``torch_concepts/nn/__init__.py`` — the import and the ``__all__``
   entry:

   .. code-block:: python

      # in torch_concepts/nn/__init__.py
      from .modules.high.models.my_model import MyConceptModel

      __all__ = [
          ...
          "MyConceptModel",
      ]

   After this, users can do ``from torch_concepts.nn import MyConceptModel``.

   **3. Conceptarium YAML** (optional, for no-code experiment runs)

   Create ``conceptarium/conf/model/my_model.yaml``:

   .. code-block:: yaml

      defaults:
        - _commons
        - _self_

      _target_: "torch_concepts.nn.MyConceptModel"

      task_names: ${dataset.default_task_names}
      embedding_size: 8

   The ``defaults: [_commons]`` line pulls in the shared backbone, optimizer,
   inference, and distribution settings from ``_commons.yaml``. Only add keys
   that are specific to your model.

   **4. API reference page**

   Add your model's name to the ``Models`` autosummary table in
   ``doc/modules/high_level_api.rst`` so the class docstring appears in the
   rendered documentation:

   .. code-block:: rst

      .. autosummary::
         :toctree: generated
         :nosignatures:

         ConceptBottleneckModel
         ConceptEmbeddingModel
         GraphConceptBottleneckModel
         CausallyReliableConceptBottleneckModel
         BlackBox
         BlackBoxTaskOnly
         MyConceptModel

   **5. Tests**

   Add a test in ``tests/`` that constructs your model, runs a forward pass, and
   checks the output shapes. Mirror the existing tests in
   ``tests/test_high_level.py`` for the expected structure.


Next Steps
----------

- Need a custom layer for your model? See :doc:`Contributing a New Layer <contributing_layer>`.
- Read the full :doc:`High-Level API reference </modules/high_level_api>` to see how
  existing models are documented.
- Explore :doc:`Using the Mid-Level API <using_mid_level>` to understand the Bayesian
  network primitives your PGM is built from.
- Run experiments without code using :doc:`Conceptarium <using_conceptarium>` once your
  YAML file is in place.
- Open a pull request to ``dev`` — see :doc:`Contributing <contributing>` for the
  full workflow.
