PYTORCH CONCEPTS DOCUMENTATION
===============================

<p align="center">
  <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/doc/_static/img/pyc_logo.png" alt="PyC Logo" width="40%">
</p>

# PyC

PyC is a library built upon PyTorch to easily implement **interpretable and causally transparent deep learning models**.
The library provides primitives for layers (encoders, predictors, special layers), Probabilistic Models, and APIs for running experiments at scale.

The name of the library stands for both
- **PyTorch Concepts**: as concepts are essential building blocks for interpretable deep learning.
- $P(y|C)$: as the main purpose of the library is to support sound probabilistic modeling of the conditional distribution of outputs $y$ given concepts $C$.

You can install PyC along with all its dependencies from
[PyPI](https://pypi.org/project/pytorch-concepts/):

```pip install pytorch-concepts ```

The folder [https://github.com/pyc-team/pytorch_concepts/tree/master/examples](https://github.com/pyc-team/pytorch_concepts/tree/master/examples)
 includes many examples showing how the library can be used.


The library is organized to be modular and accessible at different levels of abstraction:
- **No-code APIs. Use case: applications and benchmarking.** These APIs allow to easily run large-scale highly parallelized and standardized experiments by interfacing with configuration files.
- **High-level APIs. Use case: use out-of-the-box state-of-the-art models.** These APIs allow to instantiate use implemented models with 1 line of code.
- **Mid-level APIs. Use case: build custom interpretable and causally transparent Probabilistic Models.** These APIs allow to build new interpretable probabilistic models and run efficient tensorial probabilistic inference using a Probabilistic Model interface.
- **Low-level APIs. Use case: assemble custom interpretable architectures.** These APIs allow to build architectures from basic interpretable layers in a plain pytorch-like interface. These APIs also include metrics, losses, and datasets.

<p align="center">
  <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/doc/_static/img/pyc_software_stack.png" alt="PyC Software Stack" width="100%">
</p>


# API overview

## Design principles of low-level APIs

### Objects
In PyC there are three types of objects:
- **Embedding**: high-dimensional latent representations shared across all concepts.
- **Exogenous**: high-dimensional latent representations related to a specific concept.
- **Logits**: Concept scores before applying an activation function.

### Layers
There are only three types of layers:
- **Encoders**: layers that map latent representations (embeddings or exogenous) to logits.
  - `ExogEncoder`: predicts exogenous representations from embeddings.
  - `ProbEncoderFromEmb`: predicts concept logits from embeddings.
  - `ProbEncoderFromExog`: predicts concept logits from exogenous representations.
  - `StochasticEncoderFromEmb`: predicts concept logits sampled from a multivariate normal distribution whose parameters are predicted from embeddings.

- **Predictors**: layers that map logits (plus optionally latent representations) to other logits.
  - `ProbPredictor`: predicts output logits from input logits.
  - `MixProbExogPredictor`: predicts output logits mixing parent logits and exogenous representations of the parent concepts.
  - `HyperLinearPredictor`: generates a linear equation using the exogenous representations of the output concepts and applies it to the input logits to predict output logits.

- **Special layers**
  - `MemorySelector`: uses an embedding to select an exogenous representation from a fixed-size memory bank (useful to implement verifiable architectures).
  - `COSMOGraphLearner`: learns a directed acyclic graph (useful to learn concept dependencies).

### Models
A model is built as a ModuleDict which may include standard PyTorch layers + PyC encoders and predictors.

### Inference
At this API level, there are two types of inference that can be performed:
- **Standard forward pass**: a standard forward pass using the forward method of each layer in the ModuleDict.
- **Interventions**: interventions are context managers that temporarily modify a layer in the ModuleDict. So, when a forward pass is performed within an intervention context, the intervened layer behaves differently with  a cascading effect on all subsequent layers.
 - `intervention`: a context manager to intervene on concept scores.
 - **Intervention strategies**: define how the intervened layer behaves within an intervention context.
   - `GroundTruthIntervention`: replaces the concept logits with ground truth values.
   - `DoIntervention`: performs a do-intervention on the concept logits with a constant value.
   - `DistributionIntervention`: replaces the concept logits with samples from a given distribution.
 - **Intervention Policies**: define the order/set of concepts to intervene on.
   - `UniformPolicy`: applies interventions on all concepts uniformly.
   - `RandomPolicy`: randomly selects concepts to intervene on.
   - `UncertaintyInterventionPolicy`: selects concepts to intervene on based on the uncertainty represented by their logits.


## Design principles of mid-level APIs

### Probabilistic Models
At this API level, models are represented as Probabilistic Models where:
- **Variables**: represent random variables in the Probabilistic Model. Variables are defined by their name, parents, and distribution type.
- **Factors**: represent conditional probability distributions (CPDs) between variables in the Probabilistic Model and are parameterized by PyC layers.
- **Probabilistic Model**: a collection of variables and factors.

### Inference
Inference is performed using efficient tensorial probabilistic inference algorithms. We currently support:
- `DeterministicInference`: standard forward pass through the ProbabilisticModel from the source variables to the sink variables of a DAG.
- `AncestralSampling`: ancestral sampling from the ProbabilisticModel from the source variables to the sink variables of a DAG.


## Design principles of high-level APIs

### Objects
- `Annotations`: A class to handle concept and task annotations.
- `ConceptGraph`: A class to handle concept graphs defining dependencies among concepts and tasks.

### High-level Models
- `BipartiteModel`: A handy model to build concept bottleneck models with a bipartite structure where concepts are independent and directly connected to tasks.
- `GraphModel`: A handy model to build concept bottleneck models with an arbitrary directed acyclic graph (DAG) structure among concepts (all labels are represented as concepts).


## Design principles of no-code APIs
- `BaseModel`: A base class you can extend to build new concept bottlenecks.
- `ConceptBottleneckModel`: A vanilla concept bottleneck model from ["Concept Bottleneck Models"](https://arxiv.org/pdf/2007.04612) (ICML 2020).
- `ResidualConceptBottleneckModel`: A residual concept bottleneck model composed of a set of supervised concepts and a residual unsupervised embedding from ["Promises and Pitfalls of Black-Box Concept Learning Models"](https://arxiv.org/abs/2106.13314) (ICML 2021, workshop).
- `ConceptEmbeddingModel`: A bottleneck of supervised concept embeddings from ["Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off"](https://arxiv.org/abs/2209.09056) (NeurIPS 2022).
- `StochasticConceptBottleneckModel`: A bottleneck of supervised concepts with their covariance matrix ["Stochastic Concept Bottleneck Models"](https://arxiv.org/pdf/2406.19272) (NeurIPS 2024).


## Evaluation APIs

### Datasets

- `TrafficLights`: A dataset loader for traffic scenarios representing road intersections.
- `ToyDataset`: A toy dataset loader. XOR, Trigonometry, and Dot datasets are from ["Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off"](https://arxiv.org/abs/2209.09056) (NeurIPS 2022). The Checkmark dataset is from ["Causal Concept Graph Models: Beyond Causal Opacity in Deep Learning"](https://arxiv.org/abs/2405.16507) (ICLR 2025).
- `CompletenessDataset`: A dataset loader for the completeness score from ["Beyond Concept Bottleneck Models: How to Make Black Boxes Intervenable?"](https://arxiv.org/abs/2401.13544) (NeurIPS 2024).
- `ColorMNISTDataset`: A dataset loader for MNIST Even/Odd where colors act as confounders inspired from ["Explaining Classifiers with Causal Concept Effect (CaCE)"](https://arxiv.org/abs/1907.07165) and ["Interpretable Concept-Based Memory Reasoning"](https://arxiv.org/abs/2407.15527) (NeurIPS 2024).
- `CelebA`: A dataset loader for CelebA dataset with attributes as concepts from ["Deep Learning Face Attributes in the Wild"](https://arxiv.org/abs/1411.7766) (ICCV 2015).
- `CUB`: A dataset loader for CUB dataset to predict bird species from ["The Caltech-UCSD Birds-200-2011 Dataset"](https://authors.library.caltech.edu/records/cvm3y-5hh21).
- `AwA2`: A dataset loader for AwA2 dataset where concepts are animal attributes from ["Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly"](https://arxiv.org/abs/1707.00600) (CVPR 2017).
- `CEBaB`: A dataset loader for CEBaB dataset where concepts describe restaurant reviews from ["CEBaB: Estimating the Causal Effects of Real-World Concepts on NLP Model Behavior"](https://arxiv.org/abs/2205.14140) (NeurIPS 2022).

### Metrics

- `intervention_score`: A score measuring the effectiveness of concept interventions from ["Concept Bottleneck Models"](https://arxiv.org/pdf/2007.04612) (ICML 2020).
- `completeness_score`: A score measuring concept completeness from ["On Completeness-aware Concept-Based Explanations in Deep Neural Networks"](https://arxiv.org/abs/1910.07969) (NeurIPS 2020).
- `cace_score`: A score measuring causal concept effects (CaCE) from ["Explaining Classifiers with Causal Concept Effect (CaCE)"](https://arxiv.org/abs/1907.07165).


Source
------

The source code and minimal working examples can be found on
`GitHub <https://github.com/pyc-team/pytorch_concepts>`__.


.. toctree::
    :caption: API Reference
    :maxdepth: 2

    modules/base
    modules/metrics
    modules/utils
    modules/data/celeba
    modules/data/mnist
    modules/data/toy
    modules/data/traffic
    modules/nn/base
    modules/nn/bottleneck
    modules/nn/functional


.. toctree::
    :caption: Copyright
    :maxdepth: 1

    user_guide/license


Indices and tables
~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Authors
-------

* `Pietro Barbiero <http://www.pietrobarbiero.eu/>`__, Universita' della Svizzera Italiana (CH) and University of Cambridge (UK).
* `Gabriele Ciravegna <https://dbdmg.polito.it/dbdmg_web/gabriele-ciravegna/>`__, Politecnico di Torino (IT).
* `David Debot <https://www.kuleuven.be/wieiswie/en/person/00165387>`__, KU Leuven (BE).
* `Michelangelo Diligenti <https://docenti.unisi.it/en/diligenti>`__, Università degli Studi di Siena (IT).
* `Gabriele Dominici <https://pc.inf.usi.ch/team/gabriele-dominici/>`__, Universita' della Svizzera Italiana (CH).
* `Mateo Espinosa Zarlenga <https://hairyballtheorem.com/>`__, University of Cambridge (UK).
* `Francesco Giannini <https://www.francescogiannini.eu/>`__, Scuola Normale Superiore di Pisa (IT).
* `Giuseppe Marra <https://www.giuseppemarra.com/>`__, KU Leuven (BE).

Licence
-------

Copyright 2024 Pietro Barbiero, Gabriele Ciravegna, David Debot, Michelangelo Diligenti, Gabriele Dominici, Mateo Espinosa Zarlenga, Francesco Giannini, Giuseppe Marra.

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.
