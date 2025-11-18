<p align="center">
  <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/doc/_static/img/pyc_logo.png" alt="PyC Logo" width="40%">
</p>

# PyC

PyC is a library built upon PyTorch to easily implement **interpretable and causally transparent deep learning models**.
The library provides primitives for layers (encoders, predictors, special layers), Probabilistic Models, and APIs for running experiments at scale.

The name of the library stands for both
- **PyTorch Concepts**: as concepts are essential building blocks for interpretable deep learning.
- $P(y|C)$: as the main purpose of the library is to support sound probabilistic modeling of the conditional distribution of targets $y$ given concepts $C$.


- [Quick start](#quick-start)
- [PyC software stack](#pyc-software-stack)
- [Design principles](#design-principles)
  - [Low-level APIs](#low-level-apis)
    - [Objects](#objects)
    - [Layers](#layers)
    - [Models](#models)
    - [Inference](#inference)
  - [Mid-level APIs](#mid-level-apis)
    - [Probabilistic Models](#probabilistic-models)
    - [Inference](#inference-1)
  - [High-level APIs](#high-level-apis)
    - [Objects](#objects-1)
    - [High-level Models](#high-level-models)
  - [No-code APIs](#no-code-apis)
- [Evaluation APIs](#evaluation-apis)
  - [Datasets](#datasets)
  - [Metrics](#metrics)
- [Contributing](#contributing)
- [PyC Book](#pyc-book)
- [Authors](#authors)
- [Licence](#licence)
- [Cite this library](#cite-this-library)


---

# Quick start

You can install PyC along with all its dependencies from
[PyPI](https://pypi.org/project/pytorch-concepts/):

```pip install pytorch-concepts ```

and then import it in your Python scripts as:

```python
import torch_concepts as pyc
```

- Examples: https://github.com/pyc-team/pytorch_concepts/tree/master/examples
- Book: https://pyc-team.github.io/pyc-book/


--- 

# PyC software stack

The library is organized to be modular and accessible at different levels of abstraction:
- **No-code APIs. Use case: applications and benchmarking.** These APIs allow to easily run large-scale highly parallelized and standardized experiments by interfacing with configuration files.
- **High-level APIs. Use case: use out-of-the-box state-of-the-art models.** These APIs allow to instantiate use implemented models with 1 line of code.
- **Mid-level APIs. Use case: build custom interpretable and causally transparent Probabilistic Models.** These APIs allow to build new interpretable probabilistic models and run efficient tensorial probabilistic inference using a Probabilistic Model interface.
- **Low-level APIs. Use case: assemble custom interpretable architectures.** These APIs allow to build architectures from basic interpretable layers in a plain pytorch-like interface. These APIs also include metrics, losses, and datasets.

<p align="center">
  <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/doc/_static/img/pyc_software_stack.png" alt="PyC Software Stack" width="100%">
</p>

---

# Design principles

## Low-level APIs

### Objects
In PyC there are three types of objects: 
- **Embedding**: high-dimensional latent representations shared across all concepts.
- **Exogenous**: high-dimensional latent representations related to a specific concept.
- **Logits**: Concept scores before applying an activation function.

### Layers
There are only three types of layers:
- **Encoders**: layers that map latent representations (embeddings or exogenous) to logits, e.g.:
    ```python
    pyc.nn.ProbEncoderFromEmb(in_features_embedding=10, out_features=3)
    ```

- **Predictors**: layers that map logits (plus optionally latent representations) to other logits.
    ```python
    pyc.nn.HyperLinearPredictor(in_features_logits=10, in_features_exogenous=7, embedding_size=24, out_features=3)
    ```

- **Special layers**: layers that perform special helpful operations such as memory selection:
    ```python
    pyc.nn.MemorySelector(in_features_embedding=10, memory_size=5, embedding_size=24, out_features=3)
    ```
    and graph learners:
    ```python
    wanda = pyc.nn.WANDAGraphLearner(['c1', 'c2', 'c3'], ['task A', 'task B', 'task C'])
    ```

### Models
A model is built as in standard PyTorch (e.g., ModuleDict or Sequential) and may include standard PyTorch layers + PyC layers:
```python
concept_bottleneck_model = torch.nn.ModuleDict({
    'encoder': pyc.nn.ProbEncoderFromEmb(in_features_embedding=10, out_features=3),
    'predictor': pyc.nn.ProbPredictor(in_features_logits=3, out_features=2),
})
```

### Inference
At this API level, there are two types of inference that can be performed:
- **Standard forward pass**: a standard forward pass using the forward method of each layer in the ModuleDict
  ```python
  logits_concepts = concept_bottleneck_model['encoder'](embedding=embedding)
  logits_tasks = concept_bottleneck_model['predictor'](logits=logits_concepts)
  ```

- **Interventions**: interventions are context managers that temporarily modify a layer.
  **Intervention strategies**: define how the intervened layer behaves within an intervention context e.g., we can fix the concept logits to a constant value:
  ```python
  int_strategy = pyc.nn.DoIntervention(model=concept_bottleneck_model["encoder"], constants=-10)
  ```
  **Intervention Policies**: define the order/set of concepts to intervene on e.g., we can intervene on all concepts uniformly:
  ```python
  int_policy = pyc.nn.UniformPolicy(out_features=3)
  ```
  When a forward pass is performed within an intervention context, the intervened layer behaves differently with a cascading effect on all subsequent layers:
  ```python
  with pyc.nn.intervention(policies=int_policy,
                           strategies=int_strategy,
                           target_concepts=[0, 2]) as new_encoder_layer:
      logits_concepts = new_encoder_layer(embedding=embedding)
      logits_tasks = concept_bottleneck_model['predictor'](logits=logits_concepts)
  ```


---


## Mid-level APIs

### Probabilistic Models
At this API level, models are represented as Probabilistic Models where:
- **Variables**: represent random variables in the Probabilistic Model. Variables are defined by their name, parents, and distribution type. For instance we can define a list of three concepts as:
  ```python
  concepts = pyc.Variable(concepts=["c1", "c2", "c3"], parents=[], distribution=torch.distributions.RelaxedBernoulli)
  ```
- **Factors**: represent conditional probability distributions (CPDs) between variables in the Probabilistic Model and are parameterized by PyC layers. For instance we can define a list of three factors for the above concepts as:
  ```python
  concept_factors = pyc.nn.Factor(concepts=["c1", "c2", "c3"], module_class=pyc.nn.ProbEncoderFromEmb(in_features_embedding=10, out_features=3))
  ```
- **Probabilistic Model**: a collection of variables and factors. For instance we can define a ProbabilisticModel as:
  ```python
  probabilistic_model = pyc.nn.ProbabilisticModel(variables=concepts, factors=concept_factors)
  ```

### Inference
Inference is performed using efficient tensorial probabilistic inference algorithms. For instance, we can perform ancestral sampling as:
```python
inference_engine = pyc.nn.AncestralSamplingInference(probabilistic_model=probabilistic_model, graph_learner=wanda, temperature=1.)
predictions = inference_engine.query(["c1"], evidence={'embedding': embedding})
```

---

## High-level APIs

To be completed...

### Objects
- `Annotations`: A class to handle concept and task annotations.
- `ConceptGraph`: A class to handle concept graphs defining dependencies among concepts and tasks.

### High-level Models
- `BipartiteModel`: A handy model to build concept bottleneck models with a bipartite structure where concepts are independent and directly connected to tasks.
- `GraphModel`: A handy model to build concept bottleneck models with an arbitrary directed acyclic graph (DAG) structure among concepts (all labels are represented as concepts).


## Conceptarium: No-code APIs and benchmarking framework

<img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/docs/source/_static/img/logos/conceptarium.svg" width="25px" align="center"/> **Conceptarium** is a high-level experimentation framework for running large-scale experiments on concept-based deep learning models. Built on top of PyC, it provides:
- **Standardized benchmarking datasets**
- **Out-of-the-box concept-based architectures** implemented in <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/docs/source/_static/img/logos/pyc.svg" width="25px" align="center"/> [PyC](https://github.com/pyc-team/pytorch_concepts). All models implemented in Conceptarium can be instantiated with 1 line of code and reused across the board.
- **Configuration-driven experiments**: Use <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/docs/source/_static/img/logos/hydra-head.svg" width="20px" align="center"/> [Hydra](https://hydra.cc/) for flexible YAML-based configuration management and run sequential multi-run experiments with a single command
- **Automated training**: Leverage <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/docs/source/_static/img/logos/lightning.svg" width="20px" align="center"/> [PyTorch Lightning](https://lightning.ai/pytorch-lightning) for streamlined training loops
- **Experiment tracking**: Integrated <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/docs/source/_static/img/logos/wandb.svg" width="20px" align="center"/> [Weights & Biases](https://wandb.ai/) logging for monitoring and reproducibility

**Get Started**: Check out the [Conceptarium README](conceptarium/README.md) for installation, configuration details, and tutorials on implementing custom models and datasets.

**Quick Example**:
```bash
# Clone the repository
git clone https://github.com/pyc-team/pytorch_concepts.git
cd pytorch_concepts/conceptarium

# Run a sweep over models and datasets
python run_experiment.py --config-name your_sweep.yaml
```

Out-of-the-box models include:

| Model                              | Description | Reference |
|------------------------------------| --- |  --- |
| `ConceptBottleneckModel`           | Vanilla concept bottleneck model. | ["Concept Bottleneck Models"](https://arxiv.org/pdf/2007.04612) (ICML 2020) |
| `ResidualConceptBottleneckModel`   | Residual concept bottleneck model with supervised concepts and residual unsupervised embedding. | ["Promises and Pitfalls of Black-Box Concept Learning Models"](https://arxiv.org/abs/2106.13314) (ICML 2021, workshop) |
| `ConceptEmbeddingModel`            | Concept embedding bottleneck model. | ["Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off"](https://arxiv.org/abs/2209.09056) (NeurIPS 2022) |
| `StochasticConceptBottleneckModel` | Stochastic concept bottleneck model with concept covariance matrix. | ["Stochastic Concept Bottleneck Models"](https://arxiv.org/pdf/2406.19272) (NeurIPS 2024) |
| `ConceptGraphModels` | Concept graph models with a causally-transparent bottleneck. | ["Causal Concept Graph Models: Beyond Causal Opacity in Deep Learning"](https://arxiv.org/abs/2405.16507) (ICLR 2025) |
| `CausallyReliableCBM` | Concept graph models with a causal bottleneck aligned with real-world. | ["Causally Reliable Concept Bottleneck Models"](https://arxiv.org/abs/2503.04363) (NeurIPS 2025) |
add more...



Out-of-the-box datasets include:
| Dataset                              | Description | Reference |
|------------------------------------| --- |  --- |
| `BnLearnDataset`           | A collection of synthetic Bayesian Networks from the [bnlearn](https://www.bnlearn.com/bnrepository/) repository. | ["Learning Bayesian Networks with the bnlearn R Package"](https://arxiv.org/abs/0908.3817) |
add more...


<!--| `TrafficLights`       | A dataset loader for traffic scenarios representing road intersections.                                                                                                                                                                                                                                                                                          | N/A                                                                                                                                                                                                                                                                                                  |
| `ToyDataset`          | A toy dataset loader (XOR, Trigonometry, Dot, and Checkmark). | ["Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off"](https://arxiv.org/abs/2209.09056) (NeurIPS 2022) and ["Causal Concept Graph Models: Beyond Causal Opacity in Deep Learning"](https://arxiv.org/abs/2405.16507) (ICLR 2025). |
| `CompletenessDataset` | A dataset loader to assess the impact of concept completeness on model performance.                                                                                                                                                                                                                                                        | ["Beyond Concept Bottleneck Models: How to Make Black Boxes Intervenable?"](https://arxiv.org/abs/2401.13544) (NeurIPS 2024)                                                                                                                                                                        |
| `ColorMNISTDataset`   | A dataset loader for MNIST Even/Odd where colors act as confounders. | ["Explaining Classifiers with Causal Concept Effect (CaCE)"](https://arxiv.org/abs/1907.07165) and ["Interpretable Concept-Based Memory Reasoning"](https://arxiv.org/abs/2407.15527) (NeurIPS 2024). |
| `CelebA`              | A dataset loader for CelebA dataset with attributes as concepts.                                                                                                                                                                                                                                                                                            | ["Deep Learning Face Attributes in the Wild"](https://arxiv.org/abs/1411.7766) (ICCV 2015)                                                                                                                                                                                                                                  |
| `CUB`                 | A dataset loader for CUB dataset to predict bird species. | ["The Caltech-UCSD Birds-200-2011 Dataset"](https://authors.library.caltech.edu/records/cvm3y-5hh21). |
| `AwA2`                | A dataset loader for AwA2 dataset where concepts are animal attributes.                                                                                                                                                                                                                                                                                            | ["Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly"](https://arxiv.org/abs/1707.00600) (CVPR 2017)                                                                                                                                                                                                                                  |-->                                                                                                                                      

<!--
### Metrics

Out-of-the-box metrics include:

| Metric                 | Description                                                                                                                                                                                                                       | Reference                                                                                                                                                                                                                           |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `intervention_score`      | A score measuring the effectiveness of concept interventions from Concept Bottleneck Models.                                                                                                                                    | ["Concept Bottleneck Models"](https://arxiv.org/pdf/2007.04612) (ICML 2020)                                                                                                                                                          |
| `completeness_score`      | A score measuring concept completeness from On Completeness-aware Concept-Based Explanations in Deep Neural Networks.                                                                                                                  | ["On Completeness-aware Concept-Based Explanations in Deep Neural Networks"](https://arxiv.org/abs/1910.07969) (NeurIPS 2020)                                                                                                                                                          |
| `cace_score`              | A score measuring causal concept effects (CaCE) from Explaining Classifiers with Causal Concept Effect (CaCE).                                                                                                                        | ["Explaining Classifiers with Causal Concept Effect (CaCE)"](https://arxiv.org/abs/1907.07165)                                                                                                                                                          |
-->

---

# Contributing

- Use the `dev` branch to write and test your contributions locally.
- Make small commits and use ["Gitmoji"](https://gitmoji.dev/) to add emojis to your commit messages.
- Make sure to write documentation and tests for your contributions.
- Make sure all tests pass before submitting the pull request.
- Submit a pull request to the `main` branch.


---

# PyC Book

You can find further reading materials and tutorials in our book [Concept-based Interpretable Deep Learning in Python](https://pyc-team.github.io/pyc-book/).

---

# Contributors

Thanks to all contributors! 🧡

<a href="https://github.com/pyc-team/pytorch_concepts/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pyc-team/pytorch_concepts" />
</a>

---

# Licence

Copyright 2025 PyC Team.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at: <http://www.apache.org/licenses/LICENSE-2.0>.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and limitations under the License.

---

# Cite this library

If you found this library useful for your research article, blog post, or product, we would be grateful if you would cite it using the following bibtex entry:

```
@software{pycteam2025concept,
    author = {Barbiero, Pietro and De Felice, Giovanni and Espinosa Zarlenga, Mateo and Ciravegna, Gabriele and Dominici, Gabriele and De Santis, Francesco and Casanova, Arianna and Debot, David and Giannini, Francesco and Diligenti, Michelangelo and Marra, Giuseppe},
    license = {MIT},
    month = {3},
    title = {{PyTorch Concepts}},
    url = {https://github.com/pyc-team/pytorch_concepts},
    year = {2025}
}
```
Reference authors: [Pietro Barbiero](http://www.pietrobarbiero.eu/) and [Giovanni De Felice](https://gdefe.github.io/).