![PyC Logo](https://raw.githubusercontent.com/pyc-team/pytorch_concepts/dev/doc/_static/img/pyc_logo_text.svg)

# PyTorch Concepts

PyC (PyTorch Concepts) is a library built upon PyTorch to easily write and train Concept-Based Deep Learning models.

## Low-level APIs

**Concept data types** (`pyc.base`):

- `ConceptTensor`: A subclass of `torch.Tensor` which stores and retrieves concepts by name.
- `ConceptDistribution`: Subclass of `torch.distributions.Distribution` which ensures that samples drawn from the distribution are `ConceptTensors`.

**Base concept layers** (`pyc.nn.base`):

- `ConceptLayer`: A layer taking as input a common `tensor` and producing a `ConceptTensor` as output.
- `ProbabilisticConceptLayer`: A layer taking as input a common `tensor` and producing a (normal) `ConceptDistribution` as output.
- `ConceptMemory`: A layer with a set of embeddings which can be decoded into different concept states.

**Base functions** (`pyc.nn.functional`):

- `intervene`: A function to intervene on concept scores.
- `intervene_on_concept_graph`: A function to intervene on a concept adjacency matrix (it can be used to perform do-interventions).
- `concept_embedding_mixture`: A function to generate a mixture of concept embeddings and concept predictions.

## Mid-level APIs

**Concept bottleneck layers** (`pyc.nn.bottleneck`):

- `ConceptBottleneck`: A vanilla concept bottleneck from ["Concept Bottleneck Models"](https://arxiv.org/pdf/2007.04612) (ICML 2020).
- `ConceptResidualBottleneck`: A residual bottleneck composed of a set of supervised concepts and a residual unsupervised embedding from ["Promises and Pitfalls of Black-Box Concept Learning Models"](https://arxiv.org/abs/2106.13314) (ICML 2021, workshop).
- `ConceptEmbeddingBottleneck`: A bottleneck composed of supervised concept embeddings from ["Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off"](https://arxiv.org/abs/2209.09056) (NeurIPS 2022).

## Evaluation APIs

**Datasets** (`pyc.data`):

- `ToyDataset`: A toy dataset loader. XOR, Trigonometry, and Dot datasets are from ["Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off"](https://arxiv.org/abs/2209.09056) (NeurIPS 2022). The Checkmark dataset is from ["Causal Concept Embedding Models: Beyond Causal Opacity in Deep Learning"](https://arxiv.org/abs/2405.16507).
- `CompletenessDataset`: A dataset loader for the completeness score from ["Beyond Concept Bottleneck Models: How to Make Black Boxes Intervenable?"](https://arxiv.org/abs/2401.13544).
- `ColorMNISTDataset`: A dataset loader for MNIST Even/Odd where colors act as confounders inspired from ["Explaining Classifiers with Causal Concept Effect (CaCE)"](https://arxiv.org/abs/1907.07165) and ["Interpretable Concept-Based Memory Reasoning"](https://arxiv.org/abs/2407.15527).
- `CelebA`: A dataset loader for CelebA dataset with attributes as concepts from ["Deep Learning Face Attributes in the Wild"](https://arxiv.org/abs/1411.7766) (ICCV 2015).

**Metrics** (`pyc.metrics`):

- `intervention_score`: A score measuring the effectiveness of concept interventions from ["Concept Bottleneck Models"](https://arxiv.org/pdf/2007.04612) (ICML 2020).
- `completeness_score`: A score measuring concept completeness from ["On Completeness-aware Concept-Based Explanations in Deep Neural Networks"](https://arxiv.org/abs/1910.07969) (NeurIPS 2020).
- `cace_score`: A score measuring causal concept effects (CaCE) from ["Explaining Classifiers with Causal Concept Effect (CaCE)"](https://arxiv.org/abs/1907.07165).

## Contributing

- Use the `dev` branch to write and test your contributions locally.
- Make small commits and use ["Gitmoji"](https://gitmoji.dev/) to add emojis to your commit messages.
- Make sure to write documentation and tests for your contributions.
- Make sure all tests pass before submitting the pull request.
- Submit a pull request to the `main` branch.

## PyC Book

You can find further reading materials and tutorials in our book [Concept-based Interpretable Deep Learning in Python](https://pyc-team.github.io/pyc-book/).

## Authors

- [Pietro Barbiero](http://www.pietrobarbiero.eu/), Universita' della Svizzera Italiana (CH) and University of Cambridge (UK).
- [Gabriele Ciravegna](https://dbdmg.polito.it/dbdmg_web/gabriele-ciravegna/), Politecnico di Torino (IT).
- [David Debot](https://www.kuleuven.be/wieiswie/en/person/00165387), KU Leuven (BE).
- [Michelangelo Diligenti](https://docenti.unisi.it/en/diligenti), Università degli Studi di Siena (IT).
- [Gabriele Dominici](https://pc.inf.usi.ch/team/gabriele-dominici/), Universita' della Svizzera Italiana (CH).
- [Mateo Espinosa Zarlenga](https://hairyballtheorem.com/), University of Cambridge (UK).
- [Francesco Giannini](https://www.francescogiannini.eu/), Scuola Normale Superiore di Pisa (IT).
- [Giuseppe Marra](https://www.giuseppemarra.com/), KU Leuven (BE).

## Licence

Copyright 2024 Pietro Barbiero, Gabriele Ciravegna, David Debot, Michelangelo Diligenti, Gabriele Dominici, Mateo Espinosa Zarlenga, Francesco Giannini, Giuseppe Marra.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at: <http://www.apache.org/licenses/LICENSE-2.0>.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and limitations under the License.
