.. image:: https://github.com/pyc-team/pytorch_concepts/blob/dev/doc/_static/img/pyc_logo_text.svg?sanitize=true
   :scale: 50 %
   :alt: PyC Logo
   :align: center

======================

PyTorch Concepts
======================

PyC (PyTorch Concepts) is a library built upon PyTorch to easily write and train Concept-Based Deep Learning models.


Low-level APIs
-------------------------

**Concept data types** (``pyc.base``):

- ``ConceptTensor``: A subclass of ``torch.Tensor`` which ensures that the tensor has at least two dimensions: batch size and number of concepts. Additionally, it stores and retrieves concepts by name.
- ``ConceptDistribution``: subclass of ``torch.distributions.Distribution`` which ensures that samples drawn from the distribution are ``ConceptTensors``.

**Base concept layers** (``pyc.nn.base``):

- ``ConceptEncoder``: A layer taking as input a common ``tensor`` and producing a ``ConceptTensor`` as output.
- ``ProbabilisticConceptEncoder``: A layer taking as input a common ``tensor`` and producing a (normal) ``ConceptDistribution`` as output.
- ``ConceptScorer``: A layer taking as input a ``ConceptTensor`` with shape ``(batch_size, n_concepts, emb_size)`` and producing as output concept logits with shape ``(batch_size, n_concepts)``.

**Base functions** (``pyc.nn.functional``):

- ``intervene``: A function to intervene on concept scores.
- ``intervene_on_concept_graph``: A function to intervene on a concept adjacency matrix (it can be used to perform do-interventions).
- ``concept_embedding_mixture``: A function to generate a mixture of concept embeddings and concept predictions.

Mid-level APIs
-------------------------

**Concept bottleneck layers** (``torch.nn.bottleneck``):

- ``ConceptBottleneck``: A vanilla concept bottleneck from `"Concept Bottleneck Models" <https://arxiv.org/pdf/2007.04612>`_ (ICML 2020).
- ``ConceptResidualBottleneck``: A residual bottleneck composed of a set of supervised concepts and a residual unsupervised embedding from `"Promises and Pitfalls of Black-Box Concept Learning Models" <https://arxiv.org/abs/2106.13314>`_ (ICML 2021, workshop).
- ``MixConceptEmbeddingBottleneck``: A bottleneck composed of supervised concept embeddings from `"Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off" <https://arxiv.org/abs/2209.09056>`_ (NeurIPS 2022).


Evaluation APIs
-------------------------

**Datasets** (``torch.data``):

- ``ToyDataset``: A toy dataset loader. XOR, Trigonometry, and Dot datasets are from `"Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off" <https://arxiv.org/abs/2209.09056>`_ (NeurIPS 2022). The Checkmark dataset is from `"Causal Concept Embedding Models: Beyond Causal Opacity in Deep Learning" <https://arxiv.org/abs/2405.16507>`_.
- ``CompletenessDataset``: A dataset loader for the completeness score from `"Beyond Concept Bottleneck Models: How to Make Black Boxes Intervenable?" <https://arxiv.org/abs/2401.13544>`_.
- ``ColorMNISTDataset``: A dataset loader for MNIST Even/Odd where colors act as confounders inspired from `"Explaining Classifiers with Causal Concept Effect (CaCE)" <https://arxiv.org/abs/1907.07165>`_ and `"Interpretable Concept-Based Memory Reasoning" <https://arxiv.org/abs/2407.15527>`_.
- ``CelebA``: A dataset loader for CelebA dataset with attributes as concepts from `"Deep Learning Face Attributes in the Wild" <https://arxiv.org/abs/1411.7766>`_ (ICCV 2015).

**Metrics** (``torch.metrics``):

- ``completeness_score``: A score measuring concept completeness from `"On Completeness-aware Concept-Based Explanations in Deep Neural Networks" <https://arxiv.org/abs/1910.07969>`_ (NeurIPS 2020).
- ``cace_score``: A score measuring causal concept effects (CaCE) from `"Explaining Classifiers with Causal Concept Effect (CaCE)" <https://arxiv.org/abs/1907.07165>`_.


Contributing
-------------------------

- Use the ``dev`` branch to write and test your contributions locally.
- Make small commits and use `"Gitmoji" <https://gitmoji.dev/>`_ to add emojis to your commit messages.
- Make sure to write documentation and tests for your contributions.
- Make sure all tests pass before submitting the pull request.
- Submit a pull request to the ``main`` branch.
