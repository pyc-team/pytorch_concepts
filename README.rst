PyTorch Concepts
======================

PyC (PyTorch Concepts) is a library built upon PyTorch to easily write and train Concept-Based Deep Learning models.


Implemented Modules
-------------------------

**Concept-based layers**:

- ConceptLinear: A linear layer to predict concepts from `"Concept Bottleneck Models" <https://arxiv.org/pdf/2007.04612>`_ (ICML 2020).
- ConceptEmbedding: A layer that generates concept embeddings from `"Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off" <https://arxiv.org/abs/2209.09056>`_ (NeurIPS 2022).

**Metrics**:
- completeness_score: A score measuring concept completeness from `"On Completeness-aware Concept-Based Explanations in Deep Neural Networks" <https://arxiv.org/abs/1910.07969>`_ (NeurIPS 2020).


Contributing
-------------------------

- Use the ``dev`` branch to write and test your contributions locally.
- Make small commits and use `"Gitmoji" <https://gitmoji.dev/>`_ to add emojis to your commit messages.
- Make sure to write documentation and tests for your contributions.
- Make sure all tests pass before submitting the pull request.
- Submit a pull request to the ``main`` branch.
