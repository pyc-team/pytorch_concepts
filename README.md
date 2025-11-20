<p align="center">
  <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/doc/_static/img/pyc_logo.png" alt="PyC Logo" width="40%">
</p>

<p align="center">
  <a href="https://pypi.org/project/pytorch-concepts/"><img src="https://img.shields.io/pypi/v/pytorch-concepts?style=for-the-badge" alt="PyPI"></a>
  <a href="https://pepy.tech/project/pytorch-concepts"><img src="https://img.shields.io/pepy/dt/pytorch-concepts?style=for-the-badge" alt="Total downloads"></a>
  <a href="https://codecov.io/gh/pyc-team/pytorch_concepts"><img src="https://img.shields.io/codecov/c/github/pyc-team/pytorch_concepts?style=for-the-badge" alt="Codecov"></a>
  <a href="https://pytorch-concepts.readthedocs.io/"><img src="https://img.shields.io/readthedocs/pytorch-concepts?style=for-the-badge" alt="Documentation Status"></a>
</p>

<p align="center">
  <a href="#get-started">🚀 Getting Started</a> - 
  <a href="https://pytorch-concepts.readthedocs.io/">📚 Documentation</a> - 
  <a href="https://colab.research.google.com/github/pyc-team/pytorch_concepts/blob/master/examples/introductory_notebook.ipynb">💻 Introductory notebook</a>
</p>

# PyC

<img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/docs/source/_static/img/logos/pyc.svg" width="25px" align="center"/> PyC is a library built upon <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/docs/source/_static/img/logos/pytorch.svg" width="25px" align="center"/> PyTorch to easily implement **interpretable and causally transparent deep learning models**.
The library provides primitives for layers (encoders, predictors, special layers), Probabilistic Models, and APIs for running experiments at scale.

The name of the library stands for both
- **PyTorch Concepts**: as concepts are essential building blocks for interpretable deep learning.
- $P(y|C)$: as the main purpose of the library is to support sound probabilistic modeling of the conditional distribution of targets $y$ given concepts $C$.

---

## Get Started

<table>
<tr>
<td width="33%" valign="top">

### 📥 Installation
Learn how to install PyC and set up your environment.

[→ Installation Guide](doc/guides/installation.rst)

</td>
<td width="33%" valign="top">

### ▶️ Using PyC
Explore tutorials and examples to get started with PyC.

[→ Using PyC](doc/guides/using.rst)

</td>
<td width="33%" valign="top">

### 💻 Contributing
Contribute to PyC and help improve the library.

[→ Contributing Guide](doc/guides/contributing.rst)

</td>
</tr>
</table>

---

## Explore Based on Your Background

PyC is designed to accommodate users with different backgrounds and expertise levels.
Pick the best entry point based on your experience:

<table>
<tr>
<td width="50%" valign="top">

### 💻 Pure torch user?
Start from the Low-Level API to build models from basic interpretable layers.

[→ Low-Level API](doc/modules/low_level_api.rst)

</td>
<td width="50%" valign="top">

### 📊 Probabilistic modeling user?
Start from the Mid-Level API to build custom Probabilistic Models.

[→ Mid-Level API](doc/modules/mid_level_api.rst)

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🚀 Just want to use state-of-the-art models out-of-the-box?
Start from the High-Level API to use pre-defined models with one line of code.

[→ High-Level API](doc/modules/high_level_api.rst)

</td>
<td width="50%" valign="top">

### 🧪 No experience with programming?
Use Conceptarium, a no-code framework built on top of PyC for running large-scale experiments on concept-based models.

[→ Conceptarium](doc/modules/conceptarium.rst)

</td>
</tr>
</table>

---

## API Reference

### Main Modules

The main modules of the library are organized into three levels of abstraction: Low-Level API, Mid-Level API, and High-Level API.
These modules allow users with different levels of abstraction to build interpretable models.

<table>
<tr>
<td width="33%" valign="top">

### 🚀 High-Level API
Use out-of-the-box state-of-the-art models with one line of code.

[→ High-Level API](doc/modules/high_level_api.rst)

</td>
<td width="33%" valign="top">

### 📊 Mid-Level API
Build custom interpretable and causally transparent Probabilistic Models.

> ⚠️ **Warning:** This API is still under development and interfaces might change in future releases.

[→ Mid-Level API](doc/modules/mid_level_api.rst)

</td>
<td width="33%" valign="top">

### 🔧 Low-Level API
Build architectures from basic interpretable layers in a plain <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/pytorch.svg" width="20px" align="center"/> PyTorch-like interface.

[→ Low-Level API](doc/modules/low_level_api.rst)

</td>
</tr>
</table>

### Shared Modules

The library also includes shared modules that provide additional functionalities such as loss functions, metrics, and utilities.

<table>
<tr>
<td width="33%" valign="top">

### 🔥 Loss Functions
Various loss functions for concept-based models.

[→ Loss Functions](doc/modules/other_modules.rst)

</td>
<td width="33%" valign="top">

### 📈 Metrics
Evaluation metrics for concept-based models.

[→ Metrics](doc/modules/other_modules.rst)

</td>
<td width="33%" valign="top">

### 📦 Utilities
Helper utilities and tools for concept-based models.

[→ Utilities](doc/modules/other_modules.rst)

</td>
</tr>
</table>

### Extra Modules

Extra modules provide additional APIs for data handling and probability distributions.
These modules have additional dependencies and can be installed separately.

<table>
<tr>
<td width="50%" valign="top">

### 💾 Data API
Access datasets, dataloaders, preprocessing, and data utilities.

[→ Data API](doc/modules/data_api.rst)

</td>
<td width="50%" valign="top">

### ∞ Distributions API
Work with probability distributions for probabilistic modeling.

[→ Distributions API](doc/modules/distributions_api.rst)

</td>
</tr>
</table>

### Conceptarium

Conceptarium is a no-code framework for running large-scale experiments on concept-based models.
The interface is based on configuration files, making it easy to set up and run experiments without writing code.
This framework is intended for benchmarking or researchers in other fields who want to use concept-based models without programming knowledge.

<table>
<tr>
<td valign="top">

### 🧪 Conceptarium
<img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/conceptarium.svg" width="20px" align="center"/> **Conceptarium** is a no-code framework for running large-scale experiments on concept-based models. Built on top of <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/pyc.svg" width="20px" align="center"/> PyC with <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/hydra-head.svg" width="20px" align="center"/> Hydra, <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/lightning.svg" width="20px" align="center"/> PyTorch Lightning, and <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/factors/doc/_static/img/logos/wandb.svg" width="20px" align="center"/> WandB.

[→ Conceptarium Documentation](doc/modules/conceptarium.rst)

</td>
</tr>
</table>

---

# Contributing

- Use the `dev` branch to write and test your contributions locally.
- Make small commits and use ["Gitmoji"](https://gitmoji.dev/) to add emojis to your commit messages.
- Make sure to write documentation and tests for your contributions.
- Make sure all tests pass before submitting the pull request.
- Submit a pull request to the `main` branch.

Thanks to all contributors! 🧡

<a href="https://github.com/pyc-team/pytorch_concepts/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pyc-team/pytorch_concepts" />
</a>

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
