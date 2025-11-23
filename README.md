<p align="center">
  <img src="doc/_static/img/pyc_logo.png" alt="PyC Logo" width="40%">
</p>

<p align="center">
  <a href="https://pypi.org/project/pytorch-concepts/"><img src="https://img.shields.io/pypi/v/pytorch-concepts?style=for-the-badge" alt="PyPI"></a>
  <a href="https://pepy.tech/project/pytorch-concepts"><img src="https://img.shields.io/pepy/dt/pytorch-concepts?style=for-the-badge" alt="Total downloads"></a>
  <a href="https://codecov.io/gh/pyc-team/pytorch_concepts"><img src="https://img.shields.io/codecov/c/github/pyc-team/pytorch_concepts?style=for-the-badge" alt="Codecov"></a>
  <a href="https://pytorch-concepts.readthedocs.io/"><img src="https://img.shields.io/readthedocs/pytorch-concepts?style=for-the-badge" alt="Documentation Status"></a>
</p>

<p align="center">
  <a href="https://pytorch-concepts.readthedocs.io/en/factors/guides/installation.html">🚀 Getting Started</a> - 
  <a href="https://pytorch-concepts.readthedocs.io/">📚 Documentation</a> - 
  <a href="https://pytorch-concepts.readthedocs.io/en/factors/guides/using.html">💻 User guide</a>
</p>

<img src="doc/_static/img/logos/pyc.svg" width="20px" align="center"> PyC is a library built upon <img src="doc/_static/img/logos/pytorch.svg" width="20px" align="center"> PyTorch to easily implement **interpretable and causally transparent deep learning models**.
The library provides primitives for layers (encoders, predictors, special layers), probabilistic models, and APIs for running experiments at scale.

The name of the library stands for both
- **PyTorch Concepts**: as concepts are essential building blocks for interpretable deep learning.
- $P(y|C)$: as the main purpose of the library is to support sound probabilistic modeling of the conditional distribution of targets $y$ given concepts $C$.

---

# <img src="doc/_static/img/logos/pyc.svg" width="20px" align="center"> PyC Software Stack
The library is organized to be modular and accessible at different levels of abstraction:
- **No-code APIs. Use case: applications and benchmarking.** These APIs allow to easily run large-scale highly parallelized and standardized experiments by interfacing with configuration files.
- **High-level APIs. Use case: use out-of-the-box state-of-the-art models.** These APIs allow to instantiate use implemented models with 1 line of code.
- **Mid-level APIs. Use case: build custom interpretable and causally transparent probabilistic graphical models.** These APIs allow to build new interpretable probabilistic models and run efficient tensorial probabilistic inference using a probabilistic graphical model interface.
- **Low-level APIs. Use case: assemble custom interpretable architectures.** These APIs allow to build architectures from basic interpretable layers in a plain pytorch-like interface. These APIs also include metrics, losses, and datasets.

<p align="center">
  <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/master/doc/_static/img/pyc_software_stack.png" alt="PyC Software Stack" width="40%">
</p>

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

## External Contributors

- [Sonia Laguna](https://sonialagunac.github.io/), ETH Zurich (CH).
- [Moritz Vandenhirtz](https://mvandenhi.github.io/), ETH Zurich (CH).

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
Reference authors: [Pietro Barbiero](http://www.pietrobarbiero.eu/), [Giovanni De Felice](https://gdefe.github.io/), and [Mateo Espinosa Zarlenga](https://hairyballtheorem.com/).

---

# Funding

This project is supported by the following organizations:

<p align="center">
  <img src="doc/_static/img/funding/fwo_kleur.png" alt="FWO - Research Foundation Flanders" height="60" style="margin: 20px;">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="doc/_static/img/funding/hasler.png" alt="Hasler Foundation" height="60" style="margin: 20px;">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="doc/_static/img/funding/snsf.png" alt="SNSF - Swiss National Science Foundation" height="60" style="margin: 20px;">
</p>

---