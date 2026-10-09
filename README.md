<p align="center">
  <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/pyc_logo.png" alt="PyC Logo" width="40%">
</p>

<p align="center">
  <a href="https://pypi.org/project/pytorch-concepts/"><img src="https://img.shields.io/pypi/v/pytorch-concepts?style=for-the-badge" alt="PyPI"></a>
  <a href="https://pepy.tech/project/pytorch-concepts"><img src="https://img.shields.io/pepy/dt/pytorch-concepts?style=for-the-badge" alt="Total downloads"></a>
  <a href="https://codecov.io/gh/pyc-team/pytorch_concepts"><img src="https://img.shields.io/codecov/c/github/pyc-team/pytorch_concepts?style=for-the-badge" alt="Codecov"></a>
  <a href="https://pytorch-concepts.readthedocs.io/"><img src="https://img.shields.io/readthedocs/pytorch-concepts?style=for-the-badge" alt="Documentation Status"></a>
</p>

<p align="center">
  <a href="https://pytorch-concepts.readthedocs.io/en/latest/guides/installation.html">🚀 Getting Started</a> -
  <a href="https://pytorch-concepts.readthedocs.io/">📚 Documentation</a> -
  <a href="https://pytorch-concepts.readthedocs.io/en/latest/guides/using.html">💻 User guide</a>
</p>

> [!CAUTION]
> Alpha software: PyC is currently under active development.
> Public APIs may change and be unstable between releases.

<img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg" width="20px"> PyC is a library built upon <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg" width="20px"> PyTorch and <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/lightning.svg" width="20px"> PyTorch Lightning to easily implement **interpretable and causally transparent deep learning models**.
The library provides primitives for annotated tensors, interpretable layers, interventions, interpretable probabilistic graphical models, and APIs for running experiments at scale.

The name of the library stands for both
- **PyTorch Concepts**: as concepts are essential building blocks for interpretable deep learning.
- $P(y|C)$: as the main purpose of the library is to support sound probabilistic modeling of the conditional distribution of targets $y$ given concepts $C$.

---

# Quick Start

Install <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg" width="20px"> PyC from [PyPI](https://pypi.org/project/pytorch-concepts/):

```bash
pip install --pre "pytorch-concepts[data]"
```

Use `pip install --pre pytorch-concepts` for core-only (no data dependencies), or see [full installation options](https://pytorch-concepts.readthedocs.io/en/latest/guides/installation.html) for conda setup.

---

# Design Principles

A few basic elements let you name concepts, compute them, train them and act on them. Here is what each one lets you do:

<table>
<tr>
<td width="50%" valign="top">

**Annotations** name the concepts, with their type and size.

</td>
<td width="50%" valign="top">

**Annotated tensors** bind values to concept names.

</td>
</tr>
<tr>
<td valign="top">

```python
import torch_concepts as pyc

ann = pyc.Annotations(
    labels=["red", "shape", "size"],
    types=[
        "binary",
        "categorical",
        "continuous",
    ],
    cardinalities=[1, 3, 1],
)
```

</td>
<td valign="top">

```python
x = torch.randn(8, 5)
c = pyc.AnnotatedTensor(x, ann)

c["red"]     # slice by concept
c.binary()   # slice by type
```

</td>
</tr>
<tr>
<td valign="top">

**Concept layers** map embeddings and/or concepts to concepts.

</td>
<td valign="top">

**Concept losses** score each concept with the loss of its type.

</td>
</tr>
<tr>
<td valign="top">

```python
from torch_concepts.nn import (
    LinearEmbeddingToConcept,
)

layer = LinearEmbeddingToConcept(
    in_embeddings=16,
    out_concepts=ann,
)
emb = torch.randn(8, 16)
out = layer.annotate(layer(emb))
```

</td>
<td valign="top">

```python
from torch import nn
from torch_concepts.nn import ConceptLoss

loss_fn = ConceptLoss(
    binary=nn.BCEWithLogitsLoss(),
    categorical=nn.CrossEntropyLoss(),
    continuous=nn.MSELoss(),
)
loss = loss_fn(out, target)
```

</td>
</tr>
<tr>
<td colspan="2" valign="top">

**Interventions** edit concepts inside a `with` block: a strategy sets the new values, a policy picks where.

</td>
</tr>
<tr>
<td colspan="2" valign="top">

```python
from torch_concepts.nn import DoIntervention, UniformPolicy, intervention

with intervention(layer, DoIntervention(1.0), UniformPolicy(), ["red"]):
    out = layer(emb)  # "red" is set to 1.0
```

</td>
</tr>
</table>

The [quickstart](https://pytorch-concepts.readthedocs.io/en/latest/guides/quickstart.html) trains a first model end to end, and the [user guide](https://pytorch-concepts.readthedocs.io/en/latest/guides/using.html) covers the rest of <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pyc.svg" width="20px"> PyC.

---

# PyC Software Stack
The library is organized to be modular and accessible at different levels of abstraction:
- <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/conceptarium.svg" width="20px"> **Conceptarium (No-code API): applications and benchmarking.** These APIs allow to easily run large-scale experiments by interfacing only with configuration files. Built on top of <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/hydra-head.svg" width="20px"> Hydra and <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/wandb.svg" width="20px"> WandB.
- **High-level APIs: use out-of-the-box models.** These APIs allow to instantiate models with 1 line of code. Models are available both as plain <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg" width="20px"> PyTorch modules (implementing `forward`, allowing custom training loops) and as <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/lightning.svg" width="20px"> PyTorch Lightning modules.
- **Mid-level APIs: interpretable probabilistic graphical models.** These APIs allow to define variables (concepts and embeddings), connect them via conditional distributions parametrized by interpretable layers, and perform probabilistic inference on the resulting graphical model.
- **Low-level APIs: interpretable layers.** These APIs allow to build architectures from basic interpretable layers in a plain <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/logos/pytorch.svg" width="20px"> PyTorch-like interface. These APIs also include annotated tensors, interventions, metrics, losses, and datasets.

<p align="center">
  <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/pyc_software_stack.png" alt="PyC Software Stack" width="90%">
</p>

---

# Contributing
Contributions are welcome! Please check our [contributing guidelines](CONTRIBUTING.md) to get started.

Thanks to all contributors! 🧡

<a href="https://github.com/pyc-team/pytorch_concepts/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pyc-team/pytorch_concepts" />
</a>

## External Contributors

- [Sonia Laguna](https://sonialagunac.github.io/), ETH Zurich (CH).
- [Moritz Vandenhirtz](https://mvandenhi.github.io/), ETH Zurich (CH).

---



# Cite this Library

If you found this library useful for your research article, blog post, or product, we would be grateful if you would cite it using the following bibtex entry:

```
@software{pycteam2025concept,
    author = {Barbiero, Pietro and De Felice, Giovanni and Espinosa Zarlenga, Mateo and Ciravegna, Gabriele and Dominici, Gabriele and De Santis, Francesco and Casanova, Arianna and Debot, David and Giannini, Francesco and Diligenti, Michelangelo and Marra, Giuseppe},
    license = {Apache 2.0},
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
  <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/funding/fwo_kleur.png" alt="FWO - Research Foundation Flanders" height="60" style="margin: 20px;">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/funding/hasler.png" alt="Hasler Foundation" height="60" style="margin: 20px;">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/pyc-team/pytorch_concepts/refs/heads/master/doc/_static/img/funding/snsf.png" alt="SNSF - Swiss National Science Foundation" height="60" style="margin: 20px;">
</p>

