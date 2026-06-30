# Contributing to PyC

Thank you for your interest in contributing! The PyC Team welcomes all contributions, from small bug fixes to major features.

## Join Our Community

Have questions or want to discuss your ideas? Join our Slack community to connect with other contributors and maintainers!

[![Slack](https://img.shields.io/badge/Slack-Join%20Us-4A154B?style=for-the-badge&logo=slack)](https://join.slack.com/t/pyc-yu37757/shared_invite/zt-3jdcsex5t-LqkU6Plj5rxFemh5bRhe_Q)

## Ways to Contribute

- **Report a bug or request a feature** — open an issue (see [Reporting Issues](#reporting-issues)).
- **Improve the documentation** — fix typos, clarify docstrings, or expand the guides.
- **Add a new component** — PyC is built to be extended. Each component type has a step-by-step implementation guide in the documentation:
  - [**New Layer**](https://pytorch-concepts.readthedocs.io/en/latest/guides/contributing_layer.html) — a semantics-aware Low-Level API layer.
  - [**New Model**](https://pytorch-concepts.readthedocs.io/en/latest/guides/contributing_model.html) — a concept-based model for the High-Level API.
  - [**New Dataset**](https://pytorch-concepts.readthedocs.io/en/latest/guides/contributing_dataset.html) — a dataset and datamodule with concept annotations.
  - [**New Loss**](https://pytorch-concepts.readthedocs.io/en/latest/guides/contributing_loss.html) — a loss compatible with `ConceptLoss`.
  - [**New Metric**](https://pytorch-concepts.readthedocs.io/en/latest/guides/contributing_metric.html) — a metric compatible with `ConceptMetrics`.

## Development Workflow

1. **Set up your environment** — install PyC in editable mode with all dependencies:

   ```bash
   git clone https://github.com/YOUR_USERNAME/pytorch_concepts.git
   cd pytorch_concepts
   git remote add upstream https://github.com/pyc-team/pytorch_concepts.git
   # create and activate the conda environment (use environment_silicon.yaml on Apple Silicon)
   conda env create -f conceptarium/environment.yaml
   conda activate conceptarium
   # install your local pyc in editable mode
   pip install -e .
   ```

2. **Branch from `dev`** — base your work on the latest upstream `dev`:

   ```bash
   git fetch upstream
   git checkout -b my-feature upstream/dev
   ```

3. **Make your changes** — write clear, descriptive commit messages. We use [Gitmoji](https://gitmoji.dev/) (e.g., `✨ Add new feature`). If you are adding a component, follow its [implementation guide](#ways-to-contribute).

4. **Add tests and docs** — add unit tests, update docstrings and `.rst` files, and verify everything passes locally.

5. **Open a pull request** — push your branch and open a PR targeting `dev` (not `master`).

## Reporting Issues

If you find a bug or have a feature request, please open an issue on our [GitHub Issues page](https://github.com/pyc-team/pytorch_concepts/issues) using the appropriate issue template.

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment (Python version, PyTorch version, OS, etc.)

## Code Style

Please follow these guidelines when contributing code:

| Guideline | Description |
|-----------|-------------|
| **PEP 8** | Follow [PEP 8](https://pep8.org/) style guidelines for Python code. |
| **Type hints** | Use type hints where appropriate to improve code clarity. |
| **Docstrings** | Write clear docstrings for all public functions and classes. |
| **Tests** | Write tests for new features and bug fixes. |
| **Documentation** | Update documentation to reflect your changes. |

## Thank You!

Every contributor helps make PyC better. We appreciate your time and effort!

Thanks to all our contributors! 🧡

<a href="https://github.com/pyc-team/pytorch_concepts/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pyc-team/pytorch_concepts" alt="Contributors" />
</a>

### External Contributors

- [Sonia Laguna](https://sonialagunac.github.io/), ETH Zurich (CH).
- [Moritz Vandenhirtz](https://mvandenhi.github.io/), ETH Zurich (CH).
