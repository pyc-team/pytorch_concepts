Contributing Guide
=================

We welcome contributions to PyC! This guide will help you contribute effectively.

Thank you for your interest in contributing! The PyC Team welcomes all contributions, whether small bug fixes or major features.

Join Our Community
------------------

Have questions or want to discuss your ideas? Join our Slack community to connect with other contributors and maintainers!

.. image:: https://img.shields.io/badge/Slack-Join%20Us-4A154B?style=for-the-badge&logo=slack
   :target: https://join.slack.com/t/pyc-yu37757/shared_invite/zt-3jdcsex5t-LqkU6Plj5rxFemh5bRhe_Q
   :alt: Slack

How to Contribute
-----------------

1. **Fork the repository** - Create your own fork of the PyC repository on GitHub.
2. **Use the** ``dev`` **branch** - Write and test your contributions locally on the ``dev`` branch.
3. **Create a new branch** - Make a new branch for your specific contribution.
4. **Make your changes** - Implement your changes with clear, descriptive commit messages.
5. **Use Gitmoji** - Add emojis to your commit messages using `Gitmoji <https://gitmoji.dev/>`_ for better clarity.
6. **Write documentation and tests** - Ensure your contributions include appropriate documentation and tests.
7. **Run all tests** - Make sure all tests pass before submitting your pull request.
8. **Submit a Pull Request** - Open a PR to the ``main`` branch describing your changes.

Development Setup
-----------------

Prerequisites
^^^^^^^^^^^^^

- Python 3.9 or higher
- PyTorch (latest stable version)

Installation
^^^^^^^^^^^^

Install PyC and its dependencies:

.. code-block:: bash

   pip install pytorch-concepts

For development, you may want to install in editable mode:

.. code-block:: bash

   git clone https://github.com/pyc-team/pytorch_concepts.git
   cd pytorch_concepts
   pip install -e .

Reporting Issues
----------------

If you find a bug or have a feature request, please open an issue on our `GitHub Issues page <https://github.com/pyc-team/pytorch_concepts/issues>`_ using the appropriate issue template.

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment (Python version, PyTorch version, OS, etc.)

Code Style
----------

Please follow these guidelines when contributing code:

- **PEP 8** - Follow `PEP 8 <https://pep8.org/>`_ style guidelines for Python code.
- **Type hints** - Use type hints where appropriate to improve code clarity.
- **Docstrings** - Write clear docstrings for all public functions and classes.
- **Tests** - Write tests for new features and bug fixes when possible.
- **Documentation** - Update documentation to reflect your changes.

Pull Request Process
--------------------

1. Ensure your code follows the style guidelines above.
2. Update the documentation if you've made changes to the API.
3. Add tests for new functionality.
4. Make sure all tests pass locally.
5. Write a clear PR description explaining what changes you made and why.
6. Link any related issues in your PR description.
7. Wait for review from the maintainers.

Thank You!
----------

Every contributor helps make PyC better. We appreciate your time and effort!

Thanks to all our contributors! 🧡

.. image:: https://contrib.rocks/image?repo=pyc-team/pytorch_concepts
   :target: https://github.com/pyc-team/pytorch_concepts/graphs/contributors
   :alt: Contributors

External Contributors
^^^^^^^^^^^^^^^^^^^^^^

- `Sonia Laguna <https://sonialagunac.github.io/>`_, ETH Zurich (CH).
- `Moritz Vandenhirtz <https://mvandenhi.github.io/>`_, ETH Zurich (CH).