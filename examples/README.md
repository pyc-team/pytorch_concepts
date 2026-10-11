# Examples

Each example is a short script that builds, trains and evaluates one thing.
They follow the three levels of the library, plus data loading:

- **`low_level/`**: PyC layers used as plain PyTorch modules.
- **`mid_level/`**: probabilistic graphical models: variables, CPDs and inference engines.
- **`high_level/`**: ready-made models, trained with PyTorch or Lightning.
- **`data/`**: datasets and concept generation.

Run an example with the `data` extra installed (`pip install -e ".[data]"`), for instance:

```bash
python examples/low_level/02_concept_bottleneck_model.py
```

Examples are split into cells by `# %%` markers, so they also run cell by cell in VS Code, PyCharm or
Jupyter. Datasets are downloaded or generated on first use into `DATA_ROOT`, defined in
[`torch_concepts/env.py`](../torch_concepts/env.py): `~/.cache/pyc`, or the folder in `PYC_CACHE`.

## Low level

| Example | What it shows |
| --- | --- |
| [01_annotations](low_level/01_annotations.py) | Naming concepts and selecting tensor columns by name or type |
| [02_concept_bottleneck_model](low_level/02_concept_bottleneck_model.py) | A CBM from PyC layers, and why a linear task head cannot learn XOR |
| [03_interventions](low_level/03_interventions.py) | Correcting concepts at test time, and asking what-if questions |
| [04_concept_embedding_model](low_level/04_concept_embedding_model.py) | Concept embeddings solve the task the CBM could not |
| [05_hypernetwork_predictor](low_level/05_hypernetwork_predictor.py) | A model whose logic is a few readable rules over concepts |
| [06_custom_layer](low_level/06_custom_layer.py) | Writing a new concept layer |
| [07_concept_whitening](low_level/07_concept_whitening.py) | Aligning latent axes with concepts (CelebA, ~1.4 GB) |
| [08_tcav](low_level/08_tcav.py) | Testing a trained classifier with concept activation vectors (CelebA) |

## Mid level

| Example | What it shows |
| --- | --- |
| [01_concept_bottleneck_model](mid_level/01_concept_bottleneck_model.py) | The CBM as a Bayesian network, with a plate of concepts |
| [02_concept_embedding_model](mid_level/02_concept_embedding_model.py) | The CEM as a Bayesian network |
| [03_evidence_and_interventions](mid_level/03_evidence_and_interventions.py) | Observing concepts as evidence, or intervening on them with a policy |
| [04_causal_effects](mid_level/04_causal_effects.py) | Seeing versus doing: conditioning and the do-operator give different answers |
| [05_learning_a_bayesian_network](mid_level/05_learning_a_bayesian_network.py) | Learning the CPDs of a known graph (ASIA) from data |
| [06_inference_engines](mid_level/06_inference_engines.py) | Forward, sampling, message-passing and exact engines on one diagnosis query |
| [07_markov_network](mid_level/07_markov_network.py) | An undirected model where observing one concept informs another |

## High level

All on Color-MNIST (`digit` and `color` as concepts, `parity` as task), except where noted.

| Example | What it shows |
| --- | --- |
| [01_concept_bottleneck_model](high_level/01_concept_bottleneck_model.py) | A ready-made CBM trained with a plain PyTorch loop |
| [02_lightning_training](high_level/02_lightning_training.py) | The same model with Lightning, losses and metrics |
| [03_interventions](high_level/03_interventions.py) | Revealing the true digit fixes parity; forcing a digit changes the prediction |
| [04_training_modes](high_level/04_training_modes.py) | Joint, independent and hard-concept training |
| [05_model_comparison](high_level/05_model_comparison.py) | A black box, a CBM and a CEM, trained the same way |
| [06_losses](high_level/06_losses.py) | Composing losses by concept type, by concept group, and custom terms |
| [07_metrics](high_level/07_metrics.py) | Configuring metrics per concept type |
| [08_causally_reliable_cbm](high_level/08_causally_reliable_cbm.py) | A CBM whose concepts follow a causal graph (ASIA) |
| [09_concept_bottleneck_vae](high_level/09_concept_bottleneck_vae.py) | Generating images from concepts |
| [10_pretrained_backbone](high_level/10_pretrained_backbone.py) | A frozen or fine-tuned image backbone inside the model (CelebA) |

## Data

| Example | What it shows |
| --- | --- |
| [01_toy_datasets](data/01_toy_datasets.py) | The synthetic datasets and their concept graphs |
| [02_custom_dag_dataset](data/02_custom_dag_dataset.py) | Sampling a dataset from your own Bayesian network |
| [03_color_mnist](data/03_color_mnist.py) | Color-MNIST with and without a color shortcut |
| [04_celeba_embeddings](data/04_celeba_embeddings.py) | Computing backbone embeddings once and training on them (CelebA) |
| [05_label_free_concepts](data/05_label_free_concepts.py) | Generating concepts with an LLM and CLIP (needs an LLM API key) |

`steerling/` holds examples for the Steerling language model (needs the `steerling` package and ~16 GB of
weights).
