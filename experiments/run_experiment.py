import argparse
import collections
import logging
import os
import pandas as pd
import torch
import torchvision
import yaml


from mnist_addition import \
    plot_intervenability, plot_concept_accuracy, plot_test_accuracy
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_concepts.data.cub import CUBDataset
from torch_concepts.nn.models import AVAILABLE_MODELS
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from utils import \
    set_seed, CustomProgressBar, GaussianNoiseTransform, model_trained


def main(
    train_loader,
    val_loader,
    test_loader,
    dataset,
    model_kwargs,
    training_kwargs,
    encoder_fn,
    config,
):

    dataset_name = dataset.name
    # check if results folder exists
    result_dir = config.get(
        'result_dir',
        os.path.join("results", dataset_name),
    )
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Initialize encoder and model parameters
    encoder = encoder_fn(**model_kwargs)

    results_df = pd.DataFrame()
    models_to_train = loaded_config.get(
        'models_to_train',
        sorted([n for (n, _) in AVAILABLE_MODELS.items()]),
    )
    print("Training models:", models_to_train)
    for model_name in models_to_train:
        model_cls = AVAILABLE_MODELS[model_name]
        for seed in range(training_kwargs["seeds"]):
            set_seed(seed)
            model = model_cls(
                encoder,
                model_kwargs["latent_dim"],
                dataset.concept_names,
                dataset.task_names,
                l_r=model_kwargs["l_r"],
                residual_size=model_kwargs["residual_size"],
                embedding_size=model_kwargs["embedding_size"],
                memory_size=model_kwargs["memory_size"],
                y_loss_fn=model_kwargs["y_loss_fn"],
                class_reg=model_kwargs.get('class_reg', 0.1),
                concept_reg=model_kwargs.get('concept_reg', 1),
            )

            checkpoint = ModelCheckpoint(
                monitor='val_loss',
                save_top_k=1,
                dirpath=result_dir,
                filename=f"{model_name}_seed_{seed}"
            )
            trainer = Trainer(
                max_epochs=training_kwargs["epochs"],
                callbacks=[checkpoint, CustomProgressBar()],
                accelerator=config.get('accelerator', 'gpu'),
                devices=config.get('devices', 1),
                check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
                log_every_n_steps=config.get("check_val_every_n_epoch", 25)
            )

            # Train the model
            file = os.path.join(
                result_dir,
                f"{model_name}_seed_{seed}.ckpt"
            )
            if (not model_trained(model, model_name, file,
                                 training_kwargs["load_results"])
                    or model_name == "ConceptMemoryReasoning (embedding)"):
                print(f"Training {model_name} with seed {seed}")
                trainer.fit(model, train_loader, val_loader)

            model.load_state_dict(torch.load(file)['state_dict'])

            test_results = trainer.test(model, test_loader)[0]
            test_results["model"] = model_name
            test_results["seed"] = seed

            results_df = pd.concat([results_df,
                                    pd.DataFrame([test_results])], axis=0)
        results_df[results_df["model"] == model_name].to_csv(
            os.path.join(result_dir, f"{model_name}.csv")
        )

    results_df.to_csv(os.path.join(result_dir, "results.csv"))


def test_intervenability(
    test_loader,
    dataset,
    model_kwargs,
    int_probs,
    noise_levels,
    training_kwargs,
    encoder_fn,
    config,
    result_dir='results',
):
    """
    Test the intervenability of the models by adding noise to the input
    and then substituting the predicted concept with the right one with
    increasing probability.
    """
    dataset_name = dataset.name
    results = []

    for model_name, model_cls in AVAILABLE_MODELS.items():
        for seed in range(training_kwargs["seeds"]):
            # Define the checkpoint to load the best model
            filename_pattern = f"{model_name}_seed_{seed}"
            best_model_path = os.path.join(
                result_dir,
                f"{filename_pattern}.ckpt",
            )
            encoder = encoder_fn(**model_kwargs)
            model = model_cls(
                encoder,
                model_kwargs["latent_dim"],
                dataset.concept_names,
                dataset.task_names,
                class_reg=model_kwargs["class_reg"],
                residual_size=model_kwargs["residual_size"],
                embedding_size=model_kwargs["embedding_size"],
                memory_size=model_kwargs["memory_size"],
                y_loss_fn=model_kwargs["y_loss_fn"],
            )
            model.load_state_dict(torch.load(best_model_path)['state_dict'])

            model.test_intervention = True
            # Test the intervenability of the model
            for noise_level in noise_levels:
                # add noise in the transform of the dataset
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    GaussianNoiseTransform(std=noise_level),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                test_loader.dataset.dataset.transform = transform
                for int_prob in int_probs:
                    # set the intervention probability
                    model.int_prob = int_prob

                    trainer = Trainer(
                        accelerator=config.get('accelerator', 'gpu'),
                        devices=config.get('devices', 1),
                    )
                    test_int_result = trainer.test(model, test_loader)[0]

                    results.append({
                        "model": model_name,
                        "test_y_acc": test_int_result["test_y_acc"],
                        "test_c_acc": test_int_result["test_c_acc"],
                        "int_prob": int_prob,
                        "noise_level": noise_level,
                    })

                    print(
                        f"Model {model_name} - Noise {noise_level} "
                        f"- Int prob {int_prob}"
                        f" - y_acc: {test_int_result['test_y_acc']}"
                    )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(
            result_dir,
            f"intervention_results.csv",
        ),
    )


def encoder_fn(**model_kwargs):
    model = torchvision.models.resnet18(
        pretrained=model_kwargs.get('imagenet_pretrained', True),
    )
    latent_dim_size = 512
    add_linear_layers = model_kwargs.get('add_linear_layers', [])
    units = [latent_dim_size] + add_linear_layers + [
        model_kwargs.get('latent_dim', 32)
    ]
    layers = []
    for i in range(1, len(units)):
        layers.append((f"nonlin_{i}", torch.nn.LeakyReLU()))
        layers.append((f"outlayer_{i}", torch.nn.Linear(units[i-1], units[i])))
    model.fc = torch.nn.Sequential(collections.OrderedDict(layers))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            'Runs the set of experiments of CBM-like models in the provided '
            'configuration file.'
        ),
    )
    parser.add_argument(
        '--config',
        '-c',
        help=(
            "YAML file with the configuration for the set of experiments to "
            "run."
        ),
        metavar="config.yaml",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="starts debug mode in our program.",
    )


    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    # Hyperparameters
    if args.config:
        with open(args.config, "r") as f:
            loaded_config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        loaded_config = dict(
            dataset='cub',
            training_kwargs=dict(
                seeds=3,
                epochs=75,
                load_results=False,
            ),
            model_kwargs=dict(
                l_r=1e-3,
                latent_dim=64,
                embedding_size=32,
                class_reg=0.1,
                residual_size=32,
                memory_size=20,
                y_loss_fn=torch.nn.CrossEntropyLoss(),
            ),
        )

    # Load the data
    if loaded_config['dataset'].lower() == 'cub':
        dataset = CUBDataset(root='/homes/me466/data/CUB200/', split='train')
    else:
        raise ValueError(
            f"Unsupported dataset {loaded_config['dataset']}"
        )
    result_dir = loaded_config.get(
        'result_dir',
        f'results/{dataset.name}/'
    )
    training_kwargs = loaded_config['training_kwargs']
    model_kwargs = loaded_config['model_kwargs']
    if model_kwargs.get('y_loss_fn', 'ce') == 'ce':
        model_kwargs['y_loss_fn'] = torch.nn.CrossEntropyLoss()
    elif model_kwargs.get('y_loss_fn', 'ce') == 'bce':
        model_kwargs['y_loss_fn'] = torch.nn.BCELoss()
    elif isinstance(model_kwargs['y_loss_fn']):
        raise ValueError(
            f'Unsupported loss function "{model_kwargs["y_loss_fn"]}"'
        )

    print(f"Running the {dataset.name} experiment")
    print("=====================================")
    print("Training kwargs:")
    print(training_kwargs)
    print("Model kwargs:")
    print(model_kwargs)
    print("=====================================")

    # Set seed for reproducibility
    batch_size = loaded_config.get('batch_size', 64)
    num_workers = loaded_config.get('num_workers', 8)
    set_seed(loaded_config.get('seed', 42))


    # Split the dataset into train, validation and test sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size // 2, val_size // 2],
    )
    train_loader = DataLoader(
        train_set,
        batch_size=loaded_config.get('train_batch_size', batch_size),
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=loaded_config.get('val_batch_size', batch_size),
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=loaded_config.get('test_batch_size', batch_size),
        shuffle=False,
        num_workers=num_workers,
    )

    # Run the experiments and plot the results
    main(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset=dataset,
        model_kwargs=model_kwargs,
        training_kwargs=training_kwargs,
        encoder_fn=encoder_fn,
        config=loaded_config,
    )

    results = pd.DataFrame()
    models_to_train = loaded_config.get(
        'models_to_train',
        sorted([n for (n, _) in AVAILABLE_MODELS.items()]),
    )
    for model_name in models_to_train:
        model_cls = AVAILABLE_MODELS[model_name]
        # read all results from all models and save them
        model_results = pd.read_csv(
            os.path.join(result_dir, f"{model_name}.csv")
        )
        results = pd.concat((results, model_results), axis=0)
    results.to_csv(os.path.join(result_dir, "results.csv"))

    plot_test_accuracy(dataset)
    plot_concept_accuracy(dataset)

    # Test the intervenability of the models
    int_probs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    noise_levels = [0.0]
    test_intervenability(
        test_loader=test_loader,
        dataset=dataset,
        model_kwargs=model_kwargs,
        int_probs=loaded_config.get('int_probs', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        noise_levels=noise_levels,
        training_kwargs=training_kwargs,
        config=loaded_config,
        encoder_fn=encoder_fn,
    )
    plot_intervenability(dataset)

