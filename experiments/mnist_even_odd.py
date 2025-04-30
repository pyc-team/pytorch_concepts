import os
import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from torch_concepts.data.mnist import MNISTEvenOdd
from torch_concepts.nn.models import AVAILABLE_MODELS, MODELS_ACRONYMS, \
    ConceptExplanationModel
from torch_concepts.utils import get_most_common_expl
from utils import set_seed, CustomProgressBar, GaussianNoiseTransform, \
    model_trained
import matplotlib.pyplot as plt
import seaborn as sns


def main(
    train_loader,
    val_loader,
    test_loader,
    dataset,
    model_kwargs,
    training_kwargs,
):

    dataset_name = dataset.name
    # check if results folder exists
    result_folder = os.path.join("results", dataset_name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    model_kwargs = model_kwargs.copy()
    latent_dim = model_kwargs.pop("latent_dim")
    results_df = pd.DataFrame()
    for model_name, model_cls in AVAILABLE_MODELS.items():
        for seed in range(training_kwargs["seeds"]):
            set_seed(seed)
            # Initialize encoder and model parameters
            encoder = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(dataset.input_dim, latent_dim * 2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(latent_dim * 2, latent_dim),
                torch.nn.LeakyReLU(),
            )
            model = model_cls(
                encoder,
                latent_dim,
                dataset.concept_names,
                dataset.task_names,
                **model_kwargs
            )

            checkpoint = ModelCheckpoint(
                monitor='val_loss',
                save_top_k=1,
                dirpath=result_folder,
                filename=f"{model_name}_seed_{seed}"
            )
            trainer = Trainer(
                max_epochs=training_kwargs["epochs"],
                callbacks=[checkpoint, CustomProgressBar()]
            )

            # Train the model
            file = os.path.join(f"{result_folder}",
                                f"{model_name}_seed_{seed}.ckpt")
            if not model_trained(model, model_name, file,
                                 training_kwargs["load_results"]):
                print(f"Training {model_name} with seed {seed}")
                trainer.fit(model, train_loader, val_loader)
            else:
                print(f"Model {model_name} with seed {seed} already trained")

            model.load_state_dict(torch.load(file)['state_dict'])

            test_results = trainer.test(model, test_loader)[0]
            test_results["model"] = model_name
            test_results["seed"] = seed

            if isinstance(model, ConceptExplanationModel):
                local_explanations = []
                for x, c, y in test_loader:
                    local_explanations += model.get_local_explanations(x)
                print("\nMost common Explanations:")
                print(get_most_common_expl(local_explanations, 5))

            results_df = pd.concat([results_df,
                                    pd.DataFrame([test_results])], axis=0)
        results_df[results_df["model"] == model_name].to_csv(
            result_folder + f"/{model_name}.csv"
        )

    results_df.to_csv(result_folder + "/results.csv")


def plot_test_accuracy(dataset):
    """
        Plot the accuracy of all models on the test set.
    """
    dataset_name = dataset.name
    # read results
    results = pd.read_csv(f"results/{dataset_name}/results.csv")

    # map model names to readable names
    results["model"] = results["model"].map(MODELS_ACRONYMS)

    # plot
    sns.barplot(x="model", y="test_y_acc", data=results)
    plt.xlabel("Model")
    plt.ylabel("Task accuracy")
    plt.title(f"{dataset_name}", fontsize=24)
    plt.tight_layout()
    plt.savefig(f"results/{dataset_name}/task_accuracy.png")
    plt.show()


def plot_concept_accuracy(dataset):
    """
        Plot the concept accuracy of all models on the test set.
    """
    dataset_name = dataset.name
    # read results
    results = pd.read_csv(f"results/{dataset_name}/results.csv")

    # map model names to readable names
    results["model"] = results["model"].map(MODELS_ACRONYMS)

    # plot
    sns.barplot(x="model", y="test_c_avg_auc", data=results)
    plt.xlabel("Model")
    plt.ylabel("Concept accuracy")
    plt.title(f"{dataset_name}", fontsize=24)
    plt.tight_layout()
    plt.savefig(f"results/{dataset_name}/concept_accuracy.png")
    plt.show()


def test_intervenability(
    test_loader,
    dataset,
    model_kwargs,
    int_probs,
    noise_levels,
    training_kwargs,
):
    """
    Test the intervenability of the models by adding noise to the input
    and then substituting the predicted concept with the right one with
    increasing probability.
    """
    dataset_name = dataset.name
    results = []

    model_kwargs = model_kwargs.copy()
    latent_dim = model_kwargs.pop("latent_dim")
    for model_name, model_cls in AVAILABLE_MODELS.items():
        for seed in range(training_kwargs["seeds"]):
            # Define the checkpoint to load the best model
            checkpoint_dir = f"results/{dataset_name}"
            filename_pattern = f"{model_name}_seed_{seed}"
            best_model_path = os.path.join(checkpoint_dir,
                                           f"{filename_pattern}.ckpt")
            encoder = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(dataset.input_dim, latent_dim * 2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(latent_dim * 2, latent_dim),
                torch.nn.LeakyReLU(),
            )
            model = model_cls(
                encoder,
                latent_dim,
                dataset.concept_names,
                dataset.task_names,
                **model_kwargs
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

                    trainer = Trainer()
                    test_int_result = trainer.test(model, test_loader)[0]

                    results.append({
                        "model": model_name,
                        "test_y_acc": test_int_result["test_y_acc"],
                        "test_c_acc": test_int_result["test_c_acc"],
                        "int_prob": int_prob,
                        "noise_level": noise_level,
                    })

                    print(f"Model {model_name} - Noise {noise_level} "
                          f"- Int prob {int_prob}"
                          f" - y_acc: {test_int_result['test_y_acc']}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results/{dataset_name}/intervention_results.csv")


def plot_intervenability(dataset):
    """
    Plot the intervenability of the models on the test set.
    For each noise level, plot the test accuracy as a function of the
    intervention probability. The plot will have as many subplots as the
    noise levels.
    """
    dataset_name = dataset.name
    # read the results
    results = pd.read_csv(f"results/{dataset_name}/intervention_results.csv")

    # map model names to readable names
    results["model"] = results["model"].map(MODELS_ACRONYMS)

    # subplots as the noise levels
    n_noise_levels = len(results["noise_level"].unique())
    fig, axs = plt.subplots(1, n_noise_levels,
                            figsize=(4 * n_noise_levels, 4))

    for i in range(n_noise_levels):
        noise_level = results["noise_level"].unique()[i]
        noise_results = results[results["noise_level"] == noise_level]
        sns.lineplot(x="int_prob", y="test_y_acc", hue="model",
                     data=noise_results, ax=axs[i])
        axs[i].set_title(f"Noise level {noise_level} - {dataset_name}")
        axs[i].set_xlabel("Intervention probability")
        axs[i].set_ylabel("Test accuracy")

    plt.tight_layout()
    plt.savefig(f"results/{dataset_name}/intervenability.png")
    plt.show()


if __name__ == "__main__":
    # Hyperparameters
    training_kwargs = {
        "seeds": 3,
        "epochs": 10,
        "load_results": False,
    }
    model_kwargs = {
        "l_r": 1e-3,
        "latent_dim": 64,
        "embedding_size": 32,
        "class_reg": 0.1,
        "residual_size": 32,
        "memory_size": 5,
        "y_loss_fn": torch.nn.CrossEntropyLoss(),
        "conc_rec_weight": .1,
    }

    print("Running the MNIST Even vs Odd experiment".center(50))
    print("=====================================")
    print("Training kwargs:")
    print(training_kwargs)
    print("Model kwargs:")
    print(model_kwargs)
    print("=====================================")

    # Set seed for reproducibility
    set_seed(42)

    # Load the MNIST dataset
    dataset = MNISTEvenOdd(root='./data', train=True)
    dataset.plot(torch.randint(0, len(dataset), (1,)).item())

    # Split the dataset into train, validation and test sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set, test_set = random_split(dataset,
                                                [train_size,
                                                 val_size // 2, val_size // 2])
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True,
                              num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    # Run the experiments and plot the results
    main(train_loader, val_loader, test_loader, dataset,
         model_kwargs, training_kwargs)

    results = pd.DataFrame()
    for model_name, model_cls in AVAILABLE_MODELS.items():
        # read all results from all models and save them
        model_results = pd.read_csv(
            f"results/{dataset.name}/{model_name}.csv")
        results = pd.concat((results, model_results), axis=0)
    results.to_csv(f"results/{dataset.name}/results.csv")

    plot_test_accuracy(dataset)
    plot_concept_accuracy(dataset)

    # Test the intervenability of the models
    int_probs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0]
    test_intervenability(test_loader, dataset, model_kwargs,
                         int_probs, noise_levels, training_kwargs)
    plot_intervenability(dataset)

