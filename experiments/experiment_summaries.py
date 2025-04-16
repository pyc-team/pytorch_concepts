import logging
import os
import pandas as pd
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns


from pytorch_lightning import Trainer
from torch_concepts.nn.models import AVAILABLE_MODELS
from torchvision import transforms
from utils import set_seed, GaussianNoiseTransform



def plot_metric(
    results,
    run_names,
    metric_name,
    save_path=None,
    title="",
    show=False,
):
    """
        Plot the accuracy of all models on the test set.
    """
    sns.barplot(x="model", y=metric_name, data=results)
    plt.xlabel("Model")
    plt.ylabel(metric_name)
    if title:
        plt.title(title, fontsize=24)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()



# def plot_intervenability(dataset):
#     """
#     Plot the intervenability of the models on the test set.
#     For each noise level, plot the test accuracy as a function of the
#     intervention probability. The plot will have as many subplots as the
#     noise levels.
#     """
#     dataset_name = dataset.name
#     # read the results
#     results = pd.read_csv(f"results/{dataset_name}/intervention_results.csv")

#     # map model names to readable names
#     results["model"] = results["model"].map(MODELS_ACRONYMS)

#     # subplots as the noise levels
#     n_noise_levels = len(results["noise_level"].unique())
#     fig, axs = plt.subplots(1, n_noise_levels,
#                             figsize=(4 * n_noise_levels, 4))

#     for i in range(n_noise_levels):
#         noise_level = results["noise_level"].unique()[i]
#         noise_results = results[results["noise_level"] == noise_level]
#         sns.lineplot(x="int_prob", y="test_y_acc", hue="model",
#                      data=noise_results, ax=axs[i])
#         axs[i].set_title(f"Noise level {noise_level} - {dataset_name}")
#         axs[i].set_xlabel("Intervention probability")
#         axs[i].set_ylabel("Test accuracy")

#     plt.tight_layout()
#     plt.savefig(f"results/{dataset_name}/intervenability.png")
#     plt.show()


# def test_intervenability(
#     test_loader,
#     dataset,
#     model_kwargs,
#     int_probs,
#     noise_levels,
#     experiment_config,
#     encoder_fn,
#     num_workers=6,
#     global_params=None,
#     filter_out_regex=None,
#     filter_in_regex=None,
# ):
#     """
#     Test the intervenability of the models by adding noise to the input
#     and then substituting the predicted concept with the right one with
#     increasing probability.
#     """
#     dataset_name = dataset.name
#     results = []

#     # parameters for data, model, and training
#     experiment_config = copy.deepcopy(experiment_config)
#     if 'shared_params' not in experiment_config:
#         experiment_config['shared_params'] = {}
#     # Move all global things into the shared params
#     shared_params = experiment_config['shared_params']
#     for key, vals in experiment_config.items():
#         if key not in ['runs', 'shared_params']:
#             shared_params[key] = vals

#     shared_params['num_workers'] = num_workers

#     experiment_utils.extend_with_global_params(
#         shared_params,
#         global_params or [],
#     )

#     result_dir = experiment_config.get(
#         'result_dir',
#         os.path.join("results", dataset_name),
#     )

#     runs = experiment_config['runs']
#     for split in range(
#         experiment_config.get('start_seed', 0),
#         experiment_config["seeds"],
#     ):
#         for current_config in runs:
#             set_seed(split + 1)
#             # Construct the config for this particular trial
#             trial_config = copy.deepcopy(shared_params)
#             trial_config.update(current_config)
#             # Time to try as many seeds as requested
#             for run_config in experiment_utils.generate_hyperparameter_configs(
#                 trial_config
#             ):
#                 torch.cuda.empty_cache()
#                 run_config = copy.deepcopy(run_config)
#                 run_config['result_dir'] = result_dir
#                 run_config['split'] = split
#                 experiment_utils.evaluate_expressions(run_config, soft=True)

#                 if "run_name" not in run_config:
#                     run_name = (
#                         f"{run_config['model_name']}"
#                         f"{run_config.get('extra_name', '')}"
#                     )
#                     logging.warning(
#                         f'Did not find a run name so using the '
#                         f'name "{run_name}" by default'
#                     )
#                     run_config["run_name"] = run_name
#                 run_name = run_config["run_name"]

#                 # Determine filtering in and filtering out of run
#                 if filter_out_regex:
#                     skip = False
#                     for reg in filter_out_regex:
#                         if re.search(reg, f'{run_name}_seed_{split}'):
#                             logging.info(
#                                 f'Skipping run '
#                                 f'{f"{run_name}_seed_{split}"} as it '
#                                 f'matched filter-out regex {reg}'
#                             )
#                             skip = True
#                             break
#                     if skip:
#                         continue
#                 if filter_in_regex:
#                     found = False
#                     for reg in filter_in_regex:
#                         if re.search(reg, f'{run_name}_seed_{split}'):
#                             found = True
#                             logging.info(
#                                 f'Including run '
#                                 f'{f"{run_name}_seed_{split}"} as it '
#                                 f'did matched filter-in regex {reg}'
#                             )
#                             break
#                     if not found:
#                         logging.info(
#                             f'Skipping run {f"{run_name}_seed_{split}"} as it '
#                             f'did not match any filter-in regexes'
#                         )
#                         continue
#                     model_name  = run_config['model_name']
#                     model_cls = AVAILABLE_MODELS[model_name]
#                     encoder_args = run_config['encoder_args']
#                     encoder = encoder_fn(**encoder_args)
#                     model = model_cls(
#                         encoder=encoder,
#                         concept_names=dataset.concept_names,
#                         task_names=dataset.task_names,
#                         **run_config,
#                     )

#                     filename_pattern = f"{run_name}_seed_{split}"
#                     best_model_path = os.path.join(
#                         result_dir,
#                         f"{filename_pattern}.ckpt",
#                     )
#                     model.load_state_dict(torch.load(best_model_path)['state_dict'])

#                     model.test_intervention = True
#                     # Test the intervenability of the model
#                     for noise_level in noise_levels:
#                         # add noise in the transform of the dataset
#                         transform = transforms.Compose([
#                             transforms.ToTensor(),
#                             GaussianNoiseTransform(std=noise_level),
#                             transforms.Normalize((0.1307,), (0.3081,))
#                         ])
#                         test_loader.dataset.dataset.transform = transform
#                         for int_prob in int_probs:
#                             # set the intervention probability
#                             model.int_prob = int_prob

#                             trainer = Trainer(
#                                 accelerator=run_config.get('accelerator', 'gpu'),
#                                 devices=run_config.get('devices', 1),
#                             )
#                             test_int_result = trainer.test(model, test_loader)[0]

#                             results.append({
#                                 "model": model_name,
#                                 "test_y_acc": test_int_result["test_y_acc"],
#                                 "test_c_acc": test_int_result["test_c_acc"],
#                                 "int_prob": int_prob,
#                                 "noise_level": noise_level,
#                             })

#                             print(
#                                 f"Model {model_name} - Noise {noise_level} "
#                                 f"- Int prob {int_prob}"
#                                 f" - y_acc: {test_int_result['test_y_acc']}"
#                             )

#     results_df = pd.DataFrame(results)
#     results_df.to_csv(
#         os.path.join(
#             result_dir,
#             f"intervention_results.csv",
#         ),
#     )