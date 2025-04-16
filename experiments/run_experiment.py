import argparse
import collections
import copy
import logging
import numpy as np
import os
import pandas as pd
import re
import torch
import torchvision
import yaml


from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_concepts.data.awa2 import AwA2Dataset
from torch_concepts.data.cub import CUBDataset
from torch_concepts.nn.models import AVAILABLE_MODELS
from torch.utils.data import DataLoader, random_split
from utils import set_seed, CustomProgressBar, model_trained

import experiment_utils
import experiment_summaries

def get_run_names(
    experiment_config,
    global_params=None,
    filter_out_regex=None,
    filter_in_regex=None,
    verbose=True,
):
    experiment_config, shared_params, result_dir = experiment_preamble(
        experiment_config=experiment_config,
        num_workers=num_workers,
        global_params=global_params,
    )
    iterator = []
    runs = experiment_config['runs']
    for split in range(
        experiment_config.get('start_seed', 0),
        experiment_config["seeds"],
    ):
        for current_config in runs:
            # Construct the config for this particular trial
            trial_config = copy.deepcopy(shared_params)
            trial_config.update(current_config)
            # Time to try as many seeds as requested
            for run_config in experiment_utils.generate_hyperparameter_configs(
                trial_config
            ):
                torch.cuda.empty_cache()
                run_config = copy.deepcopy(run_config)
                run_config['result_dir'] = result_dir
                run_config['split'] = split
                experiment_utils.evaluate_expressions(run_config, soft=True)

                if "run_name" not in run_config:
                    run_name = (
                        f"{run_config['model_name']}"
                        f"{run_config.get('extra_name', '')}"
                    )
                    logging.warning(
                        f'Did not find a run name so using the '
                        f'name "{run_name}" by default'
                    )
                    run_config["run_name"] = run_name
                run_name = run_config["run_name"]

                # Determine filtering in and filtering out of run
                if filter_out_regex:
                    skip = False
                    for reg in filter_out_regex:
                        if re.search(reg, f'{run_name}_seed_{split}'):
                            if verbose:
                                logging.info(
                                    f'Skipping run '
                                    f'{f"{run_name}_seed_{split}"} as it '
                                    f'matched filter-out regex {reg}'
                                )
                            skip = True
                            break
                    if skip:
                        continue
                if filter_in_regex:
                    found = False
                    for reg in filter_in_regex:
                        if re.search(reg, f'{run_name}_seed_{split}'):
                            found = True
                            if verbose:
                                logging.info(
                                    f'Including run '
                                    f'{f"{run_name}_seed_{split}"} as it '
                                    f'did matched filter-in regex {reg}'
                                )
                            break
                    if not found:
                        if verbose:
                            logging.info(
                                f'Skipping run {f"{run_name}_seed_{split}"} as it '
                                f'did not match any filter-in regexes'
                            )
                        continue
                if run_config.get('y_loss_fn', 'ce') == 'ce':
                    run_config['y_loss_fn'] = torch.nn.CrossEntropyLoss()
                elif run_config.get('y_loss_fn', 'ce') == 'bce':
                    run_config['y_loss_fn'] = torch.nn.BCELoss()
                elif isinstance(run_config['y_loss_fn']):
                    raise ValueError(
                        f'Unsupported loss function "{run_config["y_loss_fn"]}"'
                    )

                # If we made it here, then this is a run we will use!
                iterator.append(
                    (run_name, run_config, split)
                )
    return iterator

def experiment_preamble(experiment_config, num_workers=6, global_params=None):
    # parameters for data, model, and training
    experiment_config = copy.deepcopy(experiment_config)
    if 'shared_params' not in experiment_config:
        experiment_config['shared_params'] = {}
    # Move all global things into the shared params
    shared_params = experiment_config['shared_params']
    for key, vals in experiment_config.items():
        if key not in ['runs', 'shared_params']:
            shared_params[key] = vals

    shared_params['num_workers'] = num_workers

    experiment_utils.extend_with_global_params(
        shared_params,
        global_params or [],
    )

    # Set log level in env variable as this will be necessary for
    # subprocessing
    os.environ['LOGLEVEL'] = os.environ.get(
        'LOGLEVEL',
        logging.getLevelName(logging.getLogger().getEffectiveLevel()),
    )

    # check if results folder exists
    result_dir = experiment_config.get(
        'result_dir',
        "results",
    )
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return experiment_config, shared_params, result_dir

def single_run(
    run_name,
    run_config,
    train_loader,
    val_loader,
    test_loader,
    dataset,
    results_df,
    split,
    logger=None,
):
    model_name  = run_config['model_name']
    model_cls = AVAILABLE_MODELS[model_name]
    encoder_config = run_config['encoder_config']
    encoder = generate_encoder(**encoder_config)
    model = model_cls(
        encoder=encoder,
        concept_names=dataset.concept_names,
        task_names=dataset.task_names,
        **run_config,
    )

    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        dirpath=result_dir,
        filename=f"{run_name}_seed_{split}"
    )
    callbacks = [checkpoint, CustomProgressBar()]
    if run_config.get('early_stopping_config', None) is not None:
        early_stopping_config = run_config['early_stopping_config']
        callbacks.append(
            EarlyStopping(
                monitor=early_stopping_config.get("monitor", "loss"),
                min_delta=early_stopping_config.get("delta", 0.00),
                patience=early_stopping_config.get('patience', 5),
                verbose=early_stopping_config.get("verbose", False),
                mode=early_stopping_config.get("mode", "min"),
            )
        )

    trainer = Trainer(
        max_epochs=run_config["epochs"],
        callbacks=callbacks,
        accelerator=run_config.get('accelerator', 'gpu'),
        devices=run_config.get('devices', 1),
        check_val_every_n_epoch=run_config.get("check_val_every_n_epoch", 5),
        log_every_n_steps=run_config.get("log_every_n_steps", 25),
        logger=logger or False,
    )

    # Train the model
    file = os.path.join(
        result_dir,
        f"{run_name}_seed_{split}.ckpt"
    )
    if not model_trained(
        model,
        model_name,
        file,
        run_config.get("load_results", True),
    ):
        print(f"Training {run_name} with split {split}")
        trainer.fit(model, train_loader, val_loader)

    model.load_state_dict(torch.load(file)['state_dict'])

    test_results = trainer.test(model, test_loader)[0]
    test_results["model"] = run_name
    test_results["split"] = split

    results_df = pd.concat(
        [results_df, pd.DataFrame([test_results])],
        axis=0,
    )
    return results_df

def main(
    train_loader,
    val_loader,
    test_loader,
    dataset,
    all_runs,
    logger=None,
):
    results_df = pd.DataFrame()
    for run_name, run_config, split, in all_runs:
        set_seed(split + 1)
        print(f"[Training {run_name} (trial {split + 1})]")
        print("config:")
        for key, val in run_config.items():
            print(f"\t{key} -> {val}")
        # Split it into a different function call so that memory can be easy
        # cleaned up after a model has been trained
        results_df = single_run(
            run_name=run_name,
            run_config=run_config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            dataset=dataset,
            results_df=results_df,
            logger=logger,
            split=split,
        )
    print(results_df)
    results_df[results_df["model"] == run_name].to_csv(
        os.path.join(result_dir, f"{run_name}.csv")
    )
    results_df.to_csv(os.path.join(result_dir, "results.csv"))
    return results_df

def generate_encoder(**encoder_config):
    if encoder_config['model'] == 'resnet18':
        model = torchvision.models.resnet18(
            pretrained=encoder_config.get('imagenet_pretrained', True),
        )
        latent_dim_size = 512
    elif encoder_config['model'] == 'resnet34':
        model = torchvision.models.resnet34(
            pretrained=encoder_config.get('imagenet_pretrained', True),
        )
        latent_dim_size = 512
    elif encoder_config['model'] == 'resnet50':
        model = torchvision.models.resnet50(
            pretrained=encoder_config.get('imagenet_pretrained', True),
        )
        latent_dim_size = 2048
    else:
        raise ValueError(
            f'Unsupported encoder architecture {encoder_config["model"]}'
        )

    add_linear_layers = encoder_config.get('add_linear_layers', [])
    units = [latent_dim_size] + add_linear_layers + [
        encoder_config.get('latent_dim', 32)
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
        'config',
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
    parser.add_argument(
        '-p',
        '--param',
        action='append',
        nargs=2,
        metavar=('param_name', 'value'),
        help=(
            'Allows the passing of a config param that will overwrite '
            'anything passed as part of the config file itself.'
        ),
        default=[],
    )
    parser.add_argument(
        "--filter_out",
        action='append',
        metavar=('regex'),
        default=None,
        help=(
            "skips runs whose names match the regexes provided via this "
            "argument. These regexes must follow Python's regex syntax."
        ),
    )
    parser.add_argument(
        "--filter_in",
        action='append',
        metavar=('regex'),
        default=None,
        help=(
            "includes only runs whose names match the regexes provided with "
            "this argument. These regexes must follow Python's regex syntax."
        ),
    )
    #################
    ## Build the argparser
    #################

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


    ###################
    ## Load the config
    ###################

    if args.config:
        with open(args.config, "r") as f:
            experiment_config = yaml.load(f, Loader=yaml.FullLoader)


    ###################
    ## Set up the config
    ###################

    filter_out_regex = args.filter_out
    filter_in_regex = args.filter_in
    global_params = args.param
    experiment_config, shared_params, result_dir = experiment_preamble(
        experiment_config=experiment_config,
        num_workers=experiment_config.get('num_workers', 6),
        global_params=global_params,
    )

    ####################
    ## Load the data
    ####################

    dataset_config = experiment_config['dataset_config']
    val_proportion = dataset_config.pop('val_proportion', 0.2)
    other_ds_args = copy.deepcopy(dataset_config)
    other_ds_args.pop('name')
    if dataset_config['name'].lower() == 'awa2':
        train_dataset = AwA2Dataset(split='train', **other_ds_args)
        test_set = AwA2Dataset(split='test', **other_ds_args)
    elif dataset_config['name'].lower() == 'cub':
        train_dataset = CUBDataset(split='train', **other_ds_args)
        test_set = CUBDataset(split='test', **other_ds_args)
    else:
        raise ValueError(
            f"Unsupported dataset {dataset_config['name']}"
        )
    print(f"[Using {train_dataset.name} as a dataset for all runs]")

    # Set split for reproducibility
    batch_size = dataset_config.get('batch_size', 64)
    num_workers = dataset_config.get(
        'num_workers',
        experiment_config.get('num_workers', 6),
    )
    set_seed(dataset_config.get('split', 42))

    # Split the dataset into train, validation and test sets
    train_size = int((1 - val_proportion) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_set, val_set = random_split(
        train_dataset,
        [train_size, val_size],
    )
    train_loader = DataLoader(
        train_set,
        batch_size=dataset_config.get('train_batch_size', batch_size),
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=dataset_config.get('val_batch_size', batch_size),
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=dataset_config.get('test_batch_size', batch_size),
        shuffle=False,
        num_workers=num_workers,
    )

    # Time to check if we will use weights for the concept loss to handle
    # imbalances
    concept_weights = None
    if experiment_config.get('concept_weights', False):
        if hasattr(train_set, 'concept_weights'):
            concept_weights = train_dataset.concept_weights()
        else:
            # Else let us compute it automatically
            attribute_count = np.zeros((len(train_dataset.concept_names),))
            samples_seen = 0
            for (_, c, _) in train_set:
                c = c.cpu().detach().numpy()
                attribute_count += np.sum(c, axis=0)
                samples_seen += c.shape[0]
            concept_weights = samples_seen / attribute_count - 1
    ###################
    ## Determine all models to run
    ###################

    print("Collecting all runs...")
    all_runs = get_run_names(
        experiment_config=experiment_config,
        global_params=global_params,
        filter_out_regex=filter_out_regex,
        filter_in_regex=filter_in_regex,
        verbose=True,
    )
    print(f"[WE WILL TRAIN A TOTAL OF {len(all_runs)} MODELS]")


    # Run the experiments and plot the results
    result_dir = experiment_config.get(
        'result_dir',
        f'results/{train_dataset.name}/'
    )
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    results = main(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset=train_dataset,
        all_runs=all_runs,
    )

    # results = pd.DataFrame()
    # for run_name, _, _ in all_runs:
    #     # read all results from all models and save them
    #     model_results = pd.read_csv(
    #         os.path.join(result_dir, f"{run_name}.csv")
    #     )
    #     results = pd.concat((results, model_results), axis=0)
    # results.to_csv(os.path.join(result_dir, "results.csv"))


    ##################
    ## Plot Basic Metrics
    ##################

    experiment_summaries.plot_metric(
        results=results,
        run_names=[name for name, _, _ in all_runs],
        metric_name="test_y_acc",
        save_path=os.path.join(result_dir, "task_accuracy.png"),
        title=train_dataset.name,
    )
    experiment_summaries.plot_metric(
        results=results,
        run_names=[name for name, _, _ in all_runs],
        metric_name="test_c_avg_auc",
        save_path=os.path.join(result_dir, "task_accuracy.png"),
        title=train_dataset.name,
    )

    ##################
    ## Test interventions
    ##################

    # # Test the intervenability of the models
    # int_probs = experiment_config.get(
    #     'int_probs',
    #     [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    # )
    # noise_levels = experiment_config.get('noise_levels', [0.0])
    # experiment_summaries.test_intervenability(
    #     test_loader=test_loader,
    #     dataset=dataset,
    #     model_kwargs=model_kwargs,
    #     int_probs=int_probs,
    #     noise_levels=noise_levels,
    #     experiment_config=experiment_config,
    #     config=loaded_config,
    #     encoder_fn=encoder_fn,
    # )
    # experiment_summaries.plot_intervenability(dataset)

