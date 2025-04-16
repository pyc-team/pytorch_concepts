################################################################################
## Taken from Espinosa Zarlenga et al.
## https://github.com/mateoespinosa/cem/blob/mateo/probcbm/experiments/experiment_utils.py
## and https://github.com/mateoespinosa/cem/blob/mateo/probcbm/cem/train/utils.py
################################################################################

import copy
import itertools
import logging
import numpy as np
import os
import re
import torch

from collections import defaultdict
from pathlib import Path
from prettytable import PrettyTable

################################################################################
## HELPER FUNCTIONS
################################################################################

def _to_val(x):
    if len(x) >= 2 and (x[0] == "[") and (x[-1] == "]"):
        return eval(x)
    try:
        return int(x)
    except ValueError:
        # Then this is not an int
        pass

    try:
        return float(x)
    except ValueError:
        # Then this is not an float
        pass

    if x.lower().strip() in ["true"]:
        return True
    if x.lower().strip() in ["false"]:
        return False

    return x


def extend_with_global_params(config, global_params):
    for param_path, value in global_params:
        var_names = list(map(lambda x: x.strip(), param_path.split(".")))
        current_obj = config
        for path_entry in var_names[:-1]:
            if path_entry not in config:
                current_obj[path_entry] = {}
            current_obj = current_obj[path_entry]
        current_obj[var_names[-1]] = _to_val(value)

def determine_rerun(
    config,
    rerun,
    run_name,
    split,
):
    if rerun:
        return True
    reruns = config.get('reruns', [])
    if "RERUNS" in os.environ:
        reruns += os.environ['RERUNS'].split(",")
    for variant in [
        run_name,
        run_name + f"_split_{split}",
        run_name + f"_fold_{split}",
    ]:
        if variant in reruns:
            return True
    return False

def get_mnist_extractor_arch(input_shape, in_channels):
    def c_extractor_arch(output_dim):
        intermediate_maps = 16
        output_dim = output_dim or 128
        first_dim_out = ((input_shape[2] - (2-1) - 1) // 2) + 1
        first_dim_out = ((first_dim_out - (2-1) - 1) // 2) + 1
        first_dim_out = ((first_dim_out - (2-1) - 1) // 2) + 1
        first_dim_out = ((first_dim_out - (3-1) - 1) // 3) + 1

        second_dim_out = ((input_shape[3] - (2-1) - 1) // 2) + 1
        second_dim_out = ((second_dim_out - (2-1) - 1) // 2) + 1
        second_dim_out = ((second_dim_out - (2-1) - 1) // 2) + 1
        second_dim_out = ((second_dim_out - (3-1) - 1) // 3) + 1
        out_shape = (first_dim_out, second_dim_out)
        return torch.nn.Sequential(*[
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=intermediate_maps,
                kernel_size=(3,3),
                padding='same',
            ),
            torch.nn.BatchNorm2d(num_features=intermediate_maps),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(
                in_channels=intermediate_maps,
                out_channels=intermediate_maps,
                kernel_size=(3,3),
                padding='same',
            ),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.BatchNorm2d(num_features=intermediate_maps),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=intermediate_maps,
                out_channels=intermediate_maps,
                kernel_size=(3,3),
                padding='same',
            ),
            torch.nn.BatchNorm2d(num_features=intermediate_maps),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(
                in_channels=intermediate_maps,
                out_channels=intermediate_maps,
                kernel_size=(3,3),
                padding='same',
            ),
            torch.nn.BatchNorm2d(num_features=intermediate_maps),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d((3, 3)),
            torch.nn.Flatten(),
            torch.nn.Linear(
                np.prod(out_shape) * intermediate_maps,
                output_dim,
            ),
        ])
    return c_extractor_arch

def get_metric_from_dict(results, method, metric):
    vals = []
    for _, metric_keys in results.items():
        for candidate_method, metric_map in metric_keys.items():
            if method != candidate_method:
                continue
            for metric_name, val in metric_map.items():
                if metric_name == metric:
                    vals.append(val)
    return vals

def perform_model_selection(
    results,
    model_groupings,
    selection_metric,
    name_filters=None,
    included_models=None,
):
    name_filters = name_filters or []
    if included_models:
        exclude_model_filter = lambda x: x not in included_models
    else:
        exclude_model_filter = lambda x: False

    un_select_method = lambda x: np.any([exclude_model_filter(x)] + [
        re.search(reg, x) for reg in name_filters
    ])
    new_results = defaultdict(lambda: defaultdict(dict))
    method_names = set()
    for _, metric_keys in results.items():
        for method_name, metric_map in metric_keys.items():
            # Make sure we do not select a method that has been filtered out
            if un_select_method(method_name):
                continue
            method_names.add(method_name)

    selection_result = {}
    for group_pattern, group_name in model_groupings:
        selected_methods = [
            name for name in method_names
            if re.search(group_pattern, name)
        ]
        selected_values = [
            (
                method_name,
                np.mean(
                    get_metric_from_dict(
                        results,
                        method_name,
                        selection_metric,
                    ),
                ),
            )
            for method_name in selected_methods
        ]
        selected_values = [
            (method_name, vals)
            for (method_name, vals) in selected_values
            if not np.isnan(vals)
        ]
        if selected_values:
            selected_values.sort(key=lambda x: -x[1])
            selected_method = selected_values[0][0]
            group_name = group_name or selected_method
            selection_result[group_name] = selected_method
            for fold, metric_keys in results.items():
                new_results[fold][group_name] = copy.deepcopy(
                    results[fold][selected_method]
                )
    return dict(new_results), selection_result


def perform_averaging(
    results,
    model_groupings,
    name_filters=None,
):
    name_filters = name_filters or []
    un_select_method = lambda x: np.any([
        re.search(reg, x) for reg in name_filters
    ])
    new_results = defaultdict(lambda: defaultdict(dict))
    method_names = set()
    metric_names = set()
    for _, metric_keys in results.items():
        for method_name, metric_map in metric_keys.items():
            # Make sure we do not select a method that has been filtered out
            if un_select_method(method_name):
                continue
            method_names.add(method_name)
            for metric_name, _ in metric_map.items():
                metric_names.add(metric_name)

    for group_pattern, group_name in model_groupings:
        selected_methods = [
            name for name in method_names
            if re.search(group_pattern, name)
        ]
        for fold, metric_keys in results.items():
            for metric_name in metric_names:
                avg = None
                count = 0
                for method_name in selected_methods:
                    if not metric_name in results[fold][method_name]:
                        continue
                    if avg is None:
                        avg = results[fold][method_name][metric_name]
                    else:
                        avg +=  results[fold][method_name][metric_name]
                    count += 1
                if count:
                    new_results[fold][group_name][metric_name] = avg/count
    return new_results

def print_table(
    results,
    result_dir,
    split=0,
    summary_table_metrics=None,
    sort_key="model",
    config=None,
    save_name="output_table",
    use_auc=False,
    use_int_auc=False,
):
    config = config or {}
    # Initialise output table
    results_table = PrettyTable()
    field_names = [
        "Method",
        "ROC-AUC" if use_auc else "Task Accuracy",

    ]
    result_table_fields_keys = [
        "test_auc_y" if use_auc else "test_acc_y",
    ]


    # Now add concept evaluation metrics
    field_names.extend([
        "Concept Accuracy",
        "Concept AUC",
    ])
    result_table_fields_keys.extend([
        "test_acc_c",
        "test_auc_c",
    ])


    # CAS, if we chose to compute it (off by default as it may be
    # computationally expensive)
    shared_params = config.get("shared_params", {})
    if (
        (not shared_params.get("skip_repr_evaluation", False)) and
        shared_params.get("run_cas", True)
    ):
        field_names.append("CAS")
        result_table_fields_keys.append("test_cas")

    # And intervention summaries if we chose to also include them
    if len(shared_params.get("intervention_config", {}).get("intervention_policies", [])) > 0:
        policy_config = shared_params['intervention_config']['intervention_policies'][0]
        # Then add the first policy we see as the default thing we print
        if policy_config['policy'] == 'random':
            useful_args = copy.deepcopy(policy_config)
            useful_args.pop('include_run_names', None)
            useful_args.pop('exclude_run_names', None)
            policy_arg_name = policy_config["policy"] + "_" + "_".join([
                f'{key}_{policy_config[key]}'
                for key in sorted(useful_args.keys())
                if key != 'policy'
            ])
            field_names.extend([
                "25% Int ROC-AUC" if use_int_auc else "25% Int Acc",
                "50% Int ROC-AUC" if use_int_auc else "50% Int Acc",
                "75% Int ROC-AUC" if use_int_auc else "75% Int Acc",
                "100% Int ROC-AUC" if use_int_auc else "100% Int Acc",
                "Val Int AUC",
                "Test Int AUC",
            ])
            result_table_fields_keys.extend([
                f"test_{'auc' if use_int_auc else 'acc'}_y_{policy_arg_name}_ints_25%",
                f"test_{'auc' if use_int_auc else 'acc'}_y_{policy_arg_name}_ints_50%",
                f"test_{'auc' if use_int_auc else 'acc'}_y_{policy_arg_name}_ints_75%",
                f"test_{'auc' if use_int_auc else 'acc'}_y_{policy_arg_name}_ints_100%",
                f"val_{'auc' if use_int_auc else 'acc'}_y_{policy_arg_name}_int_auc",
                f"test_{'auc' if use_int_auc else 'acc'}_y_{policy_arg_name}_int_auc",
            ])

    if summary_table_metrics is not None:
        for field in summary_table_metrics:
            if not isinstance(field, (tuple, list)):
                field = field, field
            field_name, field_pretty_name = field
            result_table_fields_keys.append(field_name)
            field_names.append(field_pretty_name)
    results_table.field_names = field_names
    table_rows_inds = {
        name: i for (i, name) in enumerate(result_table_fields_keys)
    }
    table_rows = {}
    end_results = defaultdict(lambda: defaultdict(list))
    for fold_idx, metric_keys in results.items():
        for method_name, metric_vals in metric_keys.items():
            for metric_name, vals in metric_vals.items():
                for desired_metric in result_table_fields_keys:
                    real_name = desired_metric
                    if (
                        ("_acc_y_" in desired_metric) or
                        ("_auc_y_" in desired_metric)
                    ) and (
                        ("_ints_" in desired_metric) and
                        (desired_metric[-1] == "%")
                    ):
                        # Then we are dealing with some interventions we wish
                        # to log
                        percent = int(
                            desired_metric[desired_metric.rfind("_") + 1 : -1]
                        )
                        desired_metric = desired_metric[:desired_metric.rfind("_")]
                    else:
                        percent = None

                    if metric_name == desired_metric:
                        if percent is None:
                            end_results[real_name][method_name].append(vals)
                        else:
                            end_results[real_name][method_name].append(
                                vals[int((len(vals) - 1) * percent/100)]
                            )

    for metric_name, runs in end_results.items():
        for method_name, trial_results in runs.items():
            if method_name not in table_rows:
                table_rows[method_name] = [
                    (None, None) for _ in result_table_fields_keys
                ]
            try:
                (mean, std) = np.mean(trial_results), np.std(trial_results)
                if metric_name in table_rows_inds:
                    table_rows[method_name][table_rows_inds[metric_name]] = \
                        (mean, std)
            except:
                logging.warning(
                    f"\tWe could not average results "
                    f"for {metric_name} in model {method_name}"
                )
    table_rows = list(table_rows.items())
    if sort_key == "model":
        # Then sort based on method name
        table_rows.sort(key=lambda x: x[0], reverse=True)
    elif sort_key in table_rows_inds:
        # Else sort based on the requested parameter
        table_rows.sort(
            key=lambda x: (
                x[1][table_rows_inds[sort_key]][0]
                if x[1][table_rows_inds[sort_key]][0] is not None
                else -float("inf")
            ),
            reverse=True,
    )
    for aggr_key, row in table_rows:
        for i, (mean, std) in enumerate(row):
            if mean is None or std is None:
                row[i] = "N/A"
            elif mean != mean: # Nan!
                row[i] = f'{mean} ± {std}'
            elif int(mean) == float(mean):
                row[i] = f'{mean} ± {std:}'
            else:
                row[i] = f'{mean:.4f} ± {std:.4f}'
        results_table.add_row([str(aggr_key)] + row)
    print("\t", "*" * 30)
    print(results_table)
    print("\n\n")

    # Also serialize the results
    if result_dir:
        with open(
            os.path.join(result_dir, f"{save_name}_fold_{split + 1}.txt"),
            "w",
        ) as f:
            f.write(str(results_table))

def filter_results(results, run_name, cut=False):
    output = {}
    for key, val in results.items():
        if run_name not in key:
            continue
        if cut:
            key = key[: -len("_" + run_name)]
        output[key] = val
    return output

def evaluate_expressions(config, parent_config=None, soft=False):
     parent_config = parent_config or config
     for key, val in config.items():
         if isinstance(val, (str,)):
             if len(val) >= 4 and (
                 val[0:2] == "{{" and val[-2:] == "}}"
             ):
                # Then do a simple substitution here
                try:
                    config[key] = val[2:-2].format(**parent_config)
                    config[key] = eval(config[key])
                except Exception as e:
                    if soft:
                        # Then we silently ignore this error
                        pass
                    else:
                        # otherwise we just simply raise it again!
                        raise e
             else:
                 config[key] = val.format(**parent_config)
         elif isinstance(val, dict):
             # Then we progress recursively
             evaluate_expressions(val, parent_config=parent_config)


def initialize_result_directory(results_dir):
    Path(
        os.path.join(
            results_dir,
            "models",
        )
    ).mkdir(parents=True, exist_ok=True)

    Path(
        os.path.join(
            results_dir,
            "history",
        )
    ).mkdir(parents=True, exist_ok=True)


def has_hierarchical_key(key, dictionary):
    for subkey in key.split("."):
        if subkey not in dictionary:
            return False
            # raise ValueError(
            #     f"Failed to find subkey {subkey} when looking for "
            #     f"hierarchical key {key}."
            # )
        dictionary = dictionary[subkey]
    # If we reached this point, then the key must be present
    return True

def get_hierarchical_key(key, dictionary):
    for subkey in key.split("."):
        if subkey not in dictionary:
            raise ValueError(
                f"Failed to find subkey {subkey} when looking for "
                f"hierarchical key {key}."
            )
        dictionary = dictionary[subkey]
    # If we reached this point, then the variable dictionary must have the
    # drones we are looking for
    return dictionary

def flatten_dictionary(dictionary, current_result=None, prefix="", sep="."):
    current_result = current_result or {}
    for key, val in dictionary.items():
        if isinstance(val, dict):
            flatten_dictionary(
                dictionary=val,
                current_result=current_result,
                prefix=prefix + key + sep,
                sep=sep,
            )
        else:
            current_result[prefix + key] = val
    return current_result

def nested_dictionary_set(dictionary, key, val):
    atoms = key.split(".")
    for idx, subkey in enumerate(atoms):
        if idx == (len(atoms) - 1):
            dictionary[subkey] = val
        elif subkey not in dictionary:
            raise ValueError(
                f"Failed to find subkey {subkey} when looking for "
                f"hierarchical key {key}."
            )
        else:
            dictionary = dictionary[subkey]

def generate_hyperparameter_configs(config):
    if "grid_variables" not in config:
        # Then nothing to see here so we will return
        # a singleton set with this config in it
        return [config]
    # Else time to do some hyperparameter search in here!
    vars = config["grid_variables"]
    options = []
    for var in vars:
        if not has_hierarchical_key(var, config):
            raise ValueError(
                f'All variable names in "grid_variables" must be exhisting '
                f'fields in the config. However, we could not find any '
                f'(nested) field with name "{var}".'
            )
        val = get_hierarchical_key(var, config)
        if not isinstance(val, list):
            raise ValueError(
                f'If we are doing a hyperparameter search over (nested) '
                f'variable "{var}", we expect it to be a list of values. '
                f'Instead we got {val}.'
            )
        options.append(val)
    mode = config.get('grid_search_mode', "exhaustive").lower().strip()
    if mode in ["grid", "exhaustive"]:
        iterator = itertools.product(*options)
    elif mode in ["paired"]:
        iterator = zip(*options)
    else:
        raise ValueError(
            f'The only supported values for grid_search_mode '
            f'are "paired" and "exhaustive". We got {mode} '
            f'instead.'
        )
    result = []
    for specific_vals in iterator:
        current = copy.deepcopy(config)
        for var_name, new_val in zip(vars, specific_vals):
            nested_dictionary_set(current, var_name, new_val)
        result.append(current)
    return result