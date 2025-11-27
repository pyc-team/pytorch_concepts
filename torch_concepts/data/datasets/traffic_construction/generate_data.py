#! /usr/bin/env python

"""
Main script to generate the traffic dataset used in this repository.
"""

import argparse
import copy
import hashlib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import time
import torch

from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from tqdm import tqdm

from . import utils

from .cars import (
    AMBULANCE, AVAILABLE_CAR_COLORS, CAR_SPRITES
)
from .lights import (
    add_light_x_axis, add_light_y_axis
)
from .intersection import (
    AVAILABLE_LANES, INTERSECTION
)


################################################################################
## Helper Functions
################################################################################

def _are_perp(dir_1, dir_2):
    if dir_1 in ['north', 'south']:
        return dir_2 in ['east', 'west']
    return dir_2 in ['north', 'south']

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
    return config

def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def show_image_grid(
    images,
    labels,
    label_names,
    extra_labels,
    grid_size=(3, 3),
    figsize=(10, 10),
):
    # Select a random sample of indices for the grid
    indices = random.sample(range(len(images)), grid_size[0] * grid_size[1])
    sampled_images = [images[i] for i in indices]
    sampled_labels = [labels[i] for i in indices]
    sampled_extra_labels = [extra_labels[i] for i in indices]

    # Plot images in a grid
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
    axes = axes.flatten()  # Flatten to easily iterate over axes if needed

    for i, ax in enumerate(axes):
        ax.imshow(
            sampled_images[i],
            cmap='gray' if sampled_images[i].ndim == 2 else None,
        )
        ax.set_title(
            f"Label: {label_names[sampled_labels[i]]}{sampled_extra_labels[i]}",
            fontsize=10,
        )
        ax.axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()

################################################################################
## Time to Construct the Actual Data
################################################################################

def make_intersection_sample(
    background,
    position_para_noise=50,
    position_perp_noise=20,
    error_probability=0.1,
    p_ambulance=0.2,
    min_num_cars=0,
    max_num_cars=7,
    resize_final_image=0.15,
    cars=CAR_SPRITES,
    available_lanes=AVAILABLE_LANES,
    possible_starting_directions=None,
    thickness=100,
    inplace=True,
    light_scale=1.5,
    ambulance_sprite=AMBULANCE,
    use_lights_sprites=False,
):
    if possible_starting_directions is None:
        crossing_lanes = [x for x in available_lanes if x['before_int']]
    else:
        crossing_lanes = [
            x for x in available_lanes if x['before_int']
            if x['abs_pos'] in possible_starting_directions
        ]

    # Make transparent pixels be set to white on the background
    background_rgb = background[:, :, :3].copy()
    background_alpha = background[:, :, 3:].copy()
    result_image = np.where(
        np.concatenate([background_alpha for _ in range(3)], axis=-1) == 0,
        np.ones_like(background_rgb),
        background_rgb,
    )
    # We operate in the downsized space to speed things up when possible!
    result_image = utils.resize_with_aspect_ratio(
        result_image,
        target_height=resize_final_image,
    )

    # We will construct meta data for this sample as we build it from the
    # ground up
    sample_meta = {}

    # First make the selection of where we will place the car we will be
    # considering as the target for this sample
    cross_idx = np.random.randint(0, len(crossing_lanes))
    selected_car = np.random.choice(cars)
    sample_meta['selected_car'] = selected_car

    # Clearly highlight the selected sample!
    thickness = int(np.ceil(thickness * resize_final_image))
    cross_car = utils.highlight_edges(selected_car['img'], thickness=thickness)
    selected_lane = crossing_lanes[cross_idx]
    cross_lane, cross_rot, cross_x_off, cross_y_off = \
        selected_lane['bounds'], selected_lane['rot'], \
            selected_lane['x_off'], selected_lane['y_off']
    sample_meta['selected_lane'] = selected_lane

    # Add some noise to its position
    if selected_lane['dir'] in ['east', 'west']:
        cross_x_off += np.random.choice(
            np.arange(-position_para_noise, position_para_noise+1)
        )
        cross_y_off += np.random.choice(
            np.arange(-position_perp_noise, position_perp_noise+1)
        )
    else:
        cross_x_off += np.random.choice(
            np.arange(-position_perp_noise, position_perp_noise+1)
        )
        cross_y_off += np.random.choice(
            np.arange(-position_para_noise, position_para_noise+1)
        )

    # And time to add it to the resulting image
    cross_lane_x_bounds, cross_lane_y_bounds = cross_lane
    sample_meta['selected_lane'] = selected_lane
    cross_x_offset = cross_lane_x_bounds[0] + cross_x_off
    cross_y_offset = cross_lane_y_bounds[0] + cross_y_off

    # We operate in the downsized space to speed things up when possible!
    cross_x_offset, cross_y_offset = utils.transform_scale_coordinates(
        cross_x_offset,
        cross_y_offset,
        ratio=resize_final_image,
    )
    cross_car = utils.resize_with_aspect_ratio(
        cross_car,
        target_height=resize_final_image,
    )

    selected_car['position'] = (cross_x_offset, cross_y_off)
    result_image = utils.add_sprite(
        sprite=cross_car,
        background=result_image,
        rotation=cross_rot,
        target_height=selected_car['scale'],
        x_offset=cross_x_offset,
        y_offset=cross_y_offset,
        inplace=inplace,
    )

    # Decide which direction the traffic light is set:
    force_law_break = np.random.choice(
        [True, False],
        p=[error_probability, 1-error_probability]
    )
    if force_law_break:
        green_in_favour = True
    else:
        green_in_favour = np.random.choice([False, True])
    sample_meta['green'] = green_in_favour
    if selected_lane['dir'] in ['east', 'west']:
        flow_directions = (
            ['east', 'west']
            if green_in_favour else ['north', 'south']
        )
        result_image = add_light_x_axis(
            result_image,
            green=green_in_favour,
            ratio=resize_final_image,
            inplace=inplace,
            light_scale=light_scale,
            use_lights_sprites=use_lights_sprites,
        )
        result_image = add_light_y_axis(
            result_image,
            green=(not green_in_favour),
            ratio=resize_final_image,
            inplace=inplace,
            light_scale=light_scale,
            use_lights_sprites=use_lights_sprites,
        )
    else:
        flow_directions = (
            ['north', 'south']
            if green_in_favour else ['east', 'west']
        )
        result_image = add_light_y_axis(
            result_image,
            green=green_in_favour,
            ratio=resize_final_image,
            inplace=inplace,
            light_scale=light_scale,
            use_lights_sprites=use_lights_sprites,
        )
        result_image = add_light_x_axis(
            result_image,
            green=(not green_in_favour),
            ratio=resize_final_image,
            inplace=inplace,
            light_scale=light_scale,
            use_lights_sprites=use_lights_sprites,
        )

    # Now add other cars:
    free_lanes = [
        x for x in available_lanes
        if x['idx'] != selected_lane['idx']
    ]
    if min_num_cars < min(len(free_lanes), max_num_cars + 1):
        num_selected_cars = np.random.randint(
            min_num_cars,
            min(len(free_lanes), max_num_cars + 1)
        )
    else:
        num_selected_cars = 0
    if force_law_break:
        # Then make sure at least one other car breaking the law is included
        num_selected_cars = max(num_selected_cars, 1)
        # And we will explicitly force the first one to be the law breaker!
        other_lanes = list(np.random.choice(
            [x for x in free_lanes if x['dir'] not in flow_directions],
            1,
            replace=False,
        ))
        other_lanes += list(np.random.choice(
            [x for x in free_lanes if x['idx'] != other_lanes[0]['idx']],
            num_selected_cars - 1,
            replace=False,
        ))
    else:
        other_lanes = np.random.choice(
            free_lanes,
            num_selected_cars,
            replace=False,
        )
    if len(other_lanes) > 0 and np.random.choice(
        [True, False],
        p=[p_ambulance, 1-p_ambulance]
    ):
        other_cars = [ambulance_sprite] + [
            np.random.choice(cars)
            for _ in other_lanes[1:]
        ]
    else:
        other_cars = [
            np.random.choice(cars)
            for _ in other_lanes
        ]
    sample_meta['perp_intersection_occupied'] = False
    sample_meta['perp_incoming_ambulance'] = False
    per_intersection_occupied = False
    forbidden_dirs = set()
    final_other_cars = []
    final_other_lanes = []
    for other_lane, other_car in zip(other_lanes, other_cars):
        new_car = other_car['img']
        other_car_meta = {
            key: val
            for (key, val) in other_car.items() if key != 'img'
        }
        new_lane, new_rot, new_x_off, new_y_off = \
            other_lane['bounds'], other_lane['rot'], \
                other_lane['x_off'], other_lane['y_off']

        # Add some noise to its position
        if other_lane['dir'] in forbidden_dirs:
            continue
        car_breaking_law = False
        if other_car['ambulance']:
            # If it is an ambulance, then it will always be the same new lane so
            # can use it to potentially claim the intersection! Everyone must
            # not use the intersection even if the car is not physically there!
            assert not per_intersection_occupied
            per_intersection_occupied = True
            if other_lane['before_int']:
                if other_lane['dir'] in ['east', 'south']:
                    used_para_noise_bottom = 0
                    used_para_noise_top = 500
                else:
                    used_para_noise_bottom = -500
                    used_para_noise_top = 50
            else:
                if other_lane['dir'] in ['east', 'south']:
                    used_para_noise_bottom = -500
                    used_para_noise_top = 50
                else:
                    used_para_noise_bottom = 0
                    used_para_noise_top = 500
            forbidden_dirs.add(other_lane['dir'])

        elif (other_lane['dir'] not in flow_directions) and (
            force_law_break and (not per_intersection_occupied)
        ):
            if other_lane['before_int']:
                if other_lane['dir'] in ['east', 'south']:
                    used_para_noise_bottom = 300
                    used_para_noise_top = 400
                else:
                    used_para_noise_bottom = -500
                    used_para_noise_top = -400
            else:
                if other_lane['dir'] in ['east', 'south']:
                    used_para_noise_bottom = -500
                    used_para_noise_top = -400
                else:
                    used_para_noise_bottom = 300
                    used_para_noise_top = 400
        elif other_lane['dir'] in flow_directions and (
            not per_intersection_occupied
        ):
            # Then the car can actually be in the middle of the lane!
            used_para_noise_bottom = -50
            used_para_noise_top = 600
        else:
            # Else we do a very small fluctuation within the lane before or
            # after the intersection
            used_para_noise_bottom = -position_para_noise
            used_para_noise_top = position_para_noise + 1

        # Add the actual deviations
        if other_lane['dir'] in ['east', 'west']:
            new_x_off += np.random.choice(
                np.arange(used_para_noise_bottom, used_para_noise_top)
            )
            new_y_off += np.random.choice(
                np.arange(-position_perp_noise, position_perp_noise+1)
            )

        else:
            new_x_off += np.random.choice(
                np.arange(-position_perp_noise, position_perp_noise+1)
            )
            new_y_off += np.random.choice(
                np.arange(used_para_noise_bottom, used_para_noise_top)
            )


        # And time to add it to the resulting image
        new_lane_x_bounds, new_lane_y_bounds = new_lane
        new_x_offset = new_lane_x_bounds[0] + new_x_off
        new_y_offset = new_lane_y_bounds[0] + new_y_off

        # Save state for generating dataset annotation
        other_car_meta['position'] = (new_x_offset, new_y_offset)

        if other_lane['dir'] in ['east', 'west']:
            car_back = new_x_offset + int(other_car_meta['scale'] * np.max(new_car.shape[:2]))
            car_front = new_x_offset
            other_car_meta['in_intersection'] = (
                ((car_back >= 450) and (car_back <= 815)) or
                ((car_front >= 450) and (car_front <= 815))

            )
        else:
            car_back = new_y_offset + int(other_car_meta['scale'] * np.max(new_car.shape[:2]))
            car_front = new_y_offset
            other_car_meta['in_intersection'] = (
                ((car_back >= 450) and (car_back <= 815)) or
                ((car_front >= 450) and (car_front <= 815))

            )

        if _are_perp(other_lane['dir'], selected_lane['dir']) and (
            other_car_meta['in_intersection'] or other_car_meta['ambulance']
        ):
            sample_meta['perp_intersection_occupied'] = True
            sample_meta['perp_incoming_ambulance'] = (
                sample_meta['perp_incoming_ambulance'] or other_car_meta['ambulance']
            )
            per_intersection_occupied = True
            car_breaking_law = not other_car_meta['ambulance']

        # We operate in the downsized space to speed things up when possible!
        new_x_offset, new_y_offset = utils.transform_scale_coordinates(
            new_x_offset,
            new_y_offset,
            ratio=resize_final_image,
        )
        new_car = utils.resize_with_aspect_ratio(
            new_car,
            target_height=resize_final_image,
        )
        other_car_meta['position'] = (new_x_offset, new_y_offset)
        other_car_meta['breaking_law'] = car_breaking_law
        other_car_meta['lane_idx'] = other_lane['idx']
        result_image = utils.add_sprite(
            sprite=new_car,
            background=result_image,
            rotation=new_rot,
            target_height=other_car_meta['scale'],
            x_offset=new_x_offset,
            y_offset=new_y_offset,
            inplace=inplace,
        )
        final_other_cars.append(other_car_meta)
        final_other_lanes.append(other_lane)

    sample_meta['other_cars'] = final_other_cars
    sample_meta['other_car_lanes'] = final_other_lanes

    if sample_meta['green']:
        sample_meta['action'] = 'continue' if not (
            sample_meta['perp_intersection_occupied'] or
            sample_meta['perp_incoming_ambulance']
        ) else 'stop'
    else:
        sample_meta['action'] = 'stop'
    # And save it after also resizing it
    sample_meta['img'] = result_image
    return sample_meta


################################################################################
## Wrapper for multiprocessing!
################################################################################

def create_sample(in_multi, as_arrays=None, seed=None):
    idx, config, record_dir = in_multi

    # To ensure determinism even when using multiprocessing, let's set the seed
    # here as a function of the index
    if seed is None:
        seed = config.get('seed', None)
    car_colors = config.get('car_colors', None)
    if as_arrays is None:
        as_arrays = config.get('as_arrays', False)

    if seed is not None:
        new_seed = seed + idx
        random.seed(new_seed)
        np.random.seed(new_seed)

    # And filter the cars
    if car_colors is None:
        cars_to_use = CAR_SPRITES
    else:
        cars_to_use = [x for x in CAR_SPRITES if x['color'] in car_colors]

    sample = make_intersection_sample(
        position_para_noise=config['position_para_noise'],
        position_perp_noise=config['position_perp_noise'],
        error_probability=config['error_probability'],
        p_ambulance=config['p_ambulance'],
        min_num_cars=config['min_num_cars'],
        max_num_cars=config['max_num_cars'],
        resize_final_image=config['resize_final_image'],
        possible_starting_directions=config['possible_starting_directions'],
        thickness=config['thickness'],
        light_scale=config['light_scale'],
        use_lights_sprites=config['use_lights_sprites'],

        background=INTERSECTION,
        cars=cars_to_use,
        available_lanes=AVAILABLE_LANES,

    )
    if as_arrays:
        return sample

    # Else we will serialize records as we generate them to avoid overloading
    # the memory!
    np.savez_compressed(
        os.path.join(record_dir, f'sample_{idx}.npz'),
        img=sample.pop('img'),
        metadata=sample,
    )
    return None

def construct_samples(
    config,
    records_dir,
    indices,
    dataset_name="",
    num_threads=1,
    verbose=False,
):

    if num_threads > 1:
        if verbose:
            print(
                f"Using {num_threads} threads to generate a "
                f"{dataset_name} dataset with {len(indices)} samples..."
            )
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(
                executor.map(
                    create_sample,
                    [
                        (idx, config, records_dir)
                        for idx in indices
                    ],
                ),
                total=len(indices),
            ))

    else:
        # Else we proceed to do everything within this same process
        if verbose:
            print(
                f"Generating a dataset with {len(indices)} "
                f"{dataset_name} samples using a single thread..."
            )
            for idx in tqdm(indices):
                create_sample((idx, config, RECORD_DIR))


################################################################################
## Arg parser
################################################################################



def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for traffic simulation"
    )

    parser.add_argument(
        "--dir_name",
        type=str,
        default="data",
        help="Directory where the output data folders will be generated",
    )
    parser.add_argument(
        "--sym_link_name",
        type=str,
        default="latest_version",
        help="Name of symlink to use for the generated dataset",
    )

    parser.add_argument(
        "--num_threads",
        type=int,
        default=6,
        help="Number of processes to use during generation",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to use for generation",
    )
    parser.add_argument(
        "--position_para_noise",
        type=float,
        default=50,
        help="Noise on the position of each car parallel to movement direction",
    )
    parser.add_argument(
        "--position_perp_noise",
        type=float,
        default=20,
        help=(
            "Noise on the position of each car perpendicular to movement "
            "direction"
        ),
    )
    parser.add_argument(
        "--error_probability",
        type=float,
        default=0.1,
        help="Probability of a car making an error (breaking the law)",
    )
    parser.add_argument(
        "--p_ambulance",
        type=float,
        default=0.2,
        help="Probability of an ambulance being in the sample",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=100,
        help="Thickness of marker edge for the target car",
    )
    parser.add_argument(
        "--min_num_cars",
        type=int,
        default=0,
        help="Minimum number of cars per image (not including target car)",
    )
    parser.add_argument(
        "--max_num_cars",
        type=int,
        default=7,
        help="Maximum number of cars per image (not including target car)",
    )
    parser.add_argument(
        "--resize_final_image",
        type=float,
        default=0.15,
        help="Downsampling factor for the final image",
    )
    parser.add_argument(
        "--car_colors",
        type=str, nargs="+",
        default=None,
        help="List of car colors to consider (e.g., red, blue)",
    )
    parser.add_argument(
        "--possible_starting_directions",
        type=str, nargs="+",
        default=["north", "east", "south", "west"],
        help="Possible starting directions for the target car",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.6,
        help="Fraction of samples for the training set",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of samples for the validation set",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Fraction of samples for the test set",
    )
    parser.add_argument(
        "--light_scale",
        type=float,
        default=1.5,
        help="Scaling for size of traffic lights (can be less or more than 1).",
    )
    parser.add_argument(
        "--use_lights_sprites",
        action="store_true",
        help=(
            "If set, then we will default to use traffic light sprites over "
            "simple colored circles to represent traffic light states."
        ),
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help=(
            "Rerun generation even if the dataset was already generated with "
            "the same config"
        ),
    )

    parser.add_argument(
        "--use_absolute_path",
        action="store_true",
        help=(
            "If given, the sym-link of the generated dataset will be done "
            "using an absolute path rather than a relative path. This is "
            "useful if you want to generate and use this dataset locally but "
            "it is not recommended if the dataset is to be exported to other "
            "directories."
        ),
    )

    parser.add_argument(
        "--visual_validation",
        action="store_true",
        help=(
            "If set, then we will first generate some example images before "
            "generating the entire dataset upon inspection by the user."
        ),
    )

    parser.add_argument(
        '--test_param',
        action='append',
        nargs=2,
        metavar=('param_name=value'),
        help=(
            'Allows the passing of a different parameters for the testing '
            'config that differ from those in the training config. All '
            'parameters not passed using this flag take their corresponding '
            'values from the original train configuration parameters.'
        ),
        default=[],
    )

    parser.add_argument(
        '--val_param',
        action='append',
        nargs=2,
        metavar=('param_name=value'),
        help=(
            'Allows the passing of a different parameters for the validation '
            'config that differ from those in the training config. All '
            'parameters not passed using this flag take their corresponding '
            'values from the original train configuration parameters.'
        ),
        default=[],
    )


    args = parser.parse_args()

    # Additional validation
    assert 0 <= args.min_num_cars <= 7, \
        "min_num_cars must be between 0 and 7"
    assert 0 <= args.max_num_cars <= 7, \
        "max_num_cars must be between 0 and 7"
    assert args.min_num_cars <= args.max_num_cars, \
        "min_num_cars must be <= max_num_cars"
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio must sum to 1"

    return args








################################################################################
##  Program Entry Point
################################################################################

if __name__ == "__main__":

    start_time = time.time()

    ## Parse the program's arguments
    args = parse_args()
    if args.car_colors is None or args.car_colors == []:
        args.car_colors = AVAILABLE_CAR_COLORS

    ## Construct config for verification
    config = dict(
        n_samples=args.n_samples,
        seed=args.seed,
        position_para_noise=args.position_para_noise,
        position_perp_noise=args.position_perp_noise,
        error_probability=args.error_probability,
        p_ambulance=args.p_ambulance,
        min_num_cars=args.min_num_cars,
        max_num_cars=args.max_num_cars,
        resize_final_image=args.resize_final_image,
        car_colors=sorted(args.car_colors),
        possible_starting_directions=sorted(args.possible_starting_directions),
        thickness=args.thickness,
        light_scale=args.light_scale,
        use_lights_sprites=args.use_lights_sprites,

        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # Generate a hash for the config so that we can determine if this dataset
    # has been previously generated
    setup_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()


    ############################################################################
    ## Set up all directories
    ############################################################################

    FINAL_DATA_DIR = os.path.join(args.dir_name, str(setup_hash))
    print(f"Real data will be dumped at {FINAL_DATA_DIR}")
    SYM_LINK = os.path.join(
        args.dir_name,
        args.sym_link_name,
    )

    RECORD_DIR = os.path.join(FINAL_DATA_DIR, 'records')
    os.makedirs(RECORD_DIR, exist_ok=True)

    # Save the config so that we can always reconstruct the dataset
    with open(os.path.join(FINAL_DATA_DIR, f'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    # If we notice the data has been already generated with the correct hash,
    # then no need to re-generate it
    if os.path.exists(os.path.join(FINAL_DATA_DIR, 'completed.txt')) and (
        not args.rerun
    ):
        print(
            'We found a dataset previously generated with the same config that '
            'has been cached.'
        )
        print('\tIf you wish to re-generate it, please use --rerun.')

    else:
        ## Fix Random Seeds
        fix_seeds(args.seed)

        ## Potentially generate visual examples
        if args.visual_validation:
            generating = True
            attempts = 0
            while generating:
                attempts += 1
                examples = []
                print("Generating 25 examples for visual validation...")
                for idx in tqdm(range(25)):
                    examples.append(
                        create_sample(
                            (idx, config, RECORD_DIR),
                            as_arrays=True,
                            seed=attempts,
                        )
                    )
                print("\tDone...")
                print("To continue, please close the figure.")
                images = np.concatenate(
                    [
                        np.expand_dims(x['img'], axis=0)
                        for x in examples
                    ],
                    axis=0,
                )
                labels = np.array([
                    int(x['action'] == 'stop')
                    for x in examples
                ])

                show_image_grid(
                    images,
                    labels,
                    ['continue', 'stop'],
                    [
                        " (ambulance)" if (
                            x["perp_incoming_ambulance"] and
                            (x['green'])
                        ) else (
                            " (law broken)" if (
                                x['perp_intersection_occupied'] and
                                x['green']
                            ) else (
                                " (green)" if x['green'] else " (red)"
                            )
                        )
                        for x in examples
                    ],
                    grid_size=(5, 5),
                    figsize=(10, 10),
                )
                answer = None
                while answer not in ['y', 'n', 'regenerate']:
                    answer = input(
                        f'Would you like to proceed with generating the entire '
                        f'dataset ({args.n_samples} images)? '
                        f'[yes/no/regenerate] '
                    ).strip().lower()
                    answer = {'yes': 'y', 'no': 'n'}.get(answer, answer)
                if answer == 'n':
                    print("Aborting generation after visual validation")
                    exit(0)
                if answer != 'regenerate':
                    generating = False

            # If we continue, let's reset the seeds then!
            fix_seeds(args.seed)


        ## Generate Samples

        ##  Split the data
        print("Splitting data...")

        # Compute split indices. Because the data is randomly generated (i.e.,
        # already shuffled), we can just split the data into three by taking
        # contiguous but disjoint areas of it
        train_size = int(np.ceil(args.n_samples * args.train_ratio))
        train_indices = np.arange(0, train_size)

        val_size = int(np.ceil(args.n_samples * args.val_ratio))
        val_indices = np.arange(train_size, train_size + val_size)

        test_size = args.n_samples - (val_size + train_size)
        test_indices = np.arange(train_size + val_size, args.n_samples)

        # Serialize them for future reference
        np.save(
            os.path.join(FINAL_DATA_DIR, 'train_indices.npy'),
            train_indices,
        )
        np.save(
            os.path.join(FINAL_DATA_DIR, 'val_indices.npy'),
            val_indices,
        )
        np.save(
            os.path.join(FINAL_DATA_DIR, 'test_indices.npy'),
            test_indices,
        )
        print("\tDone!")

        val_config = extend_with_global_params(
            copy.deepcopy(config),
            args.val_param,
        )
        test_config = extend_with_global_params(
            copy.deepcopy(config),
            args.test_param,
        )

        construct_samples(
            config=config,
            records_dir=RECORD_DIR,
            indices=train_indices,
            dataset_name="training",
            num_threads=args.num_threads,
            verbose=True,
        )
        construct_samples(
            config=val_config,
            records_dir=RECORD_DIR,
            indices=val_indices,
            dataset_name="validation",
            num_threads=args.num_threads,
            verbose=True,
        )
        construct_samples(
            config=test_config,
            records_dir=RECORD_DIR,
            indices=test_indices,
            dataset_name="test",
            num_threads=args.num_threads,
            verbose=True,
        )
        with open(os.path.join(FINAL_DATA_DIR, 'completed.txt'), "w") as file:
            file.write("1")

    # Make the symbolic link (overwriting a previous one if it already
    # exists)
    if os.path.islink(SYM_LINK):
        os.remove(SYM_LINK)
    if args.use_absolute_path:
        os.symlink(os.path.abspath(FINAL_DATA_DIR), SYM_LINK)
    else:
        os.symlink(FINAL_DATA_DIR, SYM_LINK)


    time_taken = str(timedelta(seconds=time.time() - start_time))

    print(f'Dataset successfully generated after {time_taken} minutes!')

    size_in_bytes = utils.get_directory_size(FINAL_DATA_DIR)
    memory_used = utils.format_size(size_in_bytes)
    print(f'Total size of dataset is {memory_used}.')

    print(
        'You can access all the data using the symbolink link '
        f'"{SYM_LINK}" (which points to "{FINAL_DATA_DIR}")'
    )
    exit(0)
