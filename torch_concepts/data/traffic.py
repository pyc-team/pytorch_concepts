"""
Synthetic traffic intersection dataset.
"""

import copy
import copy
import hashlib
import json
import numpy as np
import numpy as np
import os
import os
import pickle
import torch
import torch

from torch.utils.data import Dataset

from .traffic_construction.cars import AVAILABLE_CAR_COLORS
from .traffic_construction.generate_data import construct_samples, fix_seeds

class TrafficLights(Dataset):
    """
    Synthetic traffic dataset
    """

    def __init__(
        self,
        root_dir,
        split='train',
        image_size=256,
        concept_transform=None,
        class_dtype=float,
        img_transform=None,
        verbose_generation=True,
        regenerate=False,
        sym_link=None,
        use_absolute_path=False,

        # Potential config values
        num_threads=4,
        n_samples=1000,
        seed=42,
        position_para_noise=50,
        position_perp_noise=20,
        error_probability=0.1,
        p_ambulance=0.2,
        min_num_cars=0,
        max_num_cars=7,
        resize_final_image=0.15,
        car_colors=AVAILABLE_CAR_COLORS,
        possible_starting_directions=[
            'north',
            'east',
            'south',
            'west',
        ],
        thickness=15,
        light_scale=1.5,
        use_lights_sprites=False,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        test_config_override_values=dict(
            p_ambulance=0.5,
            error_probability=0.5,
        ),
        val_config_override_values=None,
    ):
        self.n_samples = n_samples
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.sym_link = sym_link
        train_config = dict(
            n_samples=n_samples,
            position_para_noise=position_para_noise,
            position_perp_noise=position_perp_noise,
            error_probability=error_probability,
            p_ambulance=p_ambulance,
            min_num_cars=min_num_cars,
            max_num_cars=max_num_cars,
            resize_final_image=resize_final_image,
            car_colors=car_colors,
            possible_starting_directions=possible_starting_directions,
            thickness=thickness,
            light_scale=light_scale,
            use_lights_sprites=use_lights_sprites,
            seed=self.seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        # Generate a hash for the config so that we can determine if this dataset
        # has been previously generated
        self.root_dir = root_dir

        has_key = copy.deepcopy(train_config)
        has_key['test_config_override_values'] = test_config_override_values
        has_key['val_config_override_values'] = val_config_override_values
        setup_hash = hashlib.sha256(
            json.dumps(has_key, sort_keys=True).encode()
        ).hexdigest()
        self.real_data_dir = os.path.join(self.root_dir, str(setup_hash))
        os.makedirs(self.real_data_dir, exist_ok=True)

        # Generate the dataset, if needed
        self._construct_dataset(
            train_config,
            num_threads=num_threads,
            test_config_override_values=test_config_override_values,
            val_config_override_values=val_config_override_values,
            use_absolute_path=use_absolute_path,
            verbose=verbose_generation,
            regenerate=regenerate,
        )

        # At this point, we can assume that the dataset has been fully generated!
        self.split = split
        self.records_dir =  os.path.join(
            self.real_data_dir,
            f'records/',
        )
        os.makedirs(self.records_dir, exist_ok=True)

        assert split in ['train', 'test', 'val']

        self.concept_names = [
            'green x-axis',
            'green y-axis',
            'ambulance seen',
            'car in intersection',
            'other cars visible',
            'selected car in north lanes',
            'selected car in east lanes',
            'selected car in south lanes',
            'selected car in west lanes',
            'green light on selected lane',
            'car in intersection perpendicular to selected car',
            'ambulance approaching perpendicular to selected car',
        ]
        self.task_names = ['continue']

        self.class_dtype = class_dtype
        if concept_transform is None:
            concept_transform = lambda x: x
        self.concept_transform = concept_transform


        self.split_array_map = np.load(
            os.path.join(self.real_data_dir, f'{split}_indices.npy')
        )

        self.transform = (
            img_transform if img_transform is not None
            else lambda x: x
        )

    def _construct_dataset(
        self,
        config,
        num_threads=4,
        test_config_override_values=None,
        val_config_override_values=None,
        use_absolute_path=False,
        verbose=True,
        regenerate=False,
    ):
        if os.path.exists(
            os.path.join(self.real_data_dir, 'completed.txt')
        ) and (
            not regenerate
        ):
            if verbose:
                print(
                    'We found a dataset previously generated with the same '
                    'config that has been cached.'
                )
                print(
                    '\tIf you wish to re-generate it, please use '
                    'regenerate=True.'
                )
                return

        ## Fix Random Seeds
        fix_seeds(config.get('seed', self.seed))

        ##  Split the data
        # Compute split indices. Because the data is randomly generated (i.e.,
        # already shuffled), we can just split the data into three by taking
        # contiguous but disjoint areas of it
        n_samples = config.get('n_samples', self.n_samples)
        train_size = int(np.ceil(
            n_samples * self.train_ratio
        ))
        train_indices = np.arange(0, train_size)

        val_size = int(np.ceil(
            n_samples * self.val_ratio
        ))
        val_indices = np.arange(train_size, train_size + val_size)

        test_indices = np.arange(train_size + val_size, n_samples)

        # Serialize them for future reference
        np.save(
            os.path.join(self.real_data_dir, 'train_indices.npy'),
            train_indices,
        )
        np.save(
            os.path.join(self.real_data_dir, 'val_indices.npy'),
            val_indices,
        )
        np.save(
            os.path.join(self.real_data_dir, 'test_indices.npy'),
            test_indices,
        )

        with open(
            os.path.join(self.real_data_dir, f'train_config.pkl'),
            'wb',
        ) as f:
            pickle.dump(config, f)
        construct_samples(
            config=config,
            records_dir=self.records_dir,
            indices=train_indices,
            dataset_name="training",
            num_threads=num_threads,
            verbose=True,
        )

        val_config = copy.deepcopy(config)
        val_config.update(val_config_override_values or {})
        with open(
            os.path.join(self.real_data_dir, f'val_config.pkl'),
            'wb',
        ) as f:
            pickle.dump(val_config, f)
        construct_samples(
            config=val_config,
            records_dir=self.records_dir,
            indices=val_indices,
            dataset_name="validation",
            num_threads=num_threads,
            verbose=True,
        )

        test_config = copy.deepcopy(config)
        test_config.update(test_config_override_values or {})
        with open(
            os.path.join(self.real_data_dir, f'test_config.pkl'),
            'wb',
        ) as f:
            pickle.dump(test_config, f)
        construct_samples(
            config=test_config,
            records_dir=self.records_dir,
            indices=test_indices,
            dataset_name="test",
            num_threads=num_threads,
            verbose=True,
        )

        # And mark the dataset as complete
        with open(os.path.join(self.real_data_dir, 'completed.txt'), "w") as f:
            f.write("1")

        if self.sym_link is not None:
            # Make the symbolic link (overwriting a previous one if it already
            # exists)
            if os.path.islink(self.sym_link):
                os.remove(self.sym_link)
            if use_absolute_path:
                os.symlink(os.path.abspath(self.real_data_dir), self.sym_link)
            else:
                os.symlink(self.real_data_dir, self.sym_link)


    def _from_meta_to_concepts(self, sample_meta):
        # Concepts will be:
        #  [0] Light color x-axis (0 if red, 1 if green)
        #  [1] Light color y-axis (0 if red, 1 if green)
        #  [2] Ambulance (1 if there is an ambulance in sight, 0 otherwise)
        #  [3] Car in intersection (1 if there is a car in the intersection,
        #      0 otherwise)
        #  [4] Other cars (1 if there are other cars visible anywhere, 0
        #      otherwise)
        #  [5] Selected car in north lane
        #  [6] Selected car in east lane
        #  [7] Selected car in south lane
        #  [8] Selected car in west lane
        #  [9] Green light on selected lane
        #  [10] Car perpendicular in intersection (1 if there is a car in the
        #      intersection in the direction perpendicular to this car, 0
        #      otherwise)
        #  [11] Ambulance Perpendicular (1 if the ambulance is in the
        #      direction perpendicular to the car, 0 otherwise)
        c = np.array([
            float(
                sample_meta['green'] and
                (sample_meta['selected_lane']['dir'] in ['east', 'west'])
            ), # [0]
            float(
                sample_meta['green'] and
                (sample_meta['selected_lane']['dir'] in ['south', 'north'])
            ), # [1]
            float(np.any(
                [x['ambulance'] for x in sample_meta['other_cars']]
            )), # [2]
            float(np.any([
                x['in_intersection']
                for x in sample_meta['other_cars']
            ])), # [3]
            float(len(sample_meta['other_cars']) > 0), # [4]
            float(sample_meta['selected_lane']['idx'] == 7), # [5]
            float(sample_meta['selected_lane']['idx'] == 1), # [6]
            float(sample_meta['selected_lane']['idx'] == 3), # [7]
            float(sample_meta['selected_lane']['idx'] == 5), # [8]
            float(sample_meta['green']), # [9]
            float(sample_meta['perp_intersection_occupied']), # [10]
            float(sample_meta['perp_incoming_ambulance']), # [11]

        ])
        return self.concept_transform(torch.FloatTensor(c))

    def _from_meta_to_label(self, sample_meta):
        y = self.class_dtype(sample_meta['action'] == 'continue')
        if self.class_dtype == float:
            y = torch.FloatTensor([y]).squeeze(-1)
        return y

    def sample_array(self, real_idx):
        sample_filename = os.path.join(
            self.records_dir,
            f'sample_{real_idx}.npz'
        )
        loaded_data = np.load(sample_filename, allow_pickle=True)
        img = loaded_data['img']
        metadata = loaded_data['metadata'].item()
        img = torch.FloatTensor(
            # Transpose the image so that channels are first
            # Note: the image is already normalized so its values are within
            #       [0, 1]
            np.transpose(img, [2, 0, 1])
        )
        img = self.transform(img)
        return img, metadata

    def __len__(self):
        return len(self.split_array_map)

    def __getitem__(self, idx):
        real_idx = self.split_array_map[idx]
        img, sample_meta = self.sample_array(real_idx)
        y = self._from_meta_to_label(sample_meta)
        c = self._from_meta_to_concepts(sample_meta)
        return (
            img,
            y,
            c,
            self.concept_transform(self.concept_names),
            self.task_names,
        )