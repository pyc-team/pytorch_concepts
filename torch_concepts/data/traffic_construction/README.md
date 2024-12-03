# Traffic Dataset Sprites Attribution
This directory contains all the necessary files and scripts to generate our
traffic toy dataset (and variations of it).


## Generating the dataset

To construct the dataset, please run the script `generate_data.py` **from this
directory**. This script has the following options:
```bash
$ python generate_data.py --help
usage: generate_data.py [-h] [--dir_name DIR_NAME] [--sym_link_name SYM_LINK_NAME] [--num_threads NUM_THREADS] [--as_arrays] [--n_samples N_SAMPLES] [--seed SEED] [--position_para_noise POSITION_PARA_NOISE] [--position_perp_noise POSITION_PERP_NOISE] [--error_probability ERROR_PROBABILITY]
                        [--p_ambulance P_AMBULANCE] [--thickness THICKNESS] [--min_num_cars MIN_NUM_CARS] [--max_num_cars MAX_NUM_CARS] [--resize_final_image RESIZE_FINAL_IMAGE] [--car_colors CAR_COLORS [CAR_COLORS ...]]
                        [--possible_starting_directions POSSIBLE_STARTING_DIRECTIONS [POSSIBLE_STARTING_DIRECTIONS ...]] [--train_ratio TRAIN_RATIO] [--val_ratio VAL_RATIO] [--test_ratio TEST_RATIO] [--light_scale LIGHT_SCALE] [--use_lights_sprites] [--rerun] [--use_absolute_path] [--visual_validation]

Generate synthetic data for traffic simulation

optional arguments:
  -h, --help            show this help message and exit
  --dir_name DIR_NAME   Directory where the output data folders will be generated
  --sym_link_name SYM_LINK_NAME
                        Name of symlink to use for the generated dataset
  --num_threads NUM_THREADS
                        Number of processes to use during generation
  --as_arrays           Store all data into a single array if set, otherwise serialize each image with metadata
  --n_samples N_SAMPLES
                        Number of samples to generate
  --seed SEED           Seed to use for generation
  --position_para_noise POSITION_PARA_NOISE
                        Noise on the position of each car parallel to movement direction
  --position_perp_noise POSITION_PERP_NOISE
                        Noise on the position of each car perpendicular to movement direction
  --error_probability ERROR_PROBABILITY
                        Probability of a car making an error (breaking the law)
  --p_ambulance P_AMBULANCE
                        Probability of an ambulance being in the sample
  --thickness THICKNESS
                        Thickness of marker edge for the target car
  --min_num_cars MIN_NUM_CARS
                        Minimum number of cars per image (not including target car)
  --max_num_cars MAX_NUM_CARS
                        Maximum number of cars per image (not including target car)
  --resize_final_image RESIZE_FINAL_IMAGE
                        Downsampling factor for the final image
  --car_colors CAR_COLORS [CAR_COLORS ...]
                        List of car colors to consider (e.g., red, blue)
  --possible_starting_directions POSSIBLE_STARTING_DIRECTIONS [POSSIBLE_STARTING_DIRECTIONS ...]
                        Possible starting directions for the target car
  --train_ratio TRAIN_RATIO
                        Fraction of samples for the training set
  --val_ratio VAL_RATIO
                        Fraction of samples for the validation set
  --test_ratio TEST_RATIO
                        Fraction of samples for the test set
  --light_scale LIGHT_SCALE
                        Scaling for size of traffic lights (can be less or more than 1).
  --use_lights_sprites  If set, then we will default to use traffic light sprites over simple colored circles to represent traffic light states.
  --rerun               Rerun generation even if the dataset was already generated with the same config
  --use_absolute_path   If given, the sym-link of the generated dataset will be done using an absolute path rather than a relative path. This is useful if you want to generate and use this dataset locally but it is not recommended if the dataset is to be exported to other directories.
  --visual_validation   If set, then we will first generate some example images before generating the entire dataset upon inspection by the user.
```

Notice that if the dataset is already cached/previously generated with the same
config, then **it will not be regenerated** unless the `--rerun` flag is set.

## Image Attribution

When constructing our traffic dataset, we made use of the following images with
their corresponding licenses:

- **sprites/ambulance.png**: top view of car PNG Designed By 58pic from [Png Tree](https://pngtree.com/freepng/ambulance-car-top-view-vector-ps-element-car-cartoon_7030589.html?sol=downref&id=bef). Distributed under a free license if not used for commercial purposes.

- **sprites/lights.png**: Taken from [PNG Egg](https://www.pngegg.com/en/png-zyrff). Distributed under a non-commercial use license.

- **sprites/single_lane_road_intersection.png**: Credit to [Jean Lambert Salvatori](https://pixabay.com/vectors/street-intersection-double-lane-8188557/). Free for use under the [Pixabay Content License](https://pixabay.com/service/license-summary/).

- **sprites/white_black_car.png**: Credit to [Mahendran](https://www.cleanpng.com/users/@mahendran.html). Free/no attribution license from [CleanPNG](https://www.cleanpng.com/png-top-view-plan-view-238/).

## Citation

If you use this dataset as part of your work, we would appreciate it if you could
cite us as follows:

```
Barbiero P., Ciravegna G., Debot D., Diligenti M.,
Dominici G., Espinosa Zarlenga M., Giannini F., Marra G. (2024).
Concept-based Interpretable Deep Learning in Python.
https://pyc-team.github.io/pyc-book/intro.html
```

Or with the following bibtex entry:

```
@book{pycteam2024concept,
  title      = {Concept-based Interpretable Deep Learning in Python},
  author     = {Pietro Barbiero, Gabriele Ciravegna, David Debot, Michelangelo Diligenti, Gabriele Dominici, Mateo Espinosa Zarlenga, Francesco Giannini, Giuseppe Marra},
  year       = {2024},
  url        = {https://pyc-team.github.io/pyc-book/intro.html}
}
```