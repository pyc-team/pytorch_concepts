import matplotlib.image as mpimg
import numpy as np

from scipy.ndimage import rotate

from .shared import SPRITES_DIRECTORY

################################################################################
## Load the sprites to memory
################################################################################

CARS_FILE = SPRITES_DIRECTORY('white_car.png')
AMBULANCE_FILE = SPRITES_DIRECTORY('ambulance.png')


################################################################################
## Helper Functions
################################################################################

def replace_white_color(
    img,
    new_color,
    thresh=10,
):
    scaling_factor = 255 if np.max(img) <= 1 else 1
    color_scaling_factor = 1 if np.max(new_color) <= 1 else 255
    replacement = (
        np.expand_dims(
            np.expand_dims(new_color, axis=0)/color_scaling_factor,
            axis=0,
        ) *
        np.ones_like(img[:, :, :3])
    )
    # And scale based on the closeness to white that the replaced pixels where
    replacement = img[:, :, :3] * replacement
    result = np.where(
        np.all(
            img[:, :, :3] >= (255 - thresh)/scaling_factor,
            axis=-1,
            keepdims=True,
        ),
        replacement,
        img[:, :, :3].copy(),
    )
    # Add the alpha channel
    result = np.concatenate([result, img[:, :, 3:].copy()], axis=-1)
    return result


################################################################################
## Construct Car Sprites
################################################################################

_WHITE_CAR = mpimg.imread(CARS_FILE)
_BLACK_CAR = replace_white_color(
    _WHITE_CAR,
    new_color=np.array([0, 0, 0]), # Black
    thresh=100,
)
_GREEN_CAR = replace_white_color(
    _WHITE_CAR,
    new_color=np.array([144, 238, 144]), # Soft green
    thresh=100,
)
_PINK_CAR = replace_white_color(
    _WHITE_CAR,
    new_color=np.array([255, 182, 193]), # Soft pink
    thresh=100,
)
_BLUE_CAR = replace_white_color(
    _WHITE_CAR,
    new_color=np.array([137, 207, 240]), # Baby blue
    thresh=100,
)
_PURPLE_CAR = replace_white_color(
    _WHITE_CAR,
    new_color=np.array([216, 191, 216]), # Pastel purple
    thresh=100,
)
_SILVER_CAR = replace_white_color(
    _WHITE_CAR,
    new_color=np.array([192, 192, 192]),
    thresh=100,
)
_BURGUNDY_CAR = replace_white_color(
    _WHITE_CAR,
    new_color=np.array([128, 0, 32]), # Pastel purple
    thresh=100,
)

################################################################################
## Aggregate Global Variables to Export
################################################################################

CAR_SPRITES = sorted(
    [
        dict(img=_WHITE_CAR, scale=0.4, color='white', ambulance=False),
        dict(img=_BLACK_CAR, scale=0.4, color='black', ambulance=False),
        dict(img=_GREEN_CAR, scale=0.4, color='green', ambulance=False),
        dict(img=_BLUE_CAR, scale=0.4, color='blue', ambulance=False),
        dict(img=_PURPLE_CAR, scale=0.4, color='purple', ambulance=False),
        dict(img=_SILVER_CAR, scale=0.4, color='silver', ambulance=False),
        dict(img=_BURGUNDY_CAR, scale=0.4, color='burgundy', ambulance=False),
        dict(img=_PINK_CAR, scale=0.4, color='pink', ambulance=False),
    ],
    key=lambda x: x['color'],
)

AVAILABLE_CAR_COLORS = [x['color'] for x in CAR_SPRITES]


################################################################################
## Construct Ambulance Sprite
################################################################################

AMBULANCE_IMG = mpimg.imread(AMBULANCE_FILE)
AMBULANCE_IMG = rotate(AMBULANCE_IMG, -90)
AMBULANCE = dict(
    img=AMBULANCE_IMG,
    scale=0.3,
    color='white',
    ambulance=True,
)


################################################################################
## Functions for exporting
################################################################################


def highlight_car(img, color, thresh=0.04):
    highlight_car = img.copy()
    if color == 'black':
        black_pixels = (
            (highlight_car[:, :, 0] <= thresh) &
            (highlight_car[:, :, 1] <= thresh) &
            (highlight_car[:, :, 2] <= thresh) &
            (highlight_car[:, :, 3] >= (1 - thresh))
        )
        highlight_car[black_pixels] = [1, 0, 0, 1]
    elif color == 'white':
        white_pixels = (
            (highlight_car[:, :, 0] <= (1 - thresh)) &
            (highlight_car[:, :, 1] <= (1 - thresh)) &
            (highlight_car[:, :, 2] <= (1 - thresh)) &
            (highlight_car[:, :, 3] <= (1 - thresh))
        )
        highlight_car[white_pixels] = [1, 0, 0, 1]
    return highlight_car

