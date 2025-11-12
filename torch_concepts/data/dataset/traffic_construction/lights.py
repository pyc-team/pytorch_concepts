import cv2
import matplotlib.image as mpimg
import numpy as np

from . import utils

from .shared import SPRITES_DIRECTORY

################################################################################
## Load the sprites to memory
################################################################################

LIGHTS_FILE = SPRITES_DIRECTORY('lights.png')

################################################################################
## Construct Light Sprites
################################################################################

img = mpimg.imread(LIGHTS_FILE)

RED_LIGHT = mpimg.imread(LIGHTS_FILE)
RED_LIGHT = RED_LIGHT[:, :RED_LIGHT.shape[1]//3, :]

YELLOW_LIGHT = mpimg.imread(LIGHTS_FILE)
YELLOW_LIGHT = \
    YELLOW_LIGHT[:, YELLOW_LIGHT.shape[1]//3:2*YELLOW_LIGHT.shape[1]//3, :]

GREEN_LIGHT = mpimg.imread(LIGHTS_FILE)
GREEN_LIGHT = GREEN_LIGHT[:, 2*GREEN_LIGHT.shape[1]//3:, :]

LIGHTS = [GREEN_LIGHT, YELLOW_LIGHT, RED_LIGHT]

_LEFT_TOP_CORNER = (400, 400)
_LEFT_BOTTOM_CORNER = (400, 860)
_RIGHT_TOP_CORNER = (875, 400)
_RIGHT_BOTTOM_CORNER = (875, 860)

################################################################################
## Helper functions
################################################################################


def add_circle_to_image(
    background,
    center,
    radius,
    color,
    vector_angle=None,
    arrow_color=(0, 0, 0),
    thickness=10,
    inplace=False,
    tip_length=0.4,
):
    if inplace:
        image = background
    else:
        image = background.copy()

    # Draw the circle
    cv2.circle(
        image[:, :, :3],
        center,
        radius,
        color,
        -1,  # -1 means filled circle
    )

    if vector_angle is not None:
        angle_rad = np.radians(vector_angle)
        start_x = int(center[0] - radius * np.cos(angle_rad))
        start_y = int(center[1] + radius * np.sin(angle_rad))
        end_x = int(center[0] + radius * np.cos(angle_rad))
        end_y = int(center[1] - radius * np.sin(angle_rad))

        cv2.arrowedLine(
            image[:, :, :3],
            (start_x, start_y),
            (end_x, end_y),
            arrow_color,
            thickness,
            tipLength=tip_length,
        )
    return image


################################################################################
## Functions to export
################################################################################


def add_light_x_axis(
    img,
    green,
    ratio=1,
    light_scale=1,
    inplace=False,
    circle_radius=50,
    use_lights_sprites=False,
    thickness=10,
):
    if use_lights_sprites:
        # Then let's use the light sprite
        light = LIGHTS[0] if green else LIGHTS[-1]
        x_offset, y_offset = 330, 850
        x_scale_shift, y_scale_shift = 0, 0
        if light_scale != 1:
            x_scale_shift = -int(0.1 * light.shape[0] * (light_scale - 1))
            y_scale_shift = int(0.1 * light.shape[1] * (light_scale - 1))
        x_offset += x_scale_shift
        y_offset += y_scale_shift
        if ratio != 1:
            x_offset, y_offset = utils.transform_scale_coordinates(
                x_offset,
                y_offset,
                ratio=ratio,
            )
            light = utils.resize_with_aspect_ratio(
                light,
                target_height=ratio,
            )
        img = utils.add_sprite(
            sprite=light,
            background=img,
            target_width=0.1 * light_scale,
            rotation=270,
            x_offset=x_offset,
            y_offset=y_offset,
            inplace=inplace,
        )
        x_offset, y_offset = 840, 360
        x_scale_shift, y_scale_shift = 0, 0
        if light_scale != 1:
            x_scale_shift = int(0.1 * light.shape[0] * (light_scale - 1))
            y_scale_shift = -int(0.1 * light.shape[1] * (light_scale - 1))
        x_offset += x_scale_shift
        y_offset += y_scale_shift
        if ratio != 1:
            x_offset, y_offset = utils.transform_scale_coordinates(
                x_offset,
                y_offset,
                ratio=ratio,
            )
        img = utils.add_sprite(
            sprite=light,
            background=img,
            target_width=0.1 * light_scale,
            rotation=90,
            x_offset=x_offset,
            y_offset=y_offset,
            inplace=inplace,
        )
        return img
    # Else, we will build a simple circule with the given radius to the image
    r = int(circle_radius * ratio * light_scale)
    color = (0, 1, 0, 1) if green else (1, 0, 0, 1)
    x_offset, y_offset = _LEFT_BOTTOM_CORNER
    if ratio != 1:
        x_offset, y_offset = utils.transform_scale_coordinates(
            x_offset,
            y_offset,
            ratio=ratio,
        )

    img = add_circle_to_image(
        background=img,
        center=(x_offset - r, y_offset + r),
        radius=r,
        color=color,
        inplace=inplace,
        vector_angle=0,
        thickness=max(1, int(thickness * ratio)),
    )

    x_offset, y_offset = _RIGHT_TOP_CORNER
    if ratio != 1:
        x_offset, y_offset = utils.transform_scale_coordinates(
            x_offset,
            y_offset,
            ratio=ratio,
        )
    img = add_circle_to_image(
        background=img,
        center=(x_offset + r, y_offset - r),
        radius=r,
        color=color,
        inplace=inplace,
        vector_angle=180,
        thickness=max(1, int(thickness * ratio)),
    )
    return img


def add_light_y_axis(
    img,
    green,
    ratio=1,
    light_scale=1,
    inplace=False,
    circle_radius=50,
    use_lights_sprites=False,
    thickness=10,
):
    if use_lights_sprites:
        # Then proceed with the light sprite
        light = LIGHTS[0] if green else LIGHTS[-1]
        x_offset, y_offset = 850, 850
        x_scale_shift, y_scale_shift = 0, 0
        if light_scale != 1:
            x_scale_shift = -int(0.1 * light.shape[0] * (light_scale - 1))
            y_scale_shift = -int(0.1 * light.shape[1] * (light_scale - 1))
        x_offset += x_scale_shift
        y_offset += y_scale_shift
        if ratio != 1:
            x_offset, y_offset = utils.transform_scale_coordinates(
                x_offset,
                y_offset,
                ratio=ratio,
            )
            light = utils.resize_with_aspect_ratio(
                light,
                target_height=ratio,
            )
        img = utils.add_sprite(
            sprite=light,
            background=img,
            target_width=0.1*light_scale,
            rotation=0,
            x_offset=x_offset,
            y_offset=y_offset,
            inplace=inplace,
        )
        x_offset, y_offset = 370, 300
        if ratio != 1:
            x_offset, y_offset = utils.transform_scale_coordinates(
                x_offset,
                y_offset,
                ratio=ratio,
            )
        x_scale_shift, y_scale_shift = 0, 0
        if light_scale != 1:
            x_scale_shift = int(0.1 * light.shape[0] * (light_scale - 1))
            y_scale_shift = -int(0.1 * light.shape[1] * (light_scale - 1))
        x_offset += x_scale_shift
        y_offset += y_scale_shift
        img = utils.add_sprite(
            sprite=light,
            background=img,
            target_width=0.1*light_scale,
            rotation=180,
            x_offset=x_offset,
            y_offset=y_offset,
            inplace=inplace,
        )
        return img

    # Else, we will build a simple circule with the given radius to the image
    r = int(circle_radius * ratio * light_scale)
    color = (0, 1, 0, 1) if green else (1, 0, 0, 1)
    x_offset, y_offset = _LEFT_TOP_CORNER
    if ratio != 1:
        x_offset, y_offset = utils.transform_scale_coordinates(
            x_offset,
            y_offset,
            ratio=ratio,
        )

    img = add_circle_to_image(
        background=img,
        center=(x_offset - r, y_offset - r),
        radius=r,
        color=color,
        inplace=inplace,
        vector_angle=270,
        thickness=max(1, int(thickness * ratio)),
    )

    x_offset, y_offset = _RIGHT_BOTTOM_CORNER
    if ratio != 1:
        x_offset, y_offset = utils.transform_scale_coordinates(
            x_offset,
            y_offset,
            ratio=ratio,
        )
    img = add_circle_to_image(
        background=img,
        center=(x_offset + r, y_offset + r),
        radius=r,
        color=color,
        inplace=inplace,
        vector_angle=90,
        thickness=max(1, int(thickness * ratio)),
    )
    return img
