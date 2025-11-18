import matplotlib.image as mpimg

from .shared import SPRITES_DIRECTORY

################################################################################
## Load the sprites to memory
################################################################################

_INTERSECTION_FILE = SPRITES_DIRECTORY('single_lane_road_intersection.png')


################################################################################
## Generate the Intersection Background
################################################################################

INTERSECTION = mpimg.imread(_INTERSECTION_FILE)


################################################################################
## Generate a Library of Lanes With Their Metadata
################################################################################


_NORTHBOUND_LANES =[
    dict(
        bounds=((650, 820), (0, 400)),
        before_int=False,
        idx=0,
        abs_pos='north',
    ),
    dict(
        bounds=((650, 820), (870, 1280)),
        before_int=True,
        idx=3,
        abs_pos='south',
    ),
]
_EASTBOUND_LANES =[
    dict(
        bounds=((0, 400), (650, 810)),
        before_int=True,
        idx=5,
        abs_pos='west',
    ),
    dict(
        bounds=((860, 1280), (650, 810)),
        before_int=False,
        idx=2,
        abs_pos='east',
    ),
]
_WESTBOUND_LANES =[
    dict(
        bounds=((0, 400), (450, 610)),
        before_int=False,
        idx=6,
        abs_pos='west',
    ),
    dict(
        bounds=((860, 1280), (450, 610)),
        before_int=True,
        idx=1,
        abs_pos='east',
    ),
]
_SOUTHBOUND_LANES =[
    dict(
        bounds=((460, 610), (0, 400)),
        before_int=True,
        idx=7,
        abs_pos='north',
    ),
    dict(
        bounds=((460, 610), (850, 1280)),
        before_int=False,
        idx=4,
        abs_pos='south',
    ),
]



AVAILABLE_LANES = (
    [
        dict(
            bounds=x['bounds'],
            before_int=x['before_int'],
            idx=x['idx'],
            dir='north',
            rot=0,
            x_off=0,
            y_off=50,
            abs_pos=x['abs_pos'],
        )
        for x in _NORTHBOUND_LANES
    ] +
    [
        dict(
            bounds=x['bounds'],
            before_int=x['before_int'],
            idx=x['idx'],
            dir='south',
            rot=180,
            x_off=0,
            y_off=50,
            abs_pos=x['abs_pos'],
        )
        for x in _SOUTHBOUND_LANES
    ] +
    [
        dict(
            bounds=x['bounds'],
            before_int=x['before_int'],
            idx=x['idx'],
            dir='east',
            rot=270,
            x_off=50,
            y_off=0,
            abs_pos=x['abs_pos'],
        )
        for x in _EASTBOUND_LANES
    ] +
    [
        dict(
            bounds=x['bounds'],
            before_int=x['before_int'],
            idx=x['idx'],
            dir='west',
            rot=90,
            x_off=50,
            y_off=0,
            abs_pos=x['abs_pos'],
        )
        for x in _WESTBOUND_LANES
    ] +
    []
)
