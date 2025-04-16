import os
from typing import Any

import torch
import numpy as np
import random

from packaging import version
if version.parse(torch.__version__) < version.parse("2.0.0"):
    # Then we will use pytorch lightning's version compatible with PyTorch < 2.0
    from pytorch_lightning.callbacks import TQDMProgressBar
else:
    from lightning.pytorch.callbacks import TQDMProgressBar


def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class CustomProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        # Override this method to disable the validation progress bar
        return None  # Returning None disables the validation progress display

    def on_validation_end(self, *args, **kwargs) -> None:
        pass

    def on_validation_batch_start(self, *args, **kwargs) -> None:
        pass

    def on_validation_batch_end(self, *args, **kwargs) -> None:
        pass


class GaussianNoiseTransform(object):

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean


def model_trained(model, model_name, file, load_results=True):
    if not os.path.exists(file) or not load_results:
        if os.path.exists(file):
            print("Model already trained, but not loading results. "
                  "Retraining.")
            os.remove(file)
        return False
    else:
        try:
            model.load_state_dict(torch.load(file)['state_dict'])
            print(
                f"Model {model_name} already trained, skipping training.")
            return True
        except RuntimeError:
            os.remove(file)
            print("Error loading model, training again.")
            return False