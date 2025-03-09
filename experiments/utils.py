from typing import Any

import torch
import numpy as np
import random
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

