"""A synthetic fleet of machines that wear out, with concepts over time, generated on first use."""
import os
from typing import List, Optional

import pandas as pd
import torch
import torch.nn.functional as F

from ...annotations import Annotations
from ..base import DynamicConceptDataset

ALPHA = torch.tensor([0.02, 0.03])      # wear acceleration of each component
ALARM_THRESHOLD = -0.45                 # the alarm fires below this mean health


def _wear(n, n_steps, n_sensors, seed):
    """The series of n machines, time first: inputs (n_steps, n, 2 + n_sensors) and
    concepts (n_steps, n, 3), with their names and graph."""
    g = torch.Generator().manual_seed(seed)
    # How each sensor responds to the conditions, and to health and load.
    base, gain = torch.randn(2, 2, n_sensors, generator=g)
    effect = torch.randn(4, n_sensors, generator=g)
    # The load of each machine, shown by the sensors only.
    load = 0.002 + 0.018 * torch.rand(n, 2, generator=g)
    c, health = -0.4 * torch.rand(n, 2, generator=g), []
    for _ in range(n_steps):
        health.append(c)
        c = c - ALPHA * F.relu(-c) - load
    health = torch.stack(health, 1)
    # Two operating conditions per step, drifting from a random start.
    start = 2 * torch.rand(n, 1, 2, generator=g) - 1
    conditions = (start + 0.15 * torch.randn(n, n_steps, 2, generator=g).cumsum(1)).clamp(-1, 1)
    # A reading: a baseline set by the conditions, plus the effect of health and
    # load, which the conditions amplify or damp.
    health_and_load = torch.cat([health, load.unsqueeze(1).expand(-1, n_steps, -1)], -1)
    baseline, response = torch.tanh(conditions @ base), torch.exp(conditions @ gain)
    sensors = baseline + response * (health_and_load @ effect)
    sensors = sensors + 0.05 * torch.randn(sensors.shape, generator=g)
    alarm = (health.mean(-1, keepdim=True) < ALARM_THRESHOLD).float()
    inputs = torch.cat([conditions, sensors], -1).transpose(0, 1).contiguous()
    concepts = torch.cat([health, alarm], -1).transpose(0, 1).contiguous()
    names = ["health_A", "health_B", "alarm"]
    graph = pd.DataFrame([[0, 0, 1], [0, 0, 1], [0, 0, 0]], index=names, columns=names)
    return inputs, concepts, names, graph


class WearDataset(DynamicConceptDataset):
    """Synthetic dataset of machines whose two components wear out, with concepts over time.

    Each of the ``n_gen`` machines runs for ``n_steps`` steps, and at every
    step has two operating conditions, noisy sensor readings, the health of
    its two components and an alarm. The health ``c`` of a component starts at
    or below 0 (0 is new), and at every step it drops by ``alpha * relu(-c)``
    plus a load of the machine, so a worn component wears faster. The sensors
    show the load, but no concept records it. The alarm fires when the mean
    health of the two components drops below ``ALARM_THRESHOLD = -0.45``.

    The dataset is modelled on N-CMAPSS. Like the flight conditions of an
    aircraft engine, two operating conditions drift from step to step and
    change every reading: its baseline, and how strongly it responds to health
    and load. Reading the health from the sensors means undoing the
    conditions, a nonlinear task.

    By default a sample is the whole series of one machine. Its inputs are
    ``(n_steps, 2 + n_sensors)``, the two conditions then the readings, and
    its concepts ``(n_steps, 3)`` hold ``health_A`` and ``health_B``
    (continuous) and ``alarm`` (binary). The graph is that of one step, where
    the health of each component drives the alarm; across steps, each health
    also depends on its value at the step before.

    Parameters
    ----------
    root : str, optional
        Directory to store and load the generated data. If None, defaults to
        ``'./data/toy_datasets/wear'``.
    seed : int, optional
        Random seed of the generation. Default: 42
    n_gen : int, optional
        Number of machines. Default: 1500
    n_steps : int, optional
        Number of steps of each series. Default: 20
    n_sensors : int, optional
        Number of sensor channels, besides the two conditions. Default: 14
    window, stride : int, optional
        Samples of ``window`` steps, one every ``stride`` steps, as in
        :class:`~torch_concepts.data.base.DynamicConceptDataset`. Default: None
        (the whole series).
    concept_subset : list of str, optional
        Subset of concept names to use. Default: None (all concepts).

    Attributes
    ----------
    alpha : torch.Tensor
        The wear acceleration of each component, of shape ``(2,)``.

    References
    ----------
    Arias Chao et al. "Aircraft Engine Run-to-Failure Dataset under Real
    Flight Conditions for Prognostics and Diagnostics", Data 2021.
    https://doi.org/10.3390/data6010005

    Examples
    --------
    >>> from torch_concepts.data import WearDataset
    >>> dataset = WearDataset(n_gen=100, n_steps=12)
    >>> dataset.input_data.shape
    torch.Size([100, 12, 16])
    >>> dataset.concepts[["health_A", "health_B"]].shape
    torch.Size([100, 12, 2])
    >>> WearDataset(n_gen=100, n_steps=12, window=4).input_data.shape  # 3 windows per machine
    torch.Size([300, 4, 16])
    """

    def __init__(
            self,
            root: str = None,
            seed: int = 42,
            n_gen: int = 1500,
            n_steps: int = 20,
            n_sensors: int = 14,
            window: Optional[int] = None,
            stride: Optional[int] = None,
            concept_subset: Optional[list] = None,
    ):
        self.seed, self.n_gen, self.n_steps, self.n_sensors = seed, n_gen, n_steps, n_sensors
        self.alpha = ALPHA.clone()
        if root is None:
            root = os.path.join(os.getcwd(), 'data', 'toy_datasets', 'wear')
        self.root = root
        input_data, concepts, annotations, graph = self.load()
        super().__init__(
            input_data=input_data,
            concepts=concepts,
            annotations=annotations,
            window=window,
            stride=stride,
            graph=graph,
            concept_names_subset=concept_subset,
        )

    @property
    def raw_filenames(self) -> List[str]:
        """Empty, since the data is generated."""
        return []

    @property
    def processed_filenames(self) -> List[str]:
        name = f"wear_N_{self.n_gen}_T_{self.n_steps}_D_{self.n_sensors}"
        return [f"{name}_seed_{self.seed}.pt"]

    def download(self):
        pass

    def build(self):
        input_data, concepts, names, graph = _wear(self.n_gen, self.n_steps, self.n_sensors, self.seed)
        types = ['binary' if name.startswith('alarm') else 'continuous' for name in names]
        annotations = Annotations(labels=names, cardinalities=[1] * len(names), types=types)
        torch.save({
            'input_data': input_data, 'concepts': concepts,
            'annotations': annotations, 'graph': graph,
        }, self.processed_paths[0])

    def load_raw(self):
        self.maybe_build()
        data = torch.load(self.processed_paths[0], weights_only=False)
        return data['input_data'], data['concepts'], data['annotations'], data['graph']
