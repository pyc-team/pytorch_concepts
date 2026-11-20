"""OpenAI CLIP adapter tests without model downloads or the optional package."""
import sys
import types
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from torch_concepts import ImageBackbone


@pytest.fixture
def clip_source(monkeypatch):
    class Visual(nn.Module):
        output_dim = 3

        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(2.))

        def forward(self, x):
            return x.mean((2, 3)) * self.scale

    def preprocess(image):
        return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255

    visual = Visual()
    load = Mock(return_value=(types.SimpleNamespace(visual=visual), preprocess))
    monkeypatch.setitem(sys.modules, 'clip', types.SimpleNamespace(load=load))
    return load, visual


def test_preprocessing_output_dimensions_and_frozen_backbone(clip_source):
    load, visual = clip_source
    model = ImageBackbone('RN50', source='openai_clip', device='cpu')
    pixels = torch.tensor([[[0., .5], [1., .25]]]).expand(3, -1, -1).clone()
    batch = pixels.unsqueeze(0).repeat(2, 1, 1, 1)
    result = model(batch)
    expected = (batch * 255).round().div(255).mean((2, 3)) * 2
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(model(pixels), expected[0])
    image = Image.fromarray((pixels.permute(1, 2, 0).numpy() * 255).round().astype('uint8'))
    torch.testing.assert_close(model([image]), expected[:1])
    load.assert_called_once_with('RN50', device=torch.device('cpu'))
    assert model.out_features == 3
    assert result.dtype == torch.float32
    assert all(not p.requires_grad for p in model.parameters())
    model.train()
    assert not model.training and not visual.training


def test_missing_openai_clip_error(monkeypatch):
    monkeypatch.setitem(sys.modules, 'clip', None)
    with pytest.raises(ImportError, match='clip'):
        ImageBackbone('RN50', source='openai_clip', device='cpu')
