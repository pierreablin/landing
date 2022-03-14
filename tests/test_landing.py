import pytest

import torch
import geoopt

from landing import LandingSGD


@pytest.mark.parametrize("momentum", [0, 0.5])
@pytest.mark.parametrize("shape", [(3, 3), (4, 3, 3), (5, 4, 3, 3)])
@pytest.mark.parametrize("safe_step", [0.3, False])
def test_forward(shape, momentum, safe_step):
    param = geoopt.ManifoldParameter(torch.randn(*shape), manifold=geoopt.Stiefel())
    optimizer = LandingSGD((param,), lr=0.1, momentum=momentum, safe_step=safe_step)
    optimizer.zero_grad()
    loss = (param ** 2).sum()
    loss.backward()
    optimizer.step()
