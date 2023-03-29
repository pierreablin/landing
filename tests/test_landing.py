import pytest

import torch
import geoopt

from landing import LandingSGD

torch.manual_seed(1)


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


@pytest.mark.parametrize("safe_step", [0.3, 0.1])
def test_forward(safe_step, n_reps=10):
    p = 3
    shape = (p, p)
    for _ in range(n_reps):
        param = geoopt.ManifoldParameter(torch.randn(*shape), manifold=geoopt.Stiefel())
        param.requires_grad = False
        param.proj_()
        param.requires_grad = True
        optimizer = LandingSGD((param,), lr=0.1, safe_step=safe_step)
        optimizer.zero_grad()
        loss = (param ** 2).sum()
        loss.backward()
        optimizer.step()
        orth_error = torch.norm(param.mm(param.t()) - torch.eye(p)) ** 2
        assert orth_error < safe_step


def test_convergence():
    p = 3
    param = geoopt.ManifoldParameter(torch.randn(p, p), manifold=geoopt.Stiefel())
    optimizer = LandingSGD((param,), lr=.1)
    n_epochs = 100
    # Trace maximization: should end up in identity
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = -torch.trace(param)
        loss.backward()
        optimizer.step()
        
    assert loss.item() + p < 1e-5
    orth_error = torch.norm(param.mm(param.t()) - torch.eye(p))
    assert orth_error < 1e-5 
