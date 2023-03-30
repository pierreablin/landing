import pytest

import torch
import geoopt

from landing import LandingSGD

torch.manual_seed(1)


@pytest.mark.parametrize("momentum", [0, 0.5])
@pytest.mark.parametrize("shape", [(3, 3), (4, 3, 3), (5, 4, 3, 3)])
@pytest.mark.parametrize("safe_step", [0.3, False])
def test_forward(shape, momentum, safe_step):
    param = geoopt.ManifoldParameter(
        torch.randn(*shape), manifold=geoopt.Stiefel()
    )
    optimizer = LandingSGD(
        (param,), lr=0.1, momentum=momentum, safe_step=safe_step
    )
    optimizer.zero_grad()
    loss = (param**2).sum()
    loss.backward()
    optimizer.step()


@pytest.mark.parametrize("safe_step", [0.3, 0.1, 1e-2])
@pytest.mark.parametrize("lbda", [0.1, 1, 10])
@pytest.mark.parametrize("n_features", [2, 10])
def test_safe(safe_step, lbda, n_features, n_reps=10, n_iters=100, tol=1e-6):
    p = n_features
    shape = (p, p)
    for _ in range(n_reps):
        param = geoopt.ManifoldParameter(
            torch.randn(*shape), manifold=geoopt.Stiefel()
        )
        # param.requires_grad = False
        # param.proj_()
        param.requires_grad = True
        target = torch.randn(*shape)
        # take large lr so that the safe step always triggers
        optimizer = LandingSGD(
            (param,), lr=1e5, safe_step=safe_step, lambda_regul=lbda
        )
        for n_iter in range(n_iters):
            # print(param)
            optimizer.zero_grad()
            loss = (param * target).sum()
            loss.backward()
            optimizer.step()
            # print(param)
            orth_error = torch.norm(param.t().mm(param) - torch.eye(p))
            assert orth_error < safe_step + tol


def test_convergence():
    p = 3
    param = geoopt.ManifoldParameter(
        torch.eye(p) + 0.1 * torch.randn(p, p), manifold=geoopt.Stiefel()
    )
    optimizer = LandingSGD((param,), lr=0.1)
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
