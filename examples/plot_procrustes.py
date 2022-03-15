"""
A simple example of the landing algorithm on Procrustes problem
===============================================================
Given n pairs of matrices in an array A and B, we want to solve
in parallel the procrustes problems min_X ||XA - B|| where X is
orthogonal. We compare Riemannian gradient descent with the
landing method.
Here, the bottleneck of the methods is not computing the gradients,
but rather moving on the manifold. Therefore, the landing algorithm
greatly accelerates convergence
"""
from time import time

import matplotlib.pyplot as plt

import torch
import geoopt
from geoopt.optim import RiemannianSGD

from landing import LandingSGD


torch.manual_seed(1)

# generate random matrices

n = 100
p = 40
A = torch.randn(n, p, p)
B = torch.randn(n, p, p)
init_weights = torch.randn(n, p, p)

# Compute closed-form solution from svd, used for monitoring.

u, _, v = torch.svd(B.matmul(A.transpose(-1, -2)))
w_star = u.matmul(v.transpose(-1, -2))
loss_star = ((torch.matmul(w_star, A) - B) ** 2).sum() / n
loss_star = loss_star.item()

method_names = ["Landing", "Retraction"]
methods = [LandingSGD, RiemannianSGD]

learning_rate = 0.3


f, axes = plt.subplots(2, 1)
for method_name, method, n_epochs in zip(method_names, methods, [2000, 500]):
    iterates = []
    loss_list = []
    time_list = []

    param = geoopt.ManifoldParameter(
        init_weights.clone(), manifold=geoopt.Stiefel(canonical=False)
    )
    with torch.no_grad():
        param.proj_()
    optimizer = method((param,), lr=learning_rate)
    t0 = time()
    for _ in range(n_epochs):

        optimizer.zero_grad()
        res = torch.matmul(param, A) - B
        loss = (res ** 2).sum() / n
        loss.backward()
        time_list.append(time() - t0)
        loss_list.append(loss.item() - loss_star)
        iterates.append(param.data.clone())
        optimizer.step()

    distance_list = []
    for matrix in iterates:
        d = (
            torch.norm(matrix.matmul(matrix.transpose(-1, -2)) - torch.eye(p))
            / n
        )
        distance_list.append(d.item())
    axes[0].semilogy(time_list, distance_list, label=method_name)
    axes[1].semilogy(time_list, loss_list, label=method_name)

axes[0].set_xlabel("time (s.)")
axes[1].set_xlabel("time (s.)")
axes[0].set_ylabel("Orthogonality error")
axes[1].set_ylabel("f - f^*")
plt.legend()
plt.show()
