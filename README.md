# Fast optimization on the orthogonal manifold with the landing flow


This repository contains the code for the landing algorithm, an algorithm for optimization under orthogonal constraint which does not use retractions. One iteration of the algorithm is faster than retraction based algorithms: it can be used to accelerate optimization under orthogonal constraint when the computational bottleneck is computing the retraction.

You can find the paper [here](https://arxiv.org/pdf/2102.07432.pdf).

![illustration](https://github.com/pierreablin/landing/blob/main/illustration.jpg?raw=true)


## Installation

To install the package, simply run
```
pip install git+https://github.com/pierreablin/landing.git
```

## Use

The package ships one optimizer, `LandingSGD`, that mimics geoopt's `RiemannianSGD`. It is a pytorch optimizer.

## Cite

If you use this code please cite:

    Pierre Ablin and Gabriel Peyré.
    "Fast and accurate optimization on the orthogonal manifold without retraction."
    AISTATS 2022