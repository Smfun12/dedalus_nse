# AOT in Python

Dedalus is a Python package for solving partial differential equations (PDE) using pseudo-spectral code. This repository contains a data assimilation (DA) code for solving Navier-Stokes equation (NSE) using Azounai-Olson-Titi (AOT) continuous DA [algorithm](http://arxiv.org/abs/1304.0997). The ground-truth solution is given by:

![equation](https://latex.codecogs.com/svg.image?\mathbf{u}_t=F(\mathbf{u})).

The AOT solution is written as:

![equation](https://latex.codecogs.com/svg.image?\mathbf{v}_t=F(\mathbf{v})&plus;\mu&space;I_h(\mathbf{u}-\mathbf{v})).

The goal of DA is by using model <b>v</b> and sparse observations <b>I_h*u</b> recover true solution <b>u</b>.

The current goal is to explore different observations strategies, e.g., sensor movement to improve convergence speed.
