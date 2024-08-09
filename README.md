# SOBMOR

Demo code for Structured optimization based model order reduction (SOBMOR)

This repository contains the reference implementation for SOBMOR and can be used to fit the transfer functions of structured dynamical systems to given frequency response data or given transfer functions. The repository contains two demo driver scripts that showcase the effectiveness of the approach by fitting a low-order port-Hamiltonian system using SOBMOR and a general parametric system, respectively. An introduction to the mathematical ideas of SOBMOR is given [here](https://www.doi.org/10.1137/20M1380235), adaptive sampling is discussed [here](https://www.doi.org/10.1016/j.ifacol.2021.11.069), and the extension to the general parametric case is discussed [here](https://arxiv.org/abs/2209.05101).

## Installation and running the demo scripts

1. Ensure julia-1.10 is installed on your machine
2. Clone this repository using `git clone git@github.com:Algopaul/SOBMOR.git`
3. Enter the `SOBMOR`-directory 
4. Run `bash configure.sh`
5. Run `make install`
4. Run `make demo_portHamiltonian20` or `make demo_parametricMOR` to execute the corresponding demo scripts

## General information about the codebase

SOBMOR is based on minimizing singular values of matrix compositions with respect to a parameter vector that a subset of these matrices depend on. This code allows for the definition of such matrix compositions and provides basic building blocks for defining positive semi-definite, skew-symmetric, diagonal, and general rectangular matrices. Moreover, the code provides compositions that are often used to define transfer functions such as matrix multiplication, addition, subtraction, and linear solves. Each building block or composition implements an update and a `tan!` routine, where the update routine simply updates the matrix(-composition), while `tan!(g, u, v)`, writes into `g` the gradient of the singular value associated with the left and right singular vectors `u` and `v` with respect to the parameter vector of the matrix(-composition). The provided compositions compute appropriate modifications to the provided singular vectors and pass these on to their components.

Building blocks and compositions are defined as functors, such that all workspace can be allocated when the compositions are constructed and all information about the sizes of the different matrices and parameter vectors is available when julia's just-in-time-compilation is executed. In this way, during the optimization, no workspace needs to be allocated repeatedly.

If you want to test your own structures (e.g. you want to optimize a second order system or a delay differential-algebraic system), my recommendation is to first run the two driver scripts with different flag-values and add your own target data to fit to familiarize yourself with method. Then look at `src/systems.jl` to understand how different types of dynamical systems can be constructed from the provided building blocks and compositions. You can use the function `test_deriv` in the test code to ensure your derivative computation is correct and your functors do not allocate any memory.

Moreover, I recommend trying the python frameworks `jax` and `flax`, which provide a **much easier** interface to defining your own matrix structures and computing derivatives with respect to given parameters. The code I provide here exploits certain rank-one structures that occur in matrix derivatives, when only a subset of singular values is computed and tends to be faster than `jax` **for my use-cases and on my machine**. However, `jax` automatically compiles your models to efficient GPU-code and adds parallelization when appropriate so depending on your setup `jax` may also be faster.

## Citation

If you want to refer to this code, you can use the following citations:

#### General Structured optimization-based model order reduction
```latex
@Article{SchwerdtnerV2023SOBMOR,
    author	= {Schwerdtner, Paul and Voigt, Matthias},
    doi		= {10.1137/20M1380235},
    journal	= {SIAM J. Sci. Comput.},
    number	= {2},
    pages	= {A502-A529},
    title	= {{SOBMOR}: Structured Optimization-Based Model Order Reduction},
    url		= {https://doi.org/10.1137/20M1380235},
    volume	= {45},
    year	= {2023},
}

@Article{SchwerdtnerMMV2023Optimization-based,
    title = {Optimization-based model order reduction of port-{H}amiltonian descriptor systems},
    journal = {Systems Control Lett.},
    volume = {182},
    pages = {105655},
    year = {2023},
    issn = {0167-6911},
    doi = {10.1016/j.sysconle.2023.105655},
    url = {https://www.sciencedirect.com/science/article/pii/S0167691123002025},
    author = {Paul Schwerdtner and Tim Moser and Volker Mehrmann and Matthias Voigt},
}
```

#### Adaptive sampling routine
```latex
@Article{SchwerdtnerV2021Adaptive,
    title	= {Adaptive Sampling for Structure-Preserving Model Order Reduction of Port-{H}amiltonian Systems},
    author	= {Schwerdtner, P. and Voigt, M.},
    journal	= {IFAC-PapersOnline},
    volume	= {54},
    number	= {19},
    pages	= {143--148},
    year	= {2021},
    doi		= {10.1016/j.ifacol.2021.11.069}
}
```
#### Parametric model order reduction
```latex
@Article{SchwerdtnerS2022Structured,
    author	= {Paul Schwerdtner and Manuel Schaller},
    title	= {Structured Optimization-Based Model Order Reduction for Parametric Systems},
    journal	= {Preprint},
    year	= {2022},
    number	= {arXiv:2209.05101},
    url		= {https://arxiv.org/abs/2209.05101}
}
```
