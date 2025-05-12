# BMM Multitask Learning

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![test_and_docs](https://github.com/intsystems/bmm-multitask-learning/actions/workflows/test_and_deploy.yaml/badge.svg)](https://github.com/intsystems/bmm-multitask-learning/actions/workflows/test_and_deploy.yaml)

Multi-task learning is a machine learning paradigm which involves optimization model parameters for multiple diverse tasks. Vanilla multitask optimization presumes that the tasks are optimized without any hierarchy, but with some possilbe weights over tasks. This is an *educational project* which aims to implement different methods to assign weights or hierarchy to tasks to make the optimization more effective. We use *python* and [*pytorch*](https://docs.pytorch.org/docs/stable/index.html) as a computational framework.

## Implemented algorithms

* [Task clustering](https://jmlr.csail.mit.edu/papers/volume4/bakker03a/bakker03a.pdf)
* [Bayesian Multitask Learning with Latent Hierarchies](https://arxiv.org/pdf/1408.2032)
* [Sparse Bayesian Multi-Task Learning](https://proceedings.neurips.cc/paper_files/paper/2011/file/4fac9ba115140ac4f1c22da82aa0bc7f-Paper.pdf)
* [Variational method](https://proceedings.neurips.cc/paper_files/paper/2021/file/afd4836712c5e77550897e25711e1d96-Paper.pdf)

## Coverage report

Can be found on [pages](https://intsystems.github.io/bmm-multitask-learning/coverage/).

## Documentation

Current documentation is available on [pages](https://intsystems.github.io/bmm-multitask-learning/).

Key sections:

* API Reference
* Theory and Mathematical Background
* Application examples

## Repo structure

```bash
.
├── bmm_multitask_learning  # python package
├── docs
├── examples                # notebooks with package applications
├── LICENSE
├── mkdocs.yml
├── poetry.lock
├── pyproject.toml
├── README.md
├── Sketch_Multitask_Learning.pdf   # blogpost about implemented algorithms
└── tests
```

## Installation

The package uses *poetry* to manage dependencies and builds.

### Via poetry

If you use poetry, you can add the package as your dependency

```bash
poetry add git+https://github.com/intsystems/bmm-multitask-learning.git
```

### Ready builds

You can download ready builds from [github](TODO:add release link).

### Build from source

```bash
git clone https://github.com/intsystems/bmm-multitask-learning.git
cd bmm-multitask-learning
poetry build -o build_dir
```

Now you can add generated package to your envirement.

## Blog post

Currently available as [pdf](MTL_Blog_Post.pdf).

## Team

* Iryna Zabarianska
* Ilgam Latypov
* Alexander Terentyev
* Kirill Semkin
