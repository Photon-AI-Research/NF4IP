# NF4IP

Normalising Flows for Inverse Problems (NF4IP) is a Python framework that provides all means for usage of normalising flows in solving inverse problems. The framework allows low-key usage of recent advances in machine learning for analysis of experimental data based on Surrogate Models as well as one-step solution to inverse problems.

Main contributions of this framework will be

1. Low-key training of normalising flows for surrogate modeling and data analysis
2. WIP: Data analysis by
 - Gradient Descent
 - Approximate Bayesian Computation
 - Bayesian Inference

Example usage of this framework can be found [Photon AI Research](https://http://photon-ai-research.github.io/).

The framework is being developed collaboratively by
- [Photon AI Research](https://http://photon-ai-research.github.io/)
- [Helmholtz AI Consultant Team](https://www.hzdr.de/db/Cms?pOid=60710&pNid=0) for MATTER research
- [Helm & Walther IT Solutions](https://helmundwalter.de)


Maintainer:
- Dr. [Nico Hoffmann](mailto:n.hoffmann@hzdr.de) of PAIR

# Introduction
The aim of this project is to make it easier to solve inverse problems.
NF4IP Based on [FrEIA from Heidelberg University](https://github.com/VLL-HD/FrEIA). 
While FrEIA provides a toolbox for building invertible networks, NF4IP uses this toolbox 
to offer a selection of ready-to-use network architectures.

In addition, NF4IP already offers useful features that simplify the development 
of your machine learning application:

* powerful configuration options with defaults, config files and commandline arguments.
* a code generator to give you a quick start with your project
* checkpointing: automatically save your learning progress and restore from it.
* Filter/Hook system to customize the application and plug your code into the library
* Plugin/Extension System to extend the functionality
* These extensions are currently available:
    * NF implementation
    * Tensorboard logging
    * Variational Autoencoder

# Installation
To install NF4IP, clone this repository, enter the cloned directory and install it locally using pip:
```bash
git clone [https://github.com/Photon-AI-Research/NF4IP.git](https://github.com/Photon-AI-Research/NF4IP.git)
cd NF4IP
pip install -r requirements.txt
pip install -e setup.py
```
The -e will make an editable install, so you can edit the NF4IP library any time and see the effect instantly.

# Getting started

You can start generating your first project by executing:
```
NF4IP generate project <targetDir>
```
where <targetDir> is the directory where the new project will be generated.
This tutorial assumes that you used the default "myapp" label name.

After executing this command and answering all questions, you can open the targetDir in your favorite
IDE. If you have no preference, we recommend the free
[pyCharm community edition](https://www.jetbrains.com/de-de/pycharm/).

The directory structure created is basicly a [Cement 3.0](https://docs.builtoncement.com)-Project.
While Cement has its own documentation, which should be consulted if there are open questions,
the NF4IP documentation will already give you everything you need to know to start your first project.

# Developer Instruction for Extending NF4IP

## Installation

```
$ pip install -r requirements.txt

$ pip install setup.py
```

## Development

This project includes a number of helpers in the `Makefile` to streamline common development tasks.

### Environment Setup

The following demonstrates setting up and working with a development environment:

```
### create a virtualenv for development

$ make virtualenv

$ source env/bin/activate


### run NF4IP cli application

$ NF4ip --help


### run pytest / coverage

$ make test
```


### Releasing to PyPi

Before releasing to PyPi, you must configure your login credentials:

**~/.pypirc**:

```
[pypi]
username = YOUR_USERNAME
password = YOUR_PASSWORD
```

Then use the included helper function via the `Makefile`:

```
$ make dist

$ make dist-upload
```

## Deployments

### Docker

Included is a basic `Dockerfile` for building and distributing `NF4IP`,
and can be built with the included `make` helper:

```
$ make docker

$ docker run -it NF4IP --help
```
