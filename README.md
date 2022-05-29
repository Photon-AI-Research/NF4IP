# NF4IP

Normalising Flows for Inverse Problems (NF4IP) is a Python framework that provides all means for usage of normalising flows in solving inverse problems. The framework allows low-key usage of recent advances in machine learning for analysis of experimental data based on Surrogate Models as well as one-step solution to inverse problems.
<!--
Applications of this framework can be found here: [Photon AI Research](https://http://photon-ai-research.github.io/).
- Injection Radiation
- SAXS
- GI-SAXS
--> 

The framework is being collaboratively developed by
- [Photon AI Research](https://http://photon-ai-research.github.io/)
- [HZDR's AI Consultant Team](https://www.hzdr.de/db/Cms?pOid=60710&pNid=0) for MATTER research
- [Helm & Walther IT Solutions](https://helmundwalter.de)


Maintainer:
- [Nico Hoffmann](mailto:n.hoffmann@hzdr.de) of PAIR

# Introduction
The aim of this project is to make it easier to solve inverse problems.
NF4IP is based on [FrEIA from Heidelberg University](https://github.com/VLL-HD/FrEIA), though
while FrEIA provides a toolbox for building invertible networks, NF4IP uses this toolbox 
to offer a selection of ready-to-use network architectures tailored to inverse problems typically encountered in physics.

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
git clone https://github.com/Photon-AI-Research/NF4IP.git
cd NF4IP
pip install -e setup.py
```
The -e will make an editable install, so you can edit the NF4IP library any time and see the effect instantly.

<!--
pip install -r requirements.txt // we need to make this file first :D
-->

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

# Framework Developer Instructions
These instructions are for anyone interested in improving NF4IP itself. The following steps are not necessary if you want to develop an application based on NF4IP. This project includes a number of helpers in the `Makefile` to streamline common development tasks.

 
## Environment Setup

The following demonstrates setting up and working with a development environment:

```
### create a virtualenv for development

$ make virtualenv

$ source env/bin/activate


### run NF4IP cli application

$ NF4IP --help


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
