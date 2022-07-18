<img src="./images/pi_design.jpeg" alt="Logo of the project" align="right" width="200">

# Optimal Experimental Design (OED) &middot; [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![DOI:10.1007/978-3-319-76207-4_15](https://zenodo.org/badge/DOI/xxx.svg)](https://doi.org/) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()
> Optimal design of experiment, machine learning, Fisher information

This project provides
- convenient interfaces for designs of experiments, metrics to evaluate them, functions and statistical models
- a library for statistical models, designs of experiments and metrics
- in particular the pi-design
- a pipeline for benchmarking and comparing different designs for different scenarios
- jupyter notebooks to easily run and learn how to use this projects content

This project is closely related to the "Parameter Individual Optimal
Experimental Design and Calibration of
Parametric Models" paper (DOI:10.1007/978-3-319-76207-4_15 Maintenance) where the background and mathematical framework used here is explained in detail.
It also contains the notebook and its html used for the simulations in the paper. 

## Installing 

Run in your terminal

```shell
git clone https://github.com/nicolaipalm/ode.git
cd ode/
```

Activate your favourite virtual environment and run 
```shell
pip install -r requirements.txt
```

In order to set up a conda environment (recommended when working in jupyter notebooks) type in your terminal
```shell
conda create -n ode
conda activate ode
conda install pip
pip install -r requirements.txt
```

## Getting started

The easiest way in order to get familiar with this project is by checking out the notebooks.
Navigate to the notebooks sub-directory and follow the instructions given in the README.md.

Alternatively you can run the aging_model_naumann_pipeline.py file in the ./pipeline/aging_models subdirectory your favourite IDE.


## Versioning
TBA



## Tests
TBA



## Licensing
TBA


