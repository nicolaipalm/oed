
# Notebooks; 
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)

## Installing 

Jupyter needs to be installed in you virtual environment. 
It is preferable to work with conda when working with jupyter notebooks

Run in your terminal

```shell
conda activate oed
conda install jupyter
```

When working frequently with jupyter notebooks it is preferable to install nbextensions.
You can install and use them via

```shell
conda install -c conda-forge jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextensions_configurator enable --user
```

## Getting started

Type in your terminal

```shell
conda activate oed
jupyter notebook
```
navigate to the notebooks sub-directory and open and run the paper.ipynb or template.ipynb notebook.


