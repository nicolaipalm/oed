<img src="./images/pi_design.jpeg" alt="Logo of the project" align="right" width="200">

# Optimal Design of Experiment &middot; [![Build Status](https://img.shields.io/travis/npm/npm/latest.svg?style=flat-square)](https://travis-ci.org/npm/npm) [![npm](https://img.shields.io/npm/v/npm.svg?style=flat-square)](https://www.npmjs.com/package/npm) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://github.com/your/your-project/blob/master/LICENSE)
> Optimal design of experiment, machine learning, Fisher information

This project provides
- convenient interfaces for designs of experiments, metrics to evaluate them, functions and statistical models
- a library for statistical models, designs of experimentsm and metrics
- in particular the pi-design
- a pipeline for benchmarking and comparing different designs for different scenarios
- jupyter notebooks to easily run and learn how to use this projects content

This project is closely related to (the paper) where the background and mathematical framework used here is explained in detail.

## Installing 

Run in your terminal

```shell
git clone https://github.com/nicolaipalm/ode.git
cd ode/
conda create -n ode
conda activate ode
conda install pip
pip install -r requirements.txt
```
to use conda environments and run the jupyter notebooks. 

## Getting started

The easiest way in order to get familiar with this projects is by checking out the notebooks.
Type in your terminal

```shell
conda activate ode
jupyter notebook
```
navigate to the notebooks sub-directory and open and run the paper.ipynb or template.ipynb notebook.

## Developing

### Built With
List main libraries, frameworks used including versions (React, Angular etc...)

### Prerequisites
What is needed to set up the dev environment. For instance, global dependencies or any other tools. include download links.


### Setting up Dev

Here's a brief intro about what a developer must do in order to start developing
the project further:

```shell
git clone https://github.com/your/your-project.git
cd doe/
pip install
```

And state what happens step-by-step. If there is any virtual environment, local server or database feeder needed, explain here.

### Building

If your project needs some additional steps for the developer to build the
project after some code changes, state them here. for example:

```shell
./configure
make
make install
```

Here again you should state what actually happens when the code above gets
executed.

### Deploying / Publishing
give instructions on how to build and release a new version
In case there's some step you have to take that publishes this project to a
server, this is the right time to state it.

```shell
packagemanager deploy your-project -s server.com -u username -p password
```

And again you'd need to tell what the previous code actually does.

## Versioning

We can maybe use [SemVer](http://semver.org/) for versioning. For the versions available, see the [link to tags on this repository](/tags).


## Configuration

Here you should write what are all of the configurations a user can enter when using the project.

## Tests

Describe and show how to run the tests with code examples.
Explain what these tests test and why.

```shell
Give an example
```

## Style guide

Explain your code style and show how to check it.


## Database

Explaining what database (and version) has been used. Provide download links.
Documents your database design and schemas, relations etc...

## Licensing

State what the license is and how to find the text version of the license.
