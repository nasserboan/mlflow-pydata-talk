MLFlow Project: Automated Code Execution and Logging
==============================

[![MLFlow Version](https://img.shields.io/badge/MLFlow-2.5.0-brightgreen)](https://mlflow.org/)
[![Python Version](https://img.shields.io/badge/Python-3.10.12-blue)](https://www.python.org/)

This repository contains a project that was showcased in the PyData Brasília talk about MLFlow. The project demonstrates the use of MLFlow Projects to execute a series of code scripts in a specific order while maintaining comprehensive logging of each run. The goal of this project is to provide an efficient and organized way to manage and monitor your machine learning workflows.

[Slides - PT-BR](https://docs.google.com/presentation/d/e/2PACX-1vSijerP5cgyugI1JjgJOkvEIFezqx8jErHxHjc4viNVpSJCrI7fG6aWOR5G9PnHRSeivRaLQr_oxg_s/pub?start=true&loop=true&delayms=5000)

## Features

- **Code Execution Order:** The project leverages MLFlow Projects to run a series of code scripts in a defined order. This is particularly useful when you have multiple interdependent scripts that need to be executed in a specific sequence.
- **Logging and Tracking:** MLFlow's logging capabilities allow you to keep track of important metrics, parameters, and artifacts produced during each run. This ensures that you have a comprehensive record of the entire workflow.
- **Reproducibility:** By using MLFlow Projects, you can ensure that your code runs consistently across different environments. This greatly aids in reproducing results and collaborating with other team members.

## Getting Started

1. **Clone Repository:** Clone this repository to your local machine:

   ```bash
   git clone https://github.com/nasserboan/mlflow-pydata-talk
   cd mlflow-pydata-talk
   ```

2. **Create Environment:** Set up a virtual environment and install any necessary dependencies for your project:

   ```bash
   conda env create -f conda.yml
   ```

3. **Define which steps should be run:** Open the <code>main.py</code> and define which steps should be run by altering the <code>run_steps</code> list.

4. **Run Project:** Execute the project using MLFlow:

   ```bash
   mlflow run . --experiment-name <your-experiment-name>
   ```

5. **View Results:** Check the MLFlow UI to view the logged metrics, parameters, and artifacts from each run:

   ```bash
   mlflow ui
   ```


## Tools used

* MLFlow
* PyTorch
* Hydra
* Scikit-Learn
* Argparse
* Pandas

Project Organization
------------

    ├── LICENSE
    ├── README.md              <- The top-level README.
    ├── data
    │   ├── indexes            <- Indexes of the images that will be used for training and testing
    │   ├── processed          <- The final, canonical data sets for modeling.
    │   └── raw                <- The original, immutable data dump.
    │
    ├── notebooks              <- Jupyter notebooks.
    │
    ├── mlruns                 <- Metada from MLFlow experiments.
    │
    ├── src                    <- Source code for use in this project.
    │   │
    │   ├── make_dataset       <- Scripts to generate data.
    │   │   │
    │   │   ├── env.yml
    │   │   ├── MLProject    
    │   │   └── make_dataset.py
    │   │
    │   ├── split              <- Scripts to split and prepare data.
    │   │   │
    │   │   ├── env.yml
    │   │   ├── MLProject    
    │   │   └── split_and_prepare.py
    │   │
    │   └── train_model        <- Scripts to train a model.
    │       │
    │       ├── env.yml
    │       ├── MLProject    
    │       └── train_model.py
    │
    │
    ├── conda.yml              <- Conda environment for the root project.
    ├── config.yaml            <- Config file with parameters to be imported by Hydra.
    ├── main.py                <- Parent project to run all the other projects inside src.
    └── MLProject              <- MLFlow project definition.

--------
