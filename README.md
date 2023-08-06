mlflow-pydata-talk
==============================

Code for the PyData BSB's talk on MLFlow.

[Slides - PT-BR](https://docs.google.com/presentation/d/e/2PACX-1vSijerP5cgyugI1JjgJOkvEIFezqx8jErHxHjc4viNVpSJCrI7fG6aWOR5G9PnHRSeivRaLQr_oxg_s/pub?start=true&loop=true&delayms=5000)


## Running

```bash
$ mlflow run . --experiment-name your_experiment_name
```

## Tools used

* MLFlow
* PyTorch
* Hydra
* Scikit-Learn
* Argparse
* Pandas
* Matplotlib

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
    │   ├── split (WIP)        <- Scripts to split and prepare data.
    │   │   │
    │   │   ├── env.yml
    │   │   ├── MLProject    
    │   │   └── split_and_prepare.py
    │   │
    │   ├── train_model (WIP)  <- Scripts to train a model.
    │   │   │
    │   │   ├── env.yml
    │   │   ├── MLProject    
    │   │   └── make_dataset.py
    │   │
    │   └── batch_pred (WIP)   <- Scripts to batch predict and evaluate models.
    │       │
    │       ├── env.yml
    │       ├── MLProject    
    │       └── make_dataset.py
    │
    ├── conda.yml              <- Conda environment for the root project.
    ├── config.yaml            <- Config file with parameters to be imported by Hydra.
    ├── main.py                <- Parent project to run all the other projects inside src.
    └── MLProject              <- MLFlow project definition.

--------
