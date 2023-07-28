import os
import pandas as pd
import argparse
import logging
import glob
from sklearn.model_selection import train_test_split
import mlflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def go(args):
    
    ## load the train dataset
    
    ## define a way to dinamically define the model architecture
    ## and train it
    
    ## log the training into mlflow
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Split dataset, please provide the image input path, \
        image extension to be looked for, the test size (in pct) and the split random state.')
    
    parser.add_argument(
        "--img_input_path",
        type = str,
        help = "Path where we can find the images.",
        required= True
    )
    
    parser.add_argument(
        "--img_extension",
        type = str,
        help = "Image extension to be used.",
        required= True
    )
    
    parser.add_argument(
        "--test_pct",
        type = float,
        help = "Test size in percentage.",
        required= True
    )
    
    parser.add_argument(
        "--split_random_state",
        type = int,
        help = "Sklearn's train_test_split random_state paramenter.",
        required= True
    )
    
    args = parser.parse_args()
    
    go(args)
