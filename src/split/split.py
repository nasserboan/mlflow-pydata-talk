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
    
    logger.info('Creating the target dataframe.')
    index_train, index_test = (
        pd.DataFrame(glob.glob(os.path.join(args.img_input_path,
                                            args.img_extension)),
                     columns=['file_name'])
        .assign(target = lambda x: 
            x.file_name.apply(
                lambda z: int(z.split('_')[-1].replace('.jpg', ''))
                )
            )
        .pipe((train_test_split),test_size=args.test_pct, 
              random_state=args.split_random_state)
    )
    
    logger.info('Saving and logging the indexes dataframes.')
    index_train.to_csv('../../data/indexes/index_train.csv', index=False)
    index_test.to_csv('../../data/indexes/index_test.csv', index=False)
    mlflow.log_table(index_train, 'mlflow_train_index.csv')
    mlflow.log_table(index_test, 'mlflow_test_index.csv')
    
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
