import os
import cv2
import numpy as np
import argparse
import logging
from sklearn.datasets import load_digits
import mlflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def go(args):
    
    logger.info('Starting the dataset creation.')
    raw_images = load_digits(n_class=args.n_classes, as_frame=True)
    
    make_dataset_log = {'n_images_created':0,
                        'count_classes':np.zeros(args.n_classes)}
    
    logger.info(f'Creating the dataset with {args.n_classes} \
        classes and saving on {args.img_output_path}')
    for idx,img,target in zip(range(len(raw_images['images'])),
                                raw_images['images'],
                                raw_images['target']):
    
        image_path = os.path.join(args.img_output_path,
                                  f'img_{idx}_{target}.jpg')
        cv2.imwrite(image_path, img)
        make_dataset_log['n_images_created'] += 1
        make_dataset_log['count_classes'][target] += 1
        
    logger.info(f'Logging params into MLFlow.')
    mlflow.log_params(make_dataset_log)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Dataset creation, please provide the \
        number of classes and the images output path.')
    
    parser.add_argument(
        "--n_classes",
        type = int,
        help = 'The number of classes that the module must create.',
        required= True
    )
    
    parser.add_argument(
        "--img_output_path",
        type = str,
        help = 'Path where the images will be saved.',
        required= True
    )
    
    args = parser.parse_args()
    
    go(args)
