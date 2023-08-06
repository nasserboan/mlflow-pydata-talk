import mlflow
import hydra
import os
from omegaconf import DictConfig

os.environ['HYDRA_FULL_ERROR'] = '1'
run_steps = ['make_dataset','split','train_model']

@hydra.main(version_base=None, config_path='.', config_name='config')
def workflow(cfg: DictConfig):

    with mlflow.start_run() as active_run:

        if 'make_dataset' in run_steps:
            mlflow.projects.run('src/make_dataset', 'make_main', 
                    run_name='make_dataset',
                    parameters={
                        "n_classes": cfg['make_dataset']['n_classes'],
                        "img_output_path": cfg['make_dataset']['img_output_path']
                    })
        
        if 'split' in run_steps:       
            mlflow.projects.run('src/split', 'split_main', 
                    run_name='split_dataset',
                    parameters={
                        "img_input_path": cfg['split']['img_input_path'],
                        "img_extension": cfg['split']['img_extension'],
                        "test_pct": cfg['split']['test_pct'],
                        "split_random_state": cfg['split']['split_random_state']
                    })
        
        if 'train_model' in run_steps:
            

if __name__ == "__main__":
    
    TRACKING_URI = os.path.join('file://', os.getcwd(), 'mlruns')
    print(TRACKING_URI)
    mlflow.set_tracking_uri(TRACKING_URI)
    
    workflow() 