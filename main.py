import mlflow
import hydra
import os
from omegaconf import DictConfig

os.environ['HYDRA_FULL_ERROR'] = '1'

@hydra.main(version_base=None, config_path='.', config_name='config')
def workflow(cfg: DictConfig):

    with mlflow.start_run() as active_run:
        
        mlflow.projects.run('src/make_dataset', 'main', 
                   run_name='make_dataset',
                   parameters={
                       "n_classes": cfg['make_dataset']['n_classes'],
                       "img_output_path": cfg['make_dataset']['img_output_path']
                   })
        
                
        # mlflow.projects.run('src/make_dataset', 'main', 
        #            run_name='make_dataset',
        #            parameters={
        #                "n_classes": cfg['make_dataset']['n_classes'],
        #                "img_output_path": cfg['make_dataset']['img_output_path']
        #            })  

if __name__ == "__main__":
    workflow() 