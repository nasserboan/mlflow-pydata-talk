import mlflow
import os

def go():
    
    ## data step
    _ = mlflow.run(
        os.path.join(os.getcwd(), 'src', 'data'),
        'main'
    )
    
    ## features step
    _ = mlflow.run(
        os.path.join(os.getcwd(), 'src', 'features'),
        'main'
    )
    
    ## model step
    _ = mlflow.run(
        os.path.join(os.getcwd(), 'src', 'model_predict'),
        'main'
    )
    
    ## train_model step
    _ = mlflow.run(
        os.path.join(os.getcwd(), 'src', 'train_model'),
        'main'
    )
    
    ## visualization step
    _ = mlflow.run(
        os.path.join(os.getcwd(), 'src', 'visualization'),
        'main'
    )

if __name__ == "__main__":
    go() 