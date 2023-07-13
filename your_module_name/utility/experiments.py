import os
from pathlib import Path
import time
import yaml
import shutil

def greeting(name):
    return f"Hello {name}!"

def create_experiment_folder(experiment_type, arguments, config):

    # Generate the results folder in the format "YYYY-MM-DD_{experiment_type}_{arguments}_{i}"
    results_path = Path(__file__).parent.parent.parent.joinpath('results')        
    experiment_name = f'{time.strftime("%Y-%m-%d")}_{experiment_type}_{arguments}'
    
    i = 0
    while os.path.exists(results_path.joinpath(experiment_name + f"_{i}")):
        i += 1
    experiment_name = experiment_name + f"_{i}"  
    experiment_path = results_path.joinpath(experiment_name)
    os.makedirs(experiment_path)
    
    config['git_hash'] = os.popen('git rev-parse HEAD').read().strip()
    
    with open(experiment_path.joinpath('config.yml'), 'w') as file:
        yaml.dump(config, file)

    return experiment_name

def delete_experiment_folder(experiment_name, force=False):
    
    results_path = Path(__file__).parent.parent.parent.joinpath('results')        
    experiment_path = results_path.joinpath(experiment_name)
    
    if not force:
        if input(f"Are you sure you want to delete {experiment_path}? (y/n)") == 'y':
            shutil.rmtree(experiment_path)
    else:
        shutil.rmtree(experiment_path)
    
    return 1
