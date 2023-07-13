import pytest
from your_module_name.utility.experiments import greeting, create_experiment_folder, delete_experiment_folder
import time

def test_greeting():
    assert greeting("world") == "Hello world!"
    
def test_create_experiment_folder():
    assert create_experiment_folder('hello', 'world', {}) == f'{time.strftime("%Y-%m-%d")}_hello_world_0'
    
def test_delete_experiment_folder():
    assert delete_experiment_folder(f'{time.strftime("%Y-%m-%d")}_hello_world_0', force=True) == 1