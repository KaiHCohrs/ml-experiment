#!/bin/bash

# Ask for new repo name
#read -p "Enter new repo name: " new_repo_name

# Rename directory
#mv new_name/ "${new_repo_name}/"

# Iterate over all Python files in the module and the script files and replace "your_module_name" with the new repo you have chosen
#your_module_name
#find "${new_repo_name}/" -type f -name "*.py" -print0 | xargs -0 sed -i "s/new_name/${new_repo_name}/g"
#find "scripts/" -type f -name "*.py" -exec sed -i "s/new_name/${new_repo_name}/g" {} \;

# Substitute the name in setup.py file
#sed -i "s/new_name/${new_repo_name}/g" "setup.py"

# Build the conda environment from environment.yml
#mamba env create -f environment.yml

# Activate the conda environment
#conda activate env-name

# Install the module in editable mode
#pip install -e .

# Create and switch to development branch
#git checkout -b dev

# Add all changes to the staging area
#git add -A

# Create an initial commit with the specified message
#git commit -m "-initial commit"

# Push the changes to the remote repository
#git push

# Run hello world unit test
python -m pytest tests/test_basic.py

# All setup
echo "You are all setup! Let's delete this file."

# Delete this file
#rm -- "$0"