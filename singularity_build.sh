export PATH=/usr/users/kcohrs/center-surround/docker/:$PATH
source ~/.bashrc
GPU=0 dockerrun --env-file env --jupyterport 9999 -name base_container eckerlabdocker/docker:cuda11.0-py3.8-torch1.7-tf2.4
sudo singularity build base_container.sif docker-daemon://local/base_container