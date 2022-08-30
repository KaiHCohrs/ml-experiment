ARG BASE_CONTAINER=eckerlabdocker/docker-stacks:cuda11.0-cudnn8-python3.8
FROM $BASE_CONTAINER

LABEL maintainer="Kai-Hendrik Cohrs <kaicohrs@uv.es>"

# switch to root for installing software
USER root

# ---- install apt packages -----

RUN apt-get update -qq \
 && DEBIAN_FRONTEND=noninteractive apt-get install -yq -qq --no-install-recommends \
  git \
  htop \
  make \
  python3-dev \
  unzip \
  vim \
  zlib1g \
  zlib1g-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# ---- install python packages -----

RUN conda install \
        # jupyter_nbextensions_configurator \
        fastprogress \
        gitpython \
        h5py \
        ipyparallel \
        ipywidgets \
        jsonschema \
        numpy \
        pandas \
        pillow \
        seaborn \
        scikit-learn \
        scikit-image \
        scipy \
        tensorboard \
        tqdm \
        # added libraries \
        
 && conda clean -tipsy \
 && fix-permissions $CONDA_DIR \
 && fix-permissions /home/$NB_USER


# Install local dependencies / git submodules
ADD . /projects/nn-repo-template
WORKDIR /projects
RUN pip install --no-cache-dir -e nn-repo-template

RUN pip install --no-cache-dir tensorflow-gpu \
 && fix-permissions $CONDA_DIR \
 && fix-permissions /home/$NB_USER

RUN pip install --no-cache-dir torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html \
 && fix-permissions $CONDA_DIR \
 && fix-permissions /home/$NB_USER

RUN pip install --no-cache-dir datajoint==0.12.4 \
 && fix-permissions $CONDA_DIR \
 && fix-permissions /home/$NB_USER

RUN rm /usr/bin/python3 && ln -s /opt/conda/bin/python

# switch back to default user (jovyan)
USER $NB_USER

