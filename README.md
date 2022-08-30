# nn-repo-template
Amazing neural network project repository.

## Execute project code

- Download Singularity image and put it into project folder: https://nc.mlcloud.uni-tuebingen.de/index.php/s/c4gkPjmfktTBf4r (at commit a2706443da081ca3d986e2003e265ef79a9faa92)
- Adapt paths etc. in `exec_dropbear_40022.sh`
- Run `exec_dropbear_40022.sh`
- Connect into the running Singularity container by `ssh -p40022 localhost`
- In the container, move to the home directory, and execute `start_jupyter_lab.sh`
- On your local machine, create an ssh portforwarding to the Jupyter Lab port on the remote machine, e.g. `ssh -f burg@hydra.cidas.uni-goettingen.de -L 9999:localhost:8888 -N`. Then, Jupyter Lab can be accessed by typing `http://localhost:9999/` in a web browser on your local machine.


## External project dependencies

<https://github.com/sinzlab/nnfabrik> (sinzlab master)
<https://github.com/KonstantinWilleke/neuralpredictors/tree/readout_position_regularizer>
<https://github.com/KonstantinWilleke/nnvision>  (Konsti master)
<https://github.com/KonstantinWilleke/nndichromacy> (Konsti master)
<https://github.com/sinzlab/mei/tree/konsti_monkey_experiments>
<https://github.com/sinzlab/data_port>
<https://github.com/sacadena/ptrnets> (Santi master)

All repos are forked into my github account with small bugfixes.


## Commit codes

The commit message is `[commit code] body of commit message` and a commit can only have one commit code, e.g. a commit should not contain both adding a new feature and fixing a bug in another. Split these into two separate commits.

List of commit codes:

[initialize] - initial commit
[add] - add a new feature
[delete] - delete features
[refactor] - refactor some code
[fix] - fix a bug
[merge] - resolve merge conflicts / merge of branches