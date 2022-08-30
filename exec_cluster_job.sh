#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=2         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=2-00:00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu		    # Partition to submit to
#SBATCH --gres=gpu:rtx5000:1    		# optionally type and number of gpus
#SBATCH --mem=20G                # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=slurm_logs/%j.log  # File to which STDOUT will be written
#SBATCH --error=slurm_logs/%j.log   # File to which STDERR will be written
#SBATCH --mail-type=FAIL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=max.burg@uni-goettingen.de   # Email to which notifications will be sent
#SBATCH -C local

# print info about current job
scontrol show job "$SLURM_JOB_ID"

# load modules
module load python/3.8.6
module load cudnn/8.0.4.30-11.1-linux-x64
module load cuda/11.1.0
module load singularity

# copy data to local storage, create temp dir
echo "Copying ./data/ivan-celltype-clustering/merged_data to $TMP_LOCAL/data/ivan-celltype-clustering/merged_data"
mkdir -p "$TMP_LOCAL"/data/ivan-celltype-clustering
cp -r ./data/ivan-celltype-clustering/merged_data "$TMP_LOCAL"/data/ivan-celltype-clustering/merged_data
mkdir "$TMP_LOCAL"/data/temp

# random sleep between 1-300 seconds reduces probability of issues with network / dj connections etc.
sleep "$(shuf -i 1-300 -n 1)"

echo "Searching random free port to create ssh proxy tunnel to connect to datajoint database"
read -r lower_port upper_port < /proc/sys/net/ipv4/ip_local_port_range
while true; do
    port=$(shuf -i "$lower_port"-"$upper_port" -n 1)
    ss -tulpn | grep :"$port" > /dev/null || break
done

echo "Create ssh proxy tunnel using port:"
echo "$port"
ssh -f -4 -N -L "$port":134.76.19.44:3306 gwdu103

export HTTP_PROXY=http://www-cache.gwdg.de:3128
export HTTS_PROXY=http://www-cache.gwdg.de:3128

# run command in singularity container
echo "Executing singularity"
singularity exec -p --nv --contain --pwd /projects/controversial-stimuli --env DJ_PORT="$port" --env-file /usr/users/burg/.env_cluster --bind /usr/users/burg/projects/controversial-stimuli:/projects/controversial-stimuli,"$TMP_LOCAL"/data/:/data controversial_stimuli.sif "$@"

# following line only for testing purposes without mounting /data .
# singularity exec -p --nv --contain --pwd /projects/controversial-stimuli --env DJ_PORT=$port --env-file /usr/users/burg/.env_cluster --bind /usr/users/burg/projects/controversial-stimuli:/projects/controversial-stimuli controversial_stimuli.sif $@