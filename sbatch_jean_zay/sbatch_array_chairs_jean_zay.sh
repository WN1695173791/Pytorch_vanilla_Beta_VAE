#!/bin/bash

#SBATCH --job-name="Chairs_L3"
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=39:00:00
#SBATCH --gres=gpu:2                # nombre de GPU a reserver
#SBATCH --cpus-per-task=3          # nombre de coeurs CPU par tache (un quart du noeud ici)

#SBATCH --output=chairs_L3_%A_%a.out       # STDOUT
#SBATCH --error="chairs_L3_%A_%a.err"       # STDERR

#SBATCH --hint=nomultithread

#BATCH --mail-type=ALL
#SBATCH --mail-user=julien.dejasmin@lis-lab.fr

#SBATCH --array=32
echo "$SLURM_ARRAY_TASK_ID"

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load pytorch-gpu/py3/1.4.0   # jean-zay

# echo des commandes lancees
set -x

LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p parameters_combinations/chairs_expes.txt)
echo $LINE

python3 -u main.py $LINE

echo "All Done!"
wait      # Wait for the end of the "child" processes (Steps) before finishing the parent process (Job).

