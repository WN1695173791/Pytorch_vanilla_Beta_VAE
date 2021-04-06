#!/bin/bash
#SBATCH --job-name="VAE_test"       # Job Name
#SBATCH --partition=gpu_p2             # partition name
#SBATCH --qos=qos_gpu-t4             # for jean-zay
#SBATCH --gres=gpu:1                # nombre de GPU a reserver
#SBATCH --cpus-per-task=3          # nombre de coeurs CPU par tache
#SBATCH --time=01:30:00                  # time (DD-HH:MM)
#SBATCH --output=output_exp/jean_zay/mnist_%A_%a.out       # STDOUT
#SBATCH --error=output_exp/jean_zay/mnist_%A_%a.err       # STDERR

# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread        # hyperthreading desactive

#BATCH --mail-type=ALL           
#SBATCH --mail-user=julien.dejasmin@lis-lab.fr

#SBATCH --array=2-81
echo "$SLURM_ARRAY_TASK_ID"

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load pytorch-gpu/py3/1.6.0   # jean-zay

# echo des commandes lancees
set -x

LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p parameters_combinations/mnist_classifier_ratio.txt)
echo $LINE

python3 -u main.py $LINE

wait      # Wait for the end of the "child" processes (Steps) before finishing the parent process (Job).
echo "All done !"
