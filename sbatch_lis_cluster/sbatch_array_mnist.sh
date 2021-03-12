#!/bin/bash
#SBATCH --job-name="structural_z"      # Job Name
#SBATCH --partition=gpu                  # Name of the Slurm partition used
#SBATCH --gres=gpu:1                # nombre de GPU a reserver
#SBATCH --time=00:30:00                # time (DD-HH:MM)
#SBATCH --output=output_exp/lis_cluster/mnist_%A_%a.out       # STDOUT
#SBATCH --error=output_exp/lis_cluster/mnist_%A_%a.err       # STDERR

#BATCH --mail-type=ALL                  # Mail notification of the events concerning the job : start time, end time,?~@?
#SBATCH --mail-user=julien.dejasmin@lis-lab.fr

#SBATCH --array=2-5  # % for run n jobs in same time

echo "$SLURM_ARRAY_TASK_ID"

LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p parameters_combinations/mnist_classifier_ratio.txt)
echo $LINE

/data1/home/julien.dejasmin/anaconda3/envs/pytorch/bin/python -u main.py $LINE

echo "All Done!"
wait      # Wait for the end of the "child" processes (Steps) before finishing the parent process (Job).
