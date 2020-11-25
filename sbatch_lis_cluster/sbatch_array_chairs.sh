#!/bin/bash
#SBATCH --job-name="Chairs_L3"      # Job Name
#SBATCH --partition=gpu                  # Name of the Slurm partition used
#SBATCH --gres=gpu:2                # nombre de GPU a reserver
#SBATCH --time=00:30:00                  # time (DD-HH:MM)
#SBATCH --output=chairs_L3_%A_%a.out       # STDOUT
#SBATCH --error="chairs_L3_%A_%a.err"       # STDERR

#BATCH --mail-type=ALL                  # Mail notification of the events concerning the job : start time, end time,?~@?
#SBATCH --mail-user=julien.dejasmin@lis-lab.fr

#SBATCH --array=1
echo "$SLURM_ARRAY_TASK_ID"

LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p parameters_combinations/chairs_expes.txt)
echo $LINE

/data1/home/julien.dejasmin/anaconda3/envs/pytorch/bin/python -u main.py $LINE

echo "All Done!"
wait      # Wait for the end of the "child" processes (Steps) before finishing the parent process (Job).
