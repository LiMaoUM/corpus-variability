#!/bin/bash
#SBATCH --nodes=1          # Use 1 Node     (Unless code is multi-node parallelized)
#SBATCH --ntasks=1
#SBATCH --account=fconrad0
#SBATCH --time=3:20:00
#SBATCH --cpus-per-task=3
#SBATCH -o slurm-%j.out-%N
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=180000m
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maolee@umich.edu   # Your email address has to be set accordingly
#SBATCH --job-name=censusSocialMedia        # the job's name you want to be used

module load python3.10-anaconda

export FILENAME=/nfs/turbo/isr-fconrad1/projects/corpus-variability/src/summaries_generate.py
srun python $FILENAME > SLURM_JOBID.out

echo "End of program at `date`"
